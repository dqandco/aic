#!/usr/bin/env python3
"""Parallel per-episode ground-truth recording across N workers.

Distributes trials from a multi-trial config across N independent
``record_trials_v2_simrestart`` workers. Each worker gets isolated:

  - Zenoh router (port ``zenoh-base-port + k``)
  - ``GZ_PARTITION`` (``aic_worker_{k}``)
  - ``ROS_DOMAIN_ID`` (``k``)
  - dataset (``{repo_id}_w{k}``)
  - work-dir (``{work_dir}/worker_{k}``)
  - ``TMPDIR`` (``{work_dir}/worker_{k}/tmp``)

After all workers finish, per-worker datasets are merged into the final
dataset at ``dataset-repo-id``. Use ``--keep-worker-datasets`` to preserve
them for debugging.

Usage::

    HF_USER=youruser pixi run python \\
        ground_truth_collection/record_trials_v2_parallel.py \\
        --config ground_truth_collection/my_config.yaml \\
        --policy my_policy.ExpertCollector \\
        --num-workers 4 \\
        --work-dir /tmp/gt_parallel
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2  # noqa: F401  — libtiff symbol workaround (import before lerobot)
import yaml

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [parallel] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
WORKER_SCRIPT = SCRIPT_DIR / "record_trials_v2_simrestart.py"

_workers: list[subprocess.Popen] = []


def _on_signal(signum, _frame):
    log.info("Received signal %d — forwarding to %d workers.", signum, len(_workers))
    for p in _workers:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass


def count_trials(config_path: Path) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return len(cfg.get("trials", {}) or {})


def chunk_indices(start: int, end: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split [start, end) into up to n_chunks roughly-equal contiguous ranges."""
    total = end - start
    if total <= 0:
        return []
    n_chunks = min(n_chunks, total)
    base, rem = divmod(total, n_chunks)
    ranges = []
    s = start
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        ranges.append((s, s + size))
        s += size
    return ranges


def spawn_worker(
    worker_id: int,
    chunk: tuple[int, int],
    args: argparse.Namespace,
    worker_dir: Path,
    repo_id: str,
) -> subprocess.Popen:
    tmpdir = worker_dir / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    partition = f"aic_worker_{worker_id}"
    zenoh_port = args.zenoh_base_port + worker_id

    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(worker_id)
    env["GZ_PARTITION"] = partition
    env["TMPDIR"] = str(tmpdir)

    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        "--worker-id", str(worker_id),
        "--zenoh-port", str(zenoh_port),
        "--gz-partition", partition,
        "--config", str(args.config),
        "--policy", args.policy,
        "--dataset-repo-id", repo_id,
        "--work-dir", str(worker_dir),
        "--episode-timeout", str(args.episode_timeout),
        "--start-idx", str(chunk[0]),
        "--end-idx", str(chunk[1]),
        "--force-overwrite",
    ]
    if args.resume:
        cmd.append("--resume")

    log_path = worker_dir / "worker.log"
    log.info(
        "Worker %d: trials [%d,%d)  zenoh=%d  partition=%s  dataset=%s",
        worker_id, chunk[0], chunk[1], zenoh_port, partition, repo_id,
    )
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(SCRIPT_DIR.parent),
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _read_progress(work_dir: Path, k: int) -> tuple[int, int]:
    """Return (completed_count, failed_count) for worker k; (0, 0) if missing."""
    p = work_dir / f"worker_{k}" / "progress.json"
    if not p.exists():
        return 0, 0
    try:
        with open(p) as f:
            data = json.load(f)
        return len(data.get("completed", [])), len(data.get("failed", []))
    except (OSError, json.JSONDecodeError):
        return 0, 0


def log_progress(
    work_dir: Path, chunks: list[tuple[int, int]], start_time: float,
):
    parts = []
    total_done = 0
    total_failed = 0
    total_size = 0
    for k, (s, e) in enumerate(chunks):
        size = e - s
        done, failed = _read_progress(work_dir, k)
        total_done += done
        total_failed += failed
        total_size += size
        part = f"w{k}:{done}/{size}"
        if failed:
            part += f"(F{failed})"
        parts.append(part)
    pct = (100.0 * total_done / total_size) if total_size else 0.0
    failed_str = f" F{total_failed}" if total_failed else ""
    log.info(
        "progress: %s  |  total %d/%d (%.0f%%)%s  |  elapsed %s",
        "  ".join(parts), total_done, total_size, pct, failed_str,
        _fmt_elapsed(time.monotonic() - start_time),
    )


def monitor_progress(
    work_dir: Path, chunks: list[tuple[int, int]],
    stop_event: threading.Event, interval: float,
):
    start = time.monotonic()
    while not stop_event.wait(interval):
        try:
            log_progress(work_dir, chunks, start)
        except Exception:
            log.exception("progress monitor error")


def parse_args() -> argparse.Namespace:
    hf_user = os.environ.get("HF_USER")
    p = argparse.ArgumentParser(
        description="Parallel ground-truth recording (wraps record_trials_v2_simrestart)."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--policy", type=str, required=True)
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument(
        "--dataset-repo-id",
        default=f"{hf_user}/aic_expert_demos" if hf_user else None,
        help="Final merged dataset repo id (default: $HF_USER/aic_expert_demos).",
    )
    p.add_argument("--work-dir", type=Path, default=Path("recording_work_parallel"))
    p.add_argument("--episode-timeout", type=float, default=300)
    p.add_argument("--start-idx", type=int, default=None)
    p.add_argument("--end-idx", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--keep-worker-datasets", action="store_true")
    p.add_argument(
        "--zenoh-base-port", type=int, default=7447,
        help="Worker k listens on zenoh-base-port + k (default: 7447).",
    )
    p.add_argument(
        "--stagger-secs", type=float, default=3.0,
        help="Delay between worker spawns (default: 3s) to avoid startup races.",
    )
    p.add_argument(
        "--progress-interval", type=float, default=30.0,
        help="How often to log an aggregate progress line (default: 30s; 0 to disable).",
    )
    args = p.parse_args()
    if not args.dataset_repo_id:
        p.error("Set HF_USER env var or pass --dataset-repo-id")
    if not args.config.exists():
        p.error(f"Config file not found: {args.config}")
    if args.num_workers < 1:
        p.error("--num-workers must be >= 1")
    args.config = args.config.resolve()
    args.work_dir = args.work_dir.resolve()
    return args


def load_worker_datasets(repo_ids: list[str]) -> list[LeRobotDataset]:
    datasets = []
    for k, repo_id in enumerate(repo_ids):
        ds_dir = HF_LEROBOT_HOME / repo_id
        if not ds_dir.exists():
            log.warning("Worker %d: no dataset at %s — skipping.", k, ds_dir)
            continue
        try:
            ds = LeRobotDataset(repo_id)
        except Exception:
            log.exception("Worker %d: failed to load dataset — skipping.", k)
            continue
        n_frames = len(ds)
        if n_frames == 0:
            log.warning("Worker %d: dataset is empty — skipping.", k)
            continue
        log.info("Worker %d: %d frames loaded.", k, n_frames)
        datasets.append(ds)
    return datasets


def main():
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    args = parse_args()

    n_trials = count_trials(args.config)
    if n_trials == 0:
        log.error("No trials found in config %s", args.config)
        sys.exit(1)

    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else n_trials
    start = max(0, min(start, n_trials))
    end = max(start, min(end, n_trials))

    chunks = chunk_indices(start, end, args.num_workers)
    n_workers = len(chunks)
    if n_workers == 0:
        log.error("Empty trial range [%d, %d).", start, end)
        sys.exit(1)
    if n_workers < args.num_workers:
        log.info(
            "Only %d trials in range; using %d workers instead of %d.",
            end - start, n_workers, args.num_workers,
        )

    args.work_dir.mkdir(parents=True, exist_ok=True)

    repo_ids = [f"{args.dataset_repo_id}_w{k}" for k in range(n_workers)]

    # Pre-clean per-worker datasets when not resuming — avoids prompt / collision.
    if not args.resume:
        for repo_id in repo_ids:
            ds_dir = HF_LEROBOT_HOME / repo_id
            if ds_dir.exists():
                log.info("Deleting stale per-worker dataset at %s", ds_dir)
                shutil.rmtree(ds_dir)

    for k, chunk in enumerate(chunks):
        worker_dir = args.work_dir / f"worker_{k}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        proc = spawn_worker(k, chunk, args, worker_dir, repo_ids[k])
        _workers.append(proc)
        if k < n_workers - 1:
            time.sleep(args.stagger_secs)

    log.info("All %d workers spawned. Waiting for completion...", n_workers)

    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    monitor_start = time.monotonic()
    if args.progress_interval > 0:
        monitor_thread = threading.Thread(
            target=monitor_progress,
            args=(args.work_dir, chunks, stop_event, args.progress_interval),
            daemon=True,
        )
        monitor_thread.start()

    exit_codes = []
    try:
        for k, p in enumerate(_workers):
            rc = p.wait()
            exit_codes.append(rc)
            log.info("Worker %d exited rc=%d", k, rc)
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)

    log_progress(args.work_dir, chunks, monitor_start)

    # Merge
    log.info("Loading per-worker datasets for merge...")
    datasets = load_worker_datasets(repo_ids)
    if not datasets:
        log.error("No non-empty per-worker datasets — nothing to merge.")
        sys.exit(1)

    final_dir = HF_LEROBOT_HOME / args.dataset_repo_id
    if final_dir.exists():
        log.info("Removing existing merged dataset at %s", final_dir)
        shutil.rmtree(final_dir)

    log.info(
        "Merging %d dataset(s) into %s", len(datasets), args.dataset_repo_id,
    )
    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=args.dataset_repo_id,
        output_dir=final_dir,
    )
    log.info("Merged dataset has %d frames.", len(merged))

    if args.push_to_hub:
        log.info("Pushing merged dataset to Hub...")
        merged.push_to_hub(private=True)

    if not args.keep_worker_datasets:
        for repo_id in repo_ids:
            ds_dir = HF_LEROBOT_HOME / repo_id
            if ds_dir.exists():
                shutil.rmtree(ds_dir)
        log.info("Per-worker datasets cleaned up.")

    non_zero = [k for k, rc in enumerate(exit_codes) if rc != 0]
    if non_zero:
        log.warning("Workers with non-zero exit codes: %s", non_zero)
        sys.exit(1)
    log.info("Done.")


if __name__ == "__main__":
    main()
