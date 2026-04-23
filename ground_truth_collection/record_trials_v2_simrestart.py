#!/usr/bin/env python3
"""Per-episode recording with full sim restart per trial.

Same as record_trials_v2.py but restarts the entire sim (Gazebo + bringup)
between episodes.  Slower (~60-90s extra per episode) but trades that for a
clean sim state each trial — useful when diagnosing sim/TF state leaks or
when aic_engine restart against a persistent sim is unreliable.

The script manages the entire lifecycle — no need to start the sim yourself.

Usage::

    HF_USER=youruser pixi run python ground_truth_collection/record_trials_v2_simrestart.py \
        --config ground_truth_collection/my_config.yaml \
        --policy my_policy.ExpertCollector \
        --work-dir /tmp/gt_collection

Environment variables:
    HF_USER  – HuggingFace username (required unless --dataset-repo-id given)
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

import cv2  # noqa: F401  — import before lerobot to avoid libtiff symbol conflict
import numpy as np
import yaml

from aic_control_interfaces.msg import MotionUpdate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR

from lerobot_robot_aic.aic_robot_aic_controller import (  # noqa: F401
    AICRobotAICController,
    AICRobotAICControllerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RECORDING_FPS = 4
TASK_DESCRIPTION = "insert cable into port"

SIM_READINESS_SERVICE = "/aic_controller/change_target_mode"
SIM_READINESS_TIMEOUT = 120

SIM_ARGS = (
    "spawn_task_board:=false spawn_cable:=false "
    "ground_truth:=true start_aic_engine:=false"
)
SIM_CMD_ENTRYPOINT = f"/entrypoint.sh {SIM_ARGS}"
SIM_CMD_WORKER = (
    f"ros2 launch aic_bringup aic_gz_bringup.launch.py {SIM_ARGS}"
)

HOME_JOINT_POSITIONS = np.array([-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110])
HOME_JOINT_TOLERANCE = 0.01

WS_SETUP = "/ws_aic/install/setup.bash"
ZENOH_CONFIG_PATH = "/aic_zenoh_config.json5"

_shutdown_requested = False
_worker_mode = False
_gz_partition: str | None = None


def _on_signal(signum, _frame):
    global _shutdown_requested
    log.info("Received signal %d, requesting shutdown...", signum)
    _shutdown_requested = True
    if signum == signal.SIGINT:
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def launch_ros_process(cmd: list[str], log_path: Path | None = None) -> subprocess.Popen:
    """Launch a ROS 2 subprocess with the workspace overlay sourced."""
    shell_cmd = f"source {WS_SETUP} && {' '.join(cmd)}"
    kwargs: dict = dict(
        shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid,
    )
    if log_path:
        kwargs["stdout"] = log_path.open("w")
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.Popen(shell_cmd, **kwargs)


def kill_process_group(proc: subprocess.Popen | None, timeout: int = 5):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


_PID_RE = re.compile(r"pid=(\d+)")


def _preflight_port_cleanup(port: int):
    """Kill any rmw_zenohd zombie holding ``port``; abort if some other process holds it.

    Zenoh routers from prior runs can survive worker death (e.g. SIGKILL before
    the worker's ``finally`` block runs). This clears them so we can bind.
    """
    try:
        result = subprocess.run(
            ["ss", "-tlnpH"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("Skipping preflight port check (%s).", e)
        return

    port_tok = f":{port} "
    killed = False
    others: list[str] = []
    for line in result.stdout.splitlines():
        if port_tok not in line:
            continue
        if "rmw_zenohd" in line:
            for m in _PID_RE.finditer(line):
                pid = int(m.group(1))
                log.warning(
                    "Port %d: killing zombie rmw_zenohd (pid=%d).", port, pid,
                )
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed = True
                except (ProcessLookupError, PermissionError) as e:
                    log.warning("Failed to kill pid %d: %s", pid, e)
        else:
            others.append(line.strip())

    if others:
        log.error(
            "Port %d is held by non-zenoh process(es); aborting:\n  %s",
            port, "\n  ".join(others),
        )
        sys.exit(1)

    if killed:
        time.sleep(1)  # let the kernel release the port


def setup_worker_env(
    worker_id: int, zenoh_port: int, gz_partition: str | None, log_dir: Path,
) -> subprocess.Popen:
    """Set up worker isolation: Zenoh router, env vars, Gazebo partition.

    Must be called before anything triggers ``rclpy.init()`` (i.e. before
    constructing the robot). Returns the Zenoh router process for teardown.
    """
    log.info(
        "Worker %d: isolating environment (zenoh port %d, gz partition %s).",
        worker_id, zenoh_port, gz_partition or "default",
    )

    _preflight_port_cleanup(zenoh_port)

    os.environ["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"
    os.environ["ZENOH_ROUTER_CONFIG_URI"] = ZENOH_CONFIG_PATH
    os.environ["ROS_DOMAIN_ID"] = str(worker_id)
    if gz_partition:
        os.environ["GZ_PARTITION"] = gz_partition

    router_override = (
        'mode="router"'
        f';listen/endpoints=["tcp/[::]:{zenoh_port}"]'
        ';connect/endpoints=[]'
        ';routing/router/peers_failover_brokering=true'
        ';transport/shared_memory/enabled=false'
    )
    router_env = os.environ.copy()
    router_env["ZENOH_CONFIG_OVERRIDE"] = router_override

    log_path = log_dir / f"zenoh_router_w{worker_id}.log"
    router_proc = subprocess.Popen(
        f"source {WS_SETUP} && exec ros2 run rmw_zenoh_cpp rmw_zenohd",
        shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        env=router_env,
    )

    # Peer override inherited by every subsequent child of this process.
    peer_override = (
        f'connect/endpoints=["tcp/localhost:{zenoh_port}"]'
        ';transport/shared_memory/enabled=false'
    )
    os.environ["ZENOH_CONFIG_OVERRIDE"] = peer_override

    time.sleep(3)
    if router_proc.poll() is not None:
        log.error("Zenoh router crashed on startup (worker %d).", worker_id)
        sys.exit(1)

    log.info(
        "Worker %d: Zenoh router ready (pid=%d, port=%d).",
        worker_id, router_proc.pid, zenoh_port,
    )
    return router_proc


def _kill_gz_stragglers_blanket():
    for pattern in ("gz sim", "ruby.*gz", "parameter_bridge"):
        subprocess.run(
            ["pkill", "-9", "-f", pattern],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def _kill_gz_stragglers_by_partition(partition: str):
    """Kill gz/bridge stragglers whose env has GZ_PARTITION=<partition>.

    Enumerates candidate pids via pgrep, reads /proc/<pid>/environ, kills
    only those matching the worker's partition. Safe under parallel workers.
    """
    marker = f"GZ_PARTITION={partition}".encode() + b"\x00"
    for pattern in ("gz sim", "ruby.*gz", "parameter_bridge"):
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, check=False, timeout=5,
        )
        for pid_str in result.stdout.split():
            try:
                pid = int(pid_str)
                with open(f"/proc/{pid}/environ", "rb") as f:
                    env = f.read()
            except (ValueError, FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if marker in env:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def launch_sim(log_path: Path) -> subprocess.Popen:
    """Launch the Gazebo sim (entrypoint.sh for single-instance, direct launch for workers)."""
    cmd = SIM_CMD_WORKER if _worker_mode else SIM_CMD_ENTRYPOINT
    if _worker_mode:
        cmd = f"source {WS_SETUP} && exec {cmd}"
    return subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )


def kill_sim(sim_proc: subprocess.Popen | None):
    """Kill sim process group; then mop up Gazebo stragglers."""
    kill_process_group(sim_proc, timeout=10)
    if _worker_mode and _gz_partition:
        _kill_gz_stragglers_by_partition(_gz_partition)
    else:
        _kill_gz_stragglers_blanket()


# ---------------------------------------------------------------------------
# Sim readiness
# ---------------------------------------------------------------------------


def wait_for_sim(timeout: int = SIM_READINESS_TIMEOUT):
    log.info("Waiting for sim (polling %s)...", SIM_READINESS_SERVICE)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            f"source {WS_SETUP} && ros2 service list",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=10,
        )
        if SIM_READINESS_SERVICE in result.stdout:
            log.info("Sim is ready.")
            return
        time.sleep(3)
    log.error(
        "Timed out waiting for sim after %ds.\n"
        "Start the sim in another terminal:\n\n"
        "  /entrypoint.sh spawn_task_board:=false spawn_cable:=false \\\n"
        "      ground_truth:=true start_aic_engine:=false\n",
        timeout,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Action capture
# ---------------------------------------------------------------------------


class ActionCapture:
    """Subscribes to /aic_controller/pose_commands to capture actions."""

    def __init__(self):
        self._lock = Lock()
        self._latest: dict[str, float] | None = None

    def attach(self, robot: AICRobotAICController):
        self._sub = robot.ros2_interface.node.create_subscription(
            MotionUpdate, "/aic_controller/pose_commands", self._callback, 10,
        )

    def _callback(self, msg: MotionUpdate):
        action = {
            "linear.x": float(msg.velocity.linear.x),
            "linear.y": float(msg.velocity.linear.y),
            "linear.z": float(msg.velocity.linear.z),
            "angular.x": float(msg.velocity.angular.x),
            "angular.y": float(msg.velocity.angular.y),
            "angular.z": float(msg.velocity.angular.z),
        }
        with self._lock:
            self._latest = action

    @property
    def latest(self) -> dict[str, float] | None:
        with self._lock:
            return self._latest.copy() if self._latest else None

    def reset(self):
        with self._lock:
            self._latest = None


# ---------------------------------------------------------------------------
# Episode boundary helper
# ---------------------------------------------------------------------------


def joints_at_home(robot: AICRobotAICController) -> bool:
    js = robot.last_joint_states
    if js is None or len(js.position) < len(HOME_JOINT_POSITIONS):
        return False
    current = np.array(js.position[: len(HOME_JOINT_POSITIONS)])
    return bool(np.all(np.abs(current - HOME_JOINT_POSITIONS) < HOME_JOINT_TOLERANCE))


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------


def create_dataset(
    robot: AICRobotAICController, repo_id: str, resume: bool,
) -> LeRobotDataset:
    _, _, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=True,
        ),
    )
    if resume:
        ds = LeRobotDataset(repo_id)
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            ds.start_image_writer(num_processes=0, num_threads=4 * len(robot.cameras))
        return ds
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=RECORDING_FPS,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
    )


# ---------------------------------------------------------------------------
# Config splitting
# ---------------------------------------------------------------------------


def split_config(config_path: Path, work_dir: Path) -> list[Path]:
    """Split a multi-trial YAML into one YAML per trial.

    Returns list of per-trial config file paths, ordered by trial key.
    """
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    trials = full_config.get("trials", {})
    if not trials:
        log.error("No trials found in config %s", config_path)
        sys.exit(1)

    shared_keys = {k: v for k, v in full_config.items() if k != "trials"}
    configs_dir = work_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for trial_key in sorted(trials.keys(), key=lambda k: int(k.split("_")[-1])):
        single = {**shared_keys, "trials": {trial_key: trials[trial_key]}}
        out_path = configs_dir / f"{trial_key}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(single, f, default_flow_style=False, sort_keys=False)
        paths.append(out_path)

    log.info("Split %d trials into %s", len(paths), configs_dir)
    return paths


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Tracks completed/failed episodes in a JSON file for resumability."""

    def __init__(self, path: Path, config_path: str, repo_id: str):
        self._path = path
        self._config = config_path
        self._repo_id = repo_id
        if path.exists():
            with open(path) as f:
                self._data = json.load(f)
        else:
            self._data = {
                "completed": [],
                "failed": [],
                "config": config_path,
                "repo_id": repo_id,
            }
            self._save()

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_completed(self, idx: int) -> bool:
        return idx in self._data["completed"]

    def mark_completed(self, idx: int):
        if idx in self._data["failed"]:
            self._data["failed"].remove(idx)
        if idx not in self._data["completed"]:
            self._data["completed"].append(idx)
        self._save()

    def mark_failed(self, idx: int):
        if idx not in self._data["failed"] and idx not in self._data["completed"]:
            self._data["failed"].append(idx)
        self._save()

    @property
    def completed(self) -> list[int]:
        return list(self._data["completed"])

    @property
    def failed(self) -> list[int]:
        return list(self._data["failed"])


# ---------------------------------------------------------------------------
# Per-episode recording
# ---------------------------------------------------------------------------


def run_episode(
    idx: int,
    trial_config: Path,
    policy: str,
    robot: AICRobotAICController,
    dataset: LeRobotDataset,
    action_capture: ActionCapture,
    log_dir: Path,
    timeout: float,
) -> bool:
    """Run a single episode with a fresh sim.  Returns True if frames were saved."""
    global _shutdown_requested

    sim_proc = None
    model_proc = None
    engine_proc = None
    episode_frames = 0
    robot_connected = False

    try:
        # Launch sim
        log.info("Episode %d: launching sim...", idx)
        sim_proc = launch_sim(log_dir / f"sim_ep{idx}.log")
        wait_for_sim()

        # Connect robot to the freshly-started sim
        robot.connect(calibrate=False)
        robot_connected = True
        action_capture.reset()
        action_capture.attach(robot)

        # Launch aic_model
        model_proc = launch_ros_process(
            [
                "ros2", "run", "aic_model", "aic_model",
                "--ros-args",
                "-p", "use_sim_time:=true",
                "-p", f"policy:={policy}",
            ],
            log_dir / f"aic_model_ep{idx}.log",
        )
        time.sleep(5)
        if model_proc.poll() is not None:
            log.error("aic_model crashed on startup (episode %d)", idx)
            return False

        # Launch aic_engine with single-trial config
        config_path = str(trial_config.resolve())
        engine_proc = launch_ros_process(
            [
                "ros2", "run", "aic_engine", "aic_engine",
                "--ros-args",
                "-p", f"config_file_path:={config_path}",
                "-p", "use_sim_time:=true",
                "-p", "ground_truth:=true",
            ],
            log_dir / f"aic_engine_ep{idx}.log",
        )

        # Recording loop
        deadline = time.monotonic() + timeout
        log.info("Episode %d: recording (timeout=%.0fs)...", idx, timeout)

        while not _shutdown_requested:
            loop_start = time.perf_counter()

            if engine_proc.poll() is not None:
                log.info(
                    "Episode %d: aic_engine exited (rc=%d).",
                    idx, engine_proc.returncode,
                )
                break

            if model_proc.poll() is not None:
                log.warning(
                    "Episode %d: aic_model crashed (rc=%d).",
                    idx, model_proc.returncode,
                )
                break

            if sim_proc.poll() is not None:
                log.error(
                    "Episode %d: sim exited unexpectedly (rc=%d).",
                    idx, sim_proc.returncode,
                )
                break

            if time.monotonic() > deadline:
                log.warning("Episode %d: timeout reached.", idx)
                break

            # Skip frames while robot is at home (aic_engine setup phase)
            if joints_at_home(robot):
                time.sleep(0.1)
                continue

            obs = robot.get_observation()
            action = action_capture.latest

            if not obs or action is None:
                time.sleep(0.05)
                continue

            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(
                dataset.features, action, prefix=ACTION
            )
            frame = {**obs_frame, **action_frame, "task": TASK_DESCRIPTION}
            dataset.add_frame(frame)
            episode_frames += 1

            dt = time.perf_counter() - loop_start
            time.sleep(max(0, 1.0 / RECORDING_FPS - dt))

        # Save or discard
        if _shutdown_requested:
            if episode_frames > 0:
                log.info("Episode %d: discarding %d frames (shutdown).", idx, episode_frames)
                dataset.clear_episode_buffer()
            return False

        if episode_frames > 0:
            dataset.save_episode()
            log.info(
                "Episode %d saved: %d frames (%.1fs).",
                idx, episode_frames, episode_frames / RECORDING_FPS,
            )
            return True
        else:
            log.warning("Episode %d: no frames recorded.", idx)
            dataset.clear_episode_buffer()
            return False

    except KeyboardInterrupt:
        if episode_frames > 0:
            dataset.clear_episode_buffer()
        return False

    finally:
        if robot_connected:
            try:
                robot.disconnect()
            except Exception:
                log.exception("Error disconnecting robot.")
        kill_process_group(engine_proc)
        kill_process_group(model_proc)
        kill_sim(sim_proc)
        time.sleep(3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    hf_user = os.environ.get("HF_USER")
    parser = argparse.ArgumentParser(
        description="Per-episode recording using aic_engine trial orchestration."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to generated YAML config (from generate_scenarios.py).",
    )
    parser.add_argument(
        "--policy", type=str, required=True,
        help="Policy module for aic_model (e.g. my_policy.ExpertCollector).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=f"{hf_user}/aic_expert_demos" if hf_user else None,
        help="LeRobot dataset repo id (default: $HF_USER/aic_expert_demos).",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("recording_work"),
        help="Directory for per-trial configs, logs, and progress (default: recording_work).",
    )
    parser.add_argument(
        "--episode-timeout", type=float, default=300,
        help="Timeout per episode in seconds (default: 300).",
    )
    parser.add_argument(
        "--start-idx", type=int, default=None,
        help="First trial index to record (0-based, inclusive).",
    )
    parser.add_argument(
        "--end-idx", type=int, default=None,
        help="Last trial index to record (0-based, exclusive).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip completed episodes, retry failed ones.",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push dataset to HuggingFace Hub when done.",
    )
    parser.add_argument(
        "--worker-id", type=int, default=None,
        help="Enable worker mode with this 0-based worker id. "
             "Starts a private Zenoh router and bypasses /entrypoint.sh.",
    )
    parser.add_argument(
        "--zenoh-port", type=int, default=7447,
        help="Zenoh router port (default: 7447). Only used in worker mode.",
    )
    parser.add_argument(
        "--gz-partition", type=str, default=None,
        help="GZ_PARTITION to isolate Gazebo transport. Only used in worker mode.",
    )
    parser.add_argument(
        "--force-overwrite", action="store_true",
        help="Overwrite existing dataset without prompting.",
    )
    args = parser.parse_args()
    if not args.dataset_repo_id:
        parser.error("Set HF_USER env var or pass --dataset-repo-id")
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global _shutdown_requested, _worker_mode, _gz_partition

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    args = parse_args()

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Worker mode: bring up isolated Zenoh router before anything touches rclpy.
    router_proc = None
    if args.worker_id is not None:
        _worker_mode = True
        _gz_partition = args.gz_partition
        router_proc = setup_worker_env(
            args.worker_id, args.zenoh_port, args.gz_partition, log_dir,
        )

    # Split config into per-trial YAMLs
    trial_configs = split_config(args.config, work_dir)
    total = len(trial_configs)

    # Apply index range
    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else total
    start = max(0, min(start, total))
    end = max(start, min(end, total))

    # Progress tracker
    progress = ProgressTracker(
        work_dir / "progress.json",
        str(args.config),
        args.dataset_repo_id,
    )

    # Check for existing dataset when not resuming
    if not args.resume:
        dataset_dir = (
            Path.home() / ".cache" / "huggingface" / "lerobot" / args.dataset_repo_id
        )
        if dataset_dir.exists():
            if args.force_overwrite or _worker_mode:
                shutil.rmtree(dataset_dir)
                log.info("Deleted %s (force-overwrite).", dataset_dir)
            else:
                answer = input(
                    f"Dataset already exists at {dataset_dir}\n"
                    f"Delete it and start fresh? [y/N] "
                )
                if answer.strip().lower() in ("y", "yes"):
                    shutil.rmtree(dataset_dir)
                    log.info("Deleted %s", dataset_dir)
                else:
                    log.error("Aborting. Use --resume to append, or delete manually.")
                    sys.exit(1)

    # Create robot (static features only — per-episode connect inside run_episode)
    config = AICRobotAICControllerConfig(
        id="aic",
        teleop_target_mode="cartesian",
        teleop_frame_id="base_link",
    )
    robot = AICRobotAICController(config)
    action_capture = ActionCapture()

    dataset = create_dataset(robot, args.dataset_repo_id, args.resume)

    # Main loop
    completed_count = 0
    failed_count = 0

    try:
        with VideoEncodingManager(dataset):
            for idx in range(start, end):
                if _shutdown_requested:
                    break

                if args.resume and progress.is_completed(idx):
                    log.info("Episode %d: already completed, skipping.", idx)
                    completed_count += 1
                    continue

                log.info(
                    "=== Episode %d/%d (trial config: %s) ===",
                    idx, end - 1, trial_configs[idx].name,
                )
                success = run_episode(
                    idx=idx,
                    trial_config=trial_configs[idx],
                    policy=args.policy,
                    robot=robot,
                    dataset=dataset,
                    action_capture=action_capture,
                    log_dir=log_dir,
                    timeout=args.episode_timeout,
                )

                if _shutdown_requested:
                    break

                if success:
                    progress.mark_completed(idx)
                    completed_count += 1
                else:
                    progress.mark_failed(idx)
                    failed_count += 1

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        try:
            dataset.finalize()
        except Exception:
            log.exception("Error finalizing dataset.")

        if getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                log.exception("Error disconnecting robot.")

        if router_proc is not None:
            log.info("Worker %d: stopping Zenoh router.", args.worker_id)
            kill_process_group(router_proc, timeout=10)

    if args.push_to_hub and completed_count > 0:
        log.info("Pushing dataset to Hub...")
        dataset.push_to_hub(private=True)

    log.info(
        "Done. %d completed, %d failed (of %d). Dataset: %s",
        completed_count, failed_count, end - start, args.dataset_repo_id,
    )


if __name__ == "__main__":
    main()
