#!/usr/bin/env python3
"""Batch-collect expert demonstrations across generated scenarios.

Orchestrates three processes per episode:
  1. Simulation (Gazebo + aic_controller, via eval container with ground_truth:=true)
  2. aic_model running ExpertCollector (drives the robot using GT TF)
  3. LeRobot dataset recording (captures observations + actions via Python API)

The script uses the LeRobot Python API directly (LeRobotDataset.create,
add_frame, save_episode) instead of shelling out to lerobot-record.  Expert
actions are captured by subscribing to the /aic_controller/pose_commands
topic, giving us the exact velocity twists the ExpertCollector sends.

Usage::

    cd ~/ws_aic/src/aic

    # 1. Generate scenarios (only needed once)
    python3 docker/my_policy/scripts/generate_scenarios.py

    # 2. Collect demonstrations
    HF_USER=youruser pixi run python docker/my_policy/scripts/batch_collect.py

    # 3. Collect a subset
    HF_USER=youruser pixi run python docker/my_policy/scripts/batch_collect.py \
        --start-idx 0 --end-idx 9

Environment variables:
    HF_USER          – HuggingFace username (required)
    SIM_LAUNCH_CMD   – command to launch the sim
                       (default: distrobox enter -r aic_eval -- /entrypoint.sh)
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

import cv2  # noqa: F401  — must be imported before lerobot to avoid libtiff symbol conflict
import numpy as np

from aic_control_interfaces.msg import MotionUpdate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR

# Import AIC robot to register the "aic_controller" subclass with LeRobot
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECORDING_FPS = 4  # must match ExpertCollector's CONTROL_HZ and ACT inference rate
TASK_DESCRIPTION = "insert cable into port"
DEFAULT_SIM_CMD = "distrobox enter -r aic_eval -- /entrypoint.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    hf_user = os.environ.get("HF_USER")
    parser = argparse.ArgumentParser(
        description="Batch-collect expert demonstrations across generated scenarios."
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("scenarios"),
        help="Directory containing scenario .args files and manifest.json",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=f"{hf_user}/aic_expert_demos" if hf_user else None,
        help="LeRobot dataset repo id (default: $HF_USER/aic_expert_demos)",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument(
        "--settle-time",
        type=int,
        default=25,
        help="Seconds to wait for sim startup",
    )
    parser.add_argument(
        "--episode-timeout",
        type=int,
        default=60,
        help="Max seconds per episode",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub when done",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume recording into an existing dataset",
    )
    args = parser.parse_args()

    if not args.dataset_repo_id:
        parser.error("Set HF_USER env var or pass --dataset-repo-id")
    return args


def load_manifest(scenario_dir: Path) -> dict:
    manifest_path = scenario_dir / "manifest.json"
    if not manifest_path.exists():
        log.info("Generating scenarios...")
        subprocess.run(
            [
                sys.executable,
                "docker/my_policy/scripts/generate_scenarios.py",
                "--output-dir",
                str(scenario_dir),
            ],
            check=True,
        )
    return json.loads(manifest_path.read_text())


def launch_subprocess(cmd: list[str] | str, log_file: Path, **kwargs) -> subprocess.Popen:
    """Launch a process with output redirected to a log file."""
    fh = log_file.open("w")
    if isinstance(cmd, str):
        return subprocess.Popen(
            cmd,
            shell=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            **kwargs,
        )
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        **kwargs,
    )


def kill_process_group(proc: subprocess.Popen | None):
    """Kill a process and its children via process group."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


# ---------------------------------------------------------------------------
# Action capture via ROS subscriber
# ---------------------------------------------------------------------------


class ActionCapture:
    """Subscribes to /aic_controller/pose_commands to capture expert actions.

    Must be attached to a connected robot (needs the ROS node for the
    subscription).  Call ``attach()`` after ``robot.connect()`` and
    ``detach()`` before ``robot.disconnect()``.
    """

    def __init__(self):
        self._lock = Lock()
        self._latest: dict[str, float] | None = None
        self._sub = None

    def attach(self, robot: AICRobotAICController):
        """Create the ROS subscription on the robot's node."""
        self._sub = robot.ros2_interface.node.create_subscription(
            MotionUpdate,
            "/aic_controller/pose_commands",
            self._callback,
            10,
        )

    def detach(self):
        """Drop the subscription reference (node destruction handles cleanup)."""
        self._sub = None
        self.reset()

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
# Dataset creation
# ---------------------------------------------------------------------------


def create_dataset(
    robot: AICRobotAICController,
    repo_id: str,
    resume: bool,
) -> LeRobotDataset:
    """Create (or resume) a LeRobotDataset with the correct features."""
    _, _, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=True,
        ),
        # Action features: 6D Cartesian velocity (same as MotionUpdateActionDict)
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,  # identity for actions
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=True,
        ),
    )

    if resume:
        ds = LeRobotDataset(repo_id)
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            ds.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
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
# Episode collection
# ---------------------------------------------------------------------------


def collect_episode(
    *,
    idx: int,
    scenario_dir: Path,
    manifest: dict,
    log_dir: Path,
    robot: AICRobotAICController,
    action_capture: ActionCapture,
    dataset: LeRobotDataset,
    settle_time: int,
    episode_timeout: int,
    sim_launch_cmd: str,
) -> bool:
    """Collect one episode. Returns True on success."""
    idx_fmt = f"{idx:03d}"
    args_file = scenario_dir / f"scenario_{idx_fmt}.args"

    if not args_file.exists():
        log.warning("SKIP: %s not found", args_file)
        return False

    scenario_args = args_file.read_text().strip()

    # Extract task info from manifest
    task_info = manifest["scenarios"][idx]["task"]
    plug_name = task_info["plug_name"]
    port_name = task_info["port_name"]
    target_module = task_info["target_module_name"]
    cable_name = task_info["cable_name"]
    cable_type = task_info["cable_type"]
    plug_type = task_info["plug_type"]
    port_type = task_info["port_type"]

    log.info(
        "=== Scenario %s === %s -> %s @ %s",
        idx_fmt,
        plug_name,
        port_name,
        target_module,
    )

    sim_proc = None
    model_proc = None
    goal_proc = None
    connected = False

    try:
        # --- 1. Launch simulation ---
        log.info("  Starting sim...")
        sim_cmd = f"{sim_launch_cmd} {scenario_args} ground_truth:=true start_aic_engine:=false gazebo_gui:=false launch_rviz:=false"
        sim_proc = launch_subprocess(sim_cmd, log_dir / f"sim_{idx_fmt}.log")
        time.sleep(settle_time)

        if sim_proc.poll() is not None:
            log.error("  FAIL: Sim crashed. See %s/sim_%s.log", log_dir, idx_fmt)
            return False

        # --- 2. Connect robot (sim must be running) ---
        log.info("  Connecting robot...")
        robot.connect(calibrate=False)
        connected = True
        action_capture.attach(robot)

        # --- 3. Tare F/T sensor ---
        subprocess.run(
            [
                "pixi",
                "run",
                "ros2",
                "service",
                "call",
                "/aic_controller/tare_force_torque_sensor",
                "std_srvs/srv/Trigger",
            ],
            capture_output=True,
            timeout=10,
        )
        time.sleep(1)

        # --- 4. Reset action capture and start ExpertCollector ---
        action_capture.reset()
        log.info("  Starting ExpertCollector...")
        model_proc = launch_subprocess(
            [
                "pixi",
                "run",
                "ros2",
                "run",
                "aic_model",
                "aic_model",
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                "policy:=my_policy.ExpertCollector",
            ],
            log_dir / f"expert_{idx_fmt}.log",
        )
        time.sleep(5)

        if model_proc.poll() is not None:
            log.error(
                "  FAIL: ExpertCollector crashed. See %s/expert_%s.log",
                log_dir,
                idx_fmt,
            )
            return False

        # --- 5. Send insert_cable action goal (non-blocking) ---
        log.info("  Sending insert_cable goal...")
        goal_yaml = (
            f"{{task: {{id: 'scenario_{idx_fmt}', cable_type: '{cable_type}', "
            f"cable_name: '{cable_name}', plug_type: '{plug_type}', "
            f"plug_name: '{plug_name}', port_type: '{port_type}', "
            f"port_name: '{port_name}', target_module_name: '{target_module}', "
            f"time_limit: {episode_timeout}}}}}"
        )
        goal_proc = launch_subprocess(
            [
                "pixi",
                "run",
                "ros2",
                "action",
                "send_goal",
                "/insert_cable",
                "aic_task_interfaces/action/InsertCable",
                goal_yaml,
            ],
            log_dir / f"goal_{idx_fmt}.log",
        )

        # --- 6. Recording loop ---
        log.info("  Recording at %d Hz...", RECORDING_FPS)
        episode_start = time.monotonic()
        frame_count = 0

        while True:
            loop_start = time.perf_counter()

            # Check termination conditions
            elapsed = time.monotonic() - episode_start
            if elapsed > episode_timeout:
                log.warning("  Episode timed out after %.0fs", elapsed)
                break
            if goal_proc.poll() is not None:
                log.info("  Action goal finished (rc=%d)", goal_proc.returncode)
                break
            if model_proc.poll() is not None:
                log.info("  ExpertCollector exited")
                break

            # Capture observation from the robot
            obs = robot.get_observation()
            if not obs:
                time.sleep(0.05)
                continue

            # Capture latest expert action
            action = action_capture.latest
            if action is None:
                time.sleep(0.05)
                continue

            # Build dataset frame
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(
                dataset.features, action, prefix=ACTION
            )
            frame = {**obs_frame, **action_frame, "task": TASK_DESCRIPTION}
            dataset.add_frame(frame)
            frame_count += 1

            # Sleep to maintain target FPS
            dt = time.perf_counter() - loop_start
            time.sleep(max(0, 1.0 / RECORDING_FPS - dt))

        # --- 7. Save episode ---
        if frame_count > 0:
            dataset.save_episode()
            log.info(
                "  OK: Saved episode with %d frames (%.1fs)",
                frame_count,
                frame_count / RECORDING_FPS,
            )
        else:
            log.warning("  No frames captured, discarding episode")
            dataset.clear_episode_buffer()
            return False

        return True

    finally:
        # --- 8. Tear down ---
        kill_process_group(goal_proc)
        kill_process_group(model_proc)

        # Disconnect robot before killing sim (clean ROS shutdown)
        action_capture.detach()
        if connected and robot.is_connected:
            robot.disconnect()

        kill_process_group(sim_proc)
        time.sleep(3)  # let ports release


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    manifest = load_manifest(args.scenario_dir)
    total = manifest["count"]

    end_idx = args.end_idx if args.end_idx is not None else total - 1
    log_dir = args.scenario_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    sim_launch_cmd = os.environ.get("SIM_LAUNCH_CMD", DEFAULT_SIM_CMD)

    log.info("=" * 44)
    log.info(" Batch Expert Demo Collection")
    log.info("=" * 44)
    log.info(" Scenarios:   %d..%d of %d", args.start_idx, end_idx, total)
    log.info(" Dataset:     %s", args.dataset_repo_id)
    log.info(" FPS:         %d", RECORDING_FPS)
    log.info(" Settle time: %ds", args.settle_time)
    log.info(" Episode max: %ds", args.episode_timeout)
    log.info(" Sim launch:  %s", sim_launch_cmd)
    log.info("=" * 44)

    # --- Create robot (not connected yet — just need features for dataset) ---
    log.info("Creating AIC robot config...")
    config = AICRobotAICControllerConfig(
        id="aic",
        teleop_target_mode="cartesian",
        teleop_frame_id="base_link",
    )
    robot = AICRobotAICController(config)
    action_capture = ActionCapture()

    # --- Create dataset (uses robot.observation_features / action_features,
    #     which don't require a connection) ---
    log.info("Creating dataset: %s", args.dataset_repo_id)
    dataset = create_dataset(robot, args.dataset_repo_id, args.resume)

    succeeded = 0
    failed = 0

    try:
        with VideoEncodingManager(dataset):
            for idx in range(args.start_idx, end_idx + 1):
                ok = collect_episode(
                    idx=idx,
                    scenario_dir=args.scenario_dir,
                    manifest=manifest,
                    log_dir=log_dir,
                    robot=robot,
                    action_capture=action_capture,
                    dataset=dataset,
                    settle_time=args.settle_time,
                    episode_timeout=args.episode_timeout,
                    sim_launch_cmd=sim_launch_cmd,
                )
                if ok:
                    succeeded += 1
                else:
                    failed += 1
    finally:
        dataset.finalize()

    if args.push_to_hub:
        log.info("Pushing dataset to Hub...")
        dataset.push_to_hub(private=True)

    log.info("=" * 44)
    log.info(" Collection complete")
    log.info("  Succeeded: %d", succeeded)
    log.info("  Failed:    %d", failed)
    log.info("  Dataset:   %s", args.dataset_repo_id)
    log.info("  Logs:      %s/", log_dir)
    log.info("=" * 44)

    hf_user = os.environ.get("HF_USER", "youruser")
    log.info("")
    log.info("Next steps:")
    log.info("  # Fine-tune ACT on the collected data")
    log.info(
        "  HF_USER=%s bash docker/my_policy/scripts/finetune_act.sh", hf_user
    )
    log.info("")
    log.info("  # Run the fine-tuned policy")
    log.info(
        "  ACT_MODEL_REPO=%s/aic_act_finetuned \\", hf_user
    )
    log.info("    pixi run ros2 run aic_model aic_model --ros-args \\")
    log.info("    -p policy:=my_policy.HybridACTInsertion")


if __name__ == "__main__":
    main()
