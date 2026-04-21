#!/usr/bin/env python3
"""Record aic_model actions into a LeRobot dataset across trials.

Launches aic_engine with the given config YAML and a user-specified aic_model
policy, then records robot observations and actions into a LeRobot dataset.
Expects a simulation already running (started by the user).

The script does NOT manage the sim lifecycle — it only:
  1. Starts aic_model (lifecycle node) with the chosen policy.
  2. Starts aic_engine with the generated config (handles spawning,
     lifecycle transitions, goal dispatch, and cleanup automatically).
  3. Subscribes to observations and commands, recording each frame into
     a LeRobot dataset at a fixed rate.
  4. Stops recording when aic_engine exits.

Usage::

    # Terminal 1 — start sim (empty world):
    /entrypoint.sh spawn_task_board:=false spawn_cable:=false \
        ground_truth:=true start_aic_engine:=false

    # Terminal 2 — record:
    HF_USER=youruser pixi run python ground_truth_collection/record_trials.py \
        --config ground_truth_collection/my_config.yaml \
        --policy my_policy.ExpertCollector

Environment variables:
    HF_USER  – HuggingFace username (required unless --dataset-repo-id given)
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

import cv2  # noqa: F401  — import before lerobot to avoid libtiff symbol conflict
import numpy as np

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

# UR5e home joint positions (from aic_engine sample_config.yaml).
# Used to detect episode boundaries when aic_engine resets the robot.
HOME_JOINT_POSITIONS = np.array([-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110])
HOME_JOINT_TOLERANCE = 0.01  # radians — joints teleport, so tolerance can be tight


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    hf_user = os.environ.get("HF_USER")
    parser = argparse.ArgumentParser(
        description="Record aic_model actions into a LeRobot dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to generated YAML config (from generate_scenarios.py).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy module for aic_model (e.g. my_policy.ExpertCollector).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=f"{hf_user}/aic_expert_demos" if hf_user else None,
        help="LeRobot dataset repo id (default: $HF_USER/aic_expert_demos).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume recording into an existing dataset.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub when done.",
    )
    args = parser.parse_args()
    if not args.dataset_repo_id:
        parser.error("Set HF_USER env var or pass --dataset-repo-id")
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")
    return args


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


WS_SETUP = "/ws_aic/install/setup.bash"


_child_procs: list[subprocess.Popen] = []


def launch_ros_process(cmd: list[str], log_path: Path | None = None) -> subprocess.Popen:
    """Launch a ROS 2 subprocess with the workspace overlay sourced.

    Wraps the command in ``bash -c "source <workspace>/setup.bash && ..."``
    so that packages built in the workspace (e.g. aic_engine) are findable,
    even when this script is invoked from a pixi environment that only has
    base ROS 2.
    """
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
    proc = subprocess.Popen(shell_cmd, **kwargs)
    _child_procs.append(proc)
    return proc


def kill_process_group(proc: subprocess.Popen | None):
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
# Episode boundary detection
# ---------------------------------------------------------------------------


def joints_at_home(robot: AICRobotAICController) -> bool:
    """Return True if the robot's joints are at the home position.

    Between trials, aic_engine calls home_robot() which teleports the
    joints to HOME_JOINT_POSITIONS.  Detecting this snap lets us split
    the recording into per-trial episodes.
    """
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
# Cleanup
# ---------------------------------------------------------------------------


def _cleanup_children():
    """Kill all child processes we spawned."""
    for proc in reversed(_child_procs):
        kill_process_group(proc)


_shutdown_requested = False


def _on_signal(signum, _frame):
    """Signal handler that requests a graceful shutdown.

    Sets a flag so the recording loop exits cleanly, allowing the finally
    block to save data and tear down child processes.  For SIGINT we also
    raise KeyboardInterrupt so Python's default behavior is preserved
    (e.g. interactive traceback).
    """
    global _shutdown_requested
    log.info("Received signal %d, requesting shutdown...", signum)
    _shutdown_requested = True
    if signum == signal.SIGINT:
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    args = parse_args()
    log_dir = Path("recording_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Check for existing dataset ---
    if not args.resume:
        dataset_dir = (
            Path.home() / ".cache" / "huggingface" / "lerobot" / args.dataset_repo_id
        )
        if dataset_dir.exists():
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

    # --- Wait for sim ---
    wait_for_sim()

    # --- Connect robot (for observations + dataset features) ---
    log.info("Connecting to AIC robot...")
    config = AICRobotAICControllerConfig(
        id="aic",
        teleop_target_mode="cartesian",
        teleop_frame_id="base_link",
    )
    robot = AICRobotAICController(config)
    action_capture = ActionCapture()

    dataset = create_dataset(robot, args.dataset_repo_id, args.resume)
    robot.connect(calibrate=False)
    action_capture.attach(robot)

    # --- Launch aic_model ---
    log.info("Launching aic_model with policy=%s", args.policy)
    model_proc = launch_ros_process(
        [
            "ros2", "run", "aic_model", "aic_model",
            "--ros-args",
            "-p", "use_sim_time:=true",
            "-p", f"policy:={args.policy}",
        ],
        log_dir / "aic_model.log",
    )
    # Give model time to start up
    time.sleep(5)
    if model_proc.poll() is not None:
        log.error("aic_model crashed on startup. See %s", log_dir / "aic_model.log")
        sys.exit(1)

    # --- Launch aic_engine (it manages the full trial lifecycle) ---
    config_path = str(args.config.resolve())
    log.info("Launching aic_engine with config=%s", config_path)
    engine_proc = launch_ros_process(
        [
            "ros2", "run", "aic_engine", "aic_engine",
            "--ros-args",
            "-p", f"config_file_path:={config_path}",
            "-p", "use_sim_time:=true",
            "-p", "ground_truth:=true",
        ],
        log_dir / "aic_engine.log",
    )

    # --- Recording loop ---
    log.info("Recording at %d Hz until aic_engine exits...", RECORDING_FPS)
    frame_count = 0
    episode_frames = 0
    episode_count = 0
    was_at_home = True  # robot starts at home before first trial

    try:
        with VideoEncodingManager(dataset):
            while not _shutdown_requested:
                loop_start = time.perf_counter()

                # Stop when engine exits
                if engine_proc.poll() is not None:
                    log.info(
                        "aic_engine exited (rc=%d). Finishing recording.",
                        engine_proc.returncode,
                    )
                    break

                # Stop if model crashes
                if model_proc.poll() is not None:
                    log.warning("aic_model exited unexpectedly (rc=%d).", model_proc.returncode)
                    break

                # --- Episode boundary detection ---
                # Between trials, aic_engine resets the robot to home.
                # When we see joints snap to home, save the current episode.
                # Don't record frames while the robot is at home (inter-trial
                # gap where entities are being spawned/removed).
                at_home = joints_at_home(robot)

                if at_home and not was_at_home and episode_frames > 0:
                    # Just arrived at home — end of a trial
                    dataset.save_episode()
                    episode_count += 1
                    log.info(
                        "Episode %d saved (%d frames, %.1fs).",
                        episode_count, episode_frames, episode_frames / RECORDING_FPS,
                    )
                    episode_frames = 0
                    action_capture.reset()

                was_at_home = at_home

                if at_home:
                    # Skip recording during inter-trial reset
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
                frame_count += 1
                episode_frames += 1

                dt = time.perf_counter() - loop_start
                time.sleep(max(0, 1.0 / RECORDING_FPS - dt))

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        # Always kill child processes first — this must not be blocked by
        # dataset operations that may fail.
        _cleanup_children()

        try:
            if _shutdown_requested:
                # Interrupted — discard partial episode (image writer may
                # not have flushed all frames to disk yet).
                if episode_frames > 0:
                    log.info("Discarding partial episode (%d frames).", episode_frames)
                    dataset.clear_episode_buffer()
            else:
                # Normal exit — save any remaining episode
                if episode_frames > 0:
                    dataset.save_episode()
                    episode_count += 1
                    log.info("Saved final episode (%d frames).", episode_frames)
        except Exception:
            log.exception("Error saving/clearing episode buffer.")

        try:
            dataset.finalize()
        except Exception:
            log.exception("Error finalizing dataset.")

        if robot.is_connected:
            robot.disconnect()

    if args.push_to_hub:
        log.info("Pushing dataset to Hub...")
        dataset.push_to_hub(private=True)

    log.info(
        "Recording complete. %d episodes, %d total frames. Dataset: %s",
        episode_count, frame_count, args.dataset_repo_id,
    )


if __name__ == "__main__":
    main()
