#!/usr/bin/env python3
"""Batch-collect expert demonstrations across generated scenarios.

Runs against a **single persistent simulation** (started by the user) and
spawns/removes task board and cable entities between episodes.  The LeRobot
Python API records observations and expert actions directly.

Run everything from inside the aic_eval distrobox container so that both
pixi (LeRobot, aic_model) and sim tools (ros2 launch, gz service) are
available.

Usage::

    cd ~/ws_aic/src/aic
    distrobox enter -r aic_eval

    # Terminal 1 — start sim (bare world, no task board / cable / engine):
    #   ALL FOUR flags below are required.
    /entrypoint.sh spawn_task_board:=false spawn_cable:=false \\
        ground_truth:=true start_aic_engine:=false

    # Terminal 2 — collect demos:
    HF_USER=youruser pixi run python docker/my_policy/scripts/batch_collect.py

    # Collect a subset:
    HF_USER=youruser pixi run python docker/my_policy/scripts/batch_collect.py \\
        --start-idx 0 --end-idx 9

Environment variables:
    HF_USER          – HuggingFace username (required)
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

# UR5e home joint positions (from aic_engine sample_config.yaml)
HOME_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME_JOINT_POSITIONS = [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]

# Prefixes used to split .args into task-board vs cable params
TASK_BOARD_PARAM_PREFIXES = (
    "task_board_",
    "lc_mount_rail_",
    "sfp_mount_rail_",
    "sc_mount_rail_",
    "sc_port_",
    "nic_card_mount_",
)
CABLE_PARAM_PREFIXES = ("cable_", "attach_cable_")

# Cable spawn position/orientation matching aic_gz_bringup.launch.py defaults.
# These place the cable right at the gripper when the robot is at home position.
# spawn_cable.launch.py has DIFFERENT defaults (-0.35, 0.4, 1.15) which are wrong.
CABLE_SPAWN_DEFAULTS = {
    "cable_x": "0.172",
    "cable_y": "0.024",
    "cable_z": "1.518",
    "cable_roll": "0.4432",
    "cable_pitch": "-0.48",
    "cable_yaw": "1.3303",
}

# Sim readiness: poll for this ROS service
SIM_READINESS_SERVICE = "/aic_controller/change_target_mode"
SIM_READINESS_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# CLI
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def launch_subprocess(
    cmd: list[str] | str, log_file: Path, **kwargs
) -> subprocess.Popen:
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


def parse_scenario_args(raw: str) -> dict[str, str]:
    """Parse 'key:=value key2:=value2 ...' into a dict."""
    params: dict[str, str] = {}
    for token in raw.split():
        if ":=" in token:
            k, v = token.split(":=", 1)
            params[k] = v
    return params


def split_scenario_params(
    params: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Split scenario params into task-board and cable launch arg lists."""
    tb_args: list[str] = []
    cable_args: list[str] = []
    for k, v in params.items():
        arg = f"{k}:={v}"
        if any(k.startswith(p) for p in TASK_BOARD_PARAM_PREFIXES):
            tb_args.append(arg)
        elif any(k.startswith(p) for p in CABLE_PARAM_PREFIXES):
            cable_args.append(arg)
    return tb_args, cable_args


# ---------------------------------------------------------------------------
# Sim readiness check
# ---------------------------------------------------------------------------


def wait_for_sim(timeout: int = SIM_READINESS_TIMEOUT):
    """Poll until the sim's aic_controller service is available."""
    log.info(
        "Waiting for sim (polling %s)...", SIM_READINESS_SERVICE
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "pixi", "run", "ros2", "service", "list",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if SIM_READINESS_SERVICE in result.stdout:
            log.info("Sim is ready.")
            return
        time.sleep(3)

    log.error(
        "Timed out waiting for sim after %ds.\n"
        "Start the sim in another terminal (all four flags required):\n\n"
        "  /entrypoint.sh spawn_task_board:=false spawn_cable:=false \\\n"
        "      ground_truth:=true start_aic_engine:=false\n\n"
        "The sim must start with an EMPTY world (no task board or cable)\n"
        "so they can be spawned/removed per episode.",
        timeout,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entity management (runs directly — assumes we're inside the container)
# ---------------------------------------------------------------------------


def _run_ws_cmd(cmd_str: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command using the container's sourced workspace (not pixi's).

    Commands like ``ros2 launch aic_bringup ...`` and ``gz service ...``
    need packages that are only available after sourcing the eval
    container's built workspace.
    """
    return subprocess.run(
        ["bash", "-c", f"source /ws_aic/install/setup.bash && {cmd_str}"],
        capture_output=True, text=True, timeout=timeout,
    )


def spawn_task_board(
    tb_args: list[str], log_dir: Path, idx_fmt: str
) -> bool:
    """Spawn task board via ros2 launch (container workspace)."""
    args_str = " ".join(tb_args)
    log.info("  Spawning task board...")
    result = _run_ws_cmd(
        f"ros2 launch aic_bringup spawn_task_board.launch.py {args_str}",
        timeout=30,
    )
    (log_dir / f"spawn_tb_{idx_fmt}.log").write_text(
        result.stdout + result.stderr
    )
    if result.returncode != 0:
        log.error("  FAIL: spawn task board (rc=%d)", result.returncode)
        return False
    return True


def spawn_cable(
    cable_args: list[str], log_dir: Path, idx_fmt: str
) -> bool:
    """Spawn cable via ros2 launch (container workspace)."""
    # Merge position defaults — scenario args override if present.
    supplied_keys = {a.split(":=")[0] for a in cable_args if ":=" in a}
    full_args = list(cable_args)
    for k, v in CABLE_SPAWN_DEFAULTS.items():
        if k not in supplied_keys:
            full_args.append(f"{k}:={v}")
    args_str = " ".join(full_args)
    log.info("  Spawning cable...")
    result = _run_ws_cmd(
        f"ros2 launch aic_bringup spawn_cable.launch.py {args_str}",
        timeout=30,
    )
    (log_dir / f"spawn_cable_{idx_fmt}.log").write_text(
        result.stdout + result.stderr
    )
    if result.returncode != 0:
        log.error("  FAIL: spawn cable (rc=%d)", result.returncode)
        return False
    return True


def remove_entity(name: str) -> bool:
    """Remove a Gazebo entity via gz service (container workspace)."""
    result = _run_ws_cmd(
        f'gz service -s /world/aic_world/remove'
        f' --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean'
        f' --timeout 5000 --req \'name: "{name}", type: MODEL\'',
        timeout=15,
    )
    if result.returncode != 0:
        log.warning("  Could not remove '%s': %s", name, result.stderr.strip())
        return False
    return True


def _switch_controller(activate: list[str], deactivate: list[str]) -> bool:
    """Activate/deactivate ros2_control controllers via controller_manager."""
    activate_str = str(activate).replace("'", '"')
    deactivate_str = str(deactivate).replace("'", '"')
    yaml = (
        f"{{activate_controllers: {activate_str},"
        f" deactivate_controllers: {deactivate_str},"
        f" strictness: 1}}"  # BEST_EFFORT
    )
    result = _run_ws_cmd(
        f"ros2 service call /controller_manager/switch_controller"
        f" controller_manager_msgs/srv/SwitchController"
        f' "{yaml}"',
        timeout=15,
    )
    if result.returncode != 0:
        log.warning("  SwitchController failed: %s", result.stderr.strip())
        return False
    return True


def reset_robot_joints() -> bool:
    """Reset robot joints to home position.

    Mirrors aic_engine's home_robot() sequence:
      1. Deactivate aic_controller (so it stops commanding joints)
      2. Call /scoring/reset_joints (Gazebo plugin teleports joints)
      3. Reactivate aic_controller
    """
    # 1. Deactivate aic_controller
    log.info("    Deactivating aic_controller...")
    if not _switch_controller([], ["aic_controller"]):
        return False

    # 2. Reset joints
    names_str = str(HOME_JOINT_NAMES).replace("'", '"')
    positions_str = str(HOME_JOINT_POSITIONS)
    yaml = f"{{joint_names: {names_str}, initial_positions: {positions_str}}}"
    result = _run_ws_cmd(
        f"ros2 service call /scoring/reset_joints"
        f" aic_engine_interfaces/srv/ResetJoints"
        f' "{yaml}"',
        timeout=15,
    )
    if result.returncode != 0:
        log.warning("  Joint reset failed: %s", result.stderr.strip())
        # Try to reactivate controller even if reset failed
        _switch_controller(["aic_controller"], [])
        return False

    # 3. Reactivate aic_controller
    log.info("    Reactivating aic_controller...")
    if not _switch_controller(["aic_controller"], []):
        return False

    return True


# ---------------------------------------------------------------------------
# Action capture via ROS subscriber
# ---------------------------------------------------------------------------


class ActionCapture:
    """Subscribes to /aic_controller/pose_commands to capture expert actions."""

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
    episode_timeout: int,
) -> bool:
    """Collect one episode. Returns True on success."""
    idx_fmt = f"{idx:03d}"
    args_file = scenario_dir / f"scenario_{idx_fmt}.args"

    if not args_file.exists():
        log.warning("SKIP: %s not found", args_file)
        return False

    scenario_params = parse_scenario_args(args_file.read_text().strip())
    tb_args, cable_args = split_scenario_params(scenario_params)

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

    model_proc = None
    goal_proc = None
    entities_spawned: list[str] = []

    try:
        # --- 1. Reset robot joints to home position ---
        log.info("  Resetting robot joints...")
        reset_robot_joints()
        time.sleep(2)

        # --- 2. Spawn task board and cable ---
        if not spawn_task_board(tb_args, log_dir, idx_fmt):
            return False
        entities_spawned.append("task_board")

        if not spawn_cable(cable_args, log_dir, idx_fmt):
            return False
        entities_spawned.append("cable_0")

        # Wait for CablePlugin to attach cable to gripper and physics to settle
        time.sleep(8)

        # Verify ground-truth TF frames are being published.
        # tf2_echo runs forever, so we launch it briefly and check output.
        port_frame = f"task_board/{target_module}/{port_name}_link"
        plug_frame = f"cable_0/{plug_name}_link"
        log.info("  Checking ground-truth TF frames...")
        tf_ok = True
        for frame in [port_frame, plug_frame]:
            try:
                result = subprocess.run(
                    [
                        "pixi", "run", "ros2", "run", "tf2_ros",
                        "tf2_echo", "base_link", frame,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                output = result.stdout
            except subprocess.TimeoutExpired as e:
                # Expected — tf2_echo never exits on its own.
                output = (e.stdout or b"").decode(errors="replace")
            if "At time" not in output:
                log.error(
                    "  TF frame '%s' not available. "
                    "Is the sim running with ground_truth:=true?",
                    frame,
                )
                tf_ok = False
        if not tf_ok:
            return False

        # --- 3. Tare F/T sensor ---
        subprocess.run(
            [
                "pixi", "run", "ros2", "service", "call",
                "/aic_controller/tare_force_torque_sensor",
                "std_srvs/srv/Trigger",
            ],
            capture_output=True,
            timeout=10,
        )
        time.sleep(1)

        # --- 4. Start ExpertCollector ---
        action_capture.reset()
        log.info("  Starting ExpertCollector...")
        model_proc = launch_subprocess(
            [
                "pixi", "run", "ros2", "run", "aic_model", "aic_model",
                "--ros-args",
                "-p", "use_sim_time:=true",
                "-p", "policy:=my_policy.ExpertCollector",
            ],
            log_dir / f"expert_{idx_fmt}.log",
        )
        time.sleep(5)

        if model_proc.poll() is not None:
            log.error(
                "  FAIL: ExpertCollector crashed. See %s/expert_%s.log",
                log_dir, idx_fmt,
            )
            return False

        # --- 5. Activate aic_model lifecycle node ---
        # aic_model is a LifecycleNode; without aic_engine it stays
        # unconfigured.  We must transition it through configure → activate
        # before it will accept action goals.
        log.info("  Activating aic_model lifecycle node...")
        for transition in ["configure", "activate"]:
            result = subprocess.run(
                [
                    "pixi", "run", "ros2", "lifecycle", "set",
                    "/aic_model", transition,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                log.error(
                    "  FAIL: lifecycle '%s' failed: %s",
                    transition,
                    result.stderr.strip(),
                )
                return False
        time.sleep(1)

        # --- 6. Send insert_cable action goal (non-blocking) ---
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
                "pixi", "run", "ros2", "action", "send_goal",
                "--feedback",
                "/insert_cable",
                "aic_task_interfaces/action/InsertCable",
                goal_yaml,
            ],
            log_dir / f"goal_{idx_fmt}.log",
        )

        # --- 7. Recording loop ---
        log.info("  Recording at %d Hz...", RECORDING_FPS)
        episode_start = time.monotonic()
        frame_count = 0

        while True:
            loop_start = time.perf_counter()

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

            obs = robot.get_observation()
            if not obs:
                time.sleep(0.05)
                continue

            action = action_capture.latest
            if action is None:
                time.sleep(0.05)
                continue

            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(
                dataset.features, action, prefix=ACTION
            )
            frame = {**obs_frame, **action_frame, "task": TASK_DESCRIPTION}
            dataset.add_frame(frame)
            frame_count += 1

            dt = time.perf_counter() - loop_start
            time.sleep(max(0, 1.0 / RECORDING_FPS - dt))

        # --- 8. Save episode ---
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
        # --- 9. Tear down episode (keep sim running) ---
        kill_process_group(goal_proc)
        kill_process_group(model_proc)

        # Remove spawned entities in reverse order
        for entity in reversed(entities_spawned):
            log.info("  Removing %s...", entity)
            remove_entity(entity)

        time.sleep(2)  # let physics settle after removal


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

    log.info("=" * 50)
    log.info(" Batch Expert Demo Collection")
    log.info("=" * 50)
    log.info(" Scenarios:   %d..%d of %d", args.start_idx, end_idx, total)
    log.info(" Dataset:     %s", args.dataset_repo_id)
    log.info(" FPS:         %d", RECORDING_FPS)
    log.info(" Episode max: %ds", args.episode_timeout)
    log.info("=" * 50)

    # --- Create robot (not connected yet — just need features for dataset) ---
    log.info("Creating AIC robot config...")
    config = AICRobotAICControllerConfig(
        id="aic",
        teleop_target_mode="cartesian",
        teleop_frame_id="base_link",
    )
    robot = AICRobotAICController(config)
    action_capture = ActionCapture()

    # --- Create dataset ---
    log.info("Creating dataset: %s", args.dataset_repo_id)
    dataset = create_dataset(robot, args.dataset_repo_id, args.resume)

    # --- Wait for user-started sim to be ready ---
    wait_for_sim()

    # Sanity check: warn if a task board already exists (means the sim was
    # started with spawn_task_board:=true, which will cause entity-name
    # collisions when we try to spawn our own).
    check = _run_ws_cmd(
        'gz service -s /world/aic_world/remove'
        ' --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean'
        ' --timeout 2000 --req \'name: "task_board", type: MODEL\'',
        timeout=10,
    )
    # If the removal succeeded, a task_board was present — warn the user.
    if check.returncode == 0 and "true" in check.stdout.lower():
        log.warning(
            "Removed a pre-existing task_board from the sim. "
            "Did you forget spawn_task_board:=false when starting the sim?"
        )
    # Also try removing a stale cable_0
    _run_ws_cmd(
        'gz service -s /world/aic_world/remove'
        ' --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean'
        ' --timeout 2000 --req \'name: "cable_0", type: MODEL\'',
        timeout=10,
    )

    # --- Connect robot ONCE ---
    log.info("Connecting to AIC robot...")
    robot.connect(calibrate=False)
    action_capture.attach(robot)

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
                    episode_timeout=args.episode_timeout,
                )
                if ok:
                    succeeded += 1
                else:
                    failed += 1
    finally:
        dataset.finalize()

        if robot.is_connected:
            robot.disconnect()

    if args.push_to_hub:
        log.info("Pushing dataset to Hub...")
        dataset.push_to_hub(private=True)

    log.info("=" * 50)
    log.info(" Collection complete")
    log.info("  Succeeded: %d", succeeded)
    log.info("  Failed:    %d", failed)
    log.info("  Dataset:   %s", args.dataset_repo_id)
    log.info("  Logs:      %s/", log_dir)
    log.info("=" * 50)

    hf_user = os.environ.get("HF_USER", "youruser")
    log.info("")
    log.info("Next steps:")
    log.info("  # Fine-tune ACT on the collected data")
    log.info(
        "  HF_USER=%s bash docker/my_policy/scripts/finetune_act.sh", hf_user
    )


if __name__ == "__main__":
    main()
