import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import copy
import json
import math
import time

import cv2
import draccus
import numpy as np
import torch
from geometry_msgs.msg import Pose, Twist, Vector3, Wrench
from huggingface_hub import snapshot_download
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from pathlib import Path
from rclpy.node import Node
from safetensors.torch import load_file
from std_msgs.msg import Header

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD = 2.0  # N - triggers phase transition
MAX_INSERTION_FORCE = 12.0  # N - pause z-advance
RETRACT_FORCE = 18.0  # N - emergency retract (penalty at 20 N)
INSERTION_FEEDFORWARD_FORCE = 5.0  # N - gentle push along insertion axis
ACT_PHASE_TIMEOUT = 35.0  # s
INSERTION_PHASE_TIMEOUT = 12.0  # s
SPIRAL_RADIUS_MAX = 0.003  # m (3 mm)
SPIRAL_RATE = 2.0  # Hz
INSERTION_DEPTH = 0.015  # m (15 mm)
INSERTION_STEP = 0.0005  # m per step (0.5 mm)
INSERTION_HZ = 10.0  # control rate for insertion phase
ACT_HZ = 4.0  # control rate for ACT phase
EMA_ALPHA = 0.3  # exponential moving average for F/T filter
STALL_TIMEOUT = 1.0  # s before triggering spiral search
RETRACT_DISTANCE = 0.002  # m (2 mm)


class HybridACTInsertion(Policy):
    """Two-phase cable insertion policy.

    Phase 1: Fine-tuned ACT model drives the robot toward the port (coarse
    localization) while monitoring the Axia80 F/T sensor for contact.

    Phase 2: Classical impedance control with force feedback performs the
    final insertion using low stiffness, feedforward wrench, and spiral
    search fallback.
    """

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------------------------------------------------
        # 1. ACT model loading (from RunACT pattern)
        # -----------------------------------------------------------------
        repo_id = os.environ.get("ACT_MODEL_REPO", "grkw/aic_act_policy")
        policy_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
            )
        )

        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]

        config = draccus.decode(ACTConfig, config_dict)
        self.act_policy = ACTPolicy(config)
        self.act_policy.load_state_dict(
            load_file(policy_path / "model.safetensors")
        )
        self.act_policy.eval()
        self.act_policy.to(self.device)
        self.get_logger().info(
            f"ACT policy loaded on {self.device} from {policy_path}"
        )

        # -----------------------------------------------------------------
        # 2. Normalization stats
        # -----------------------------------------------------------------
        stats_path = (
            policy_path
            / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        stats = load_file(stats_path)

        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            cam: {
                "mean": get_stat(f"observation.images.{cam}_camera.mean", (1, 3, 1, 1)),
                "std": get_stat(f"observation.images.{cam}_camera.std", (1, 3, 1, 1)),
            }
            for cam in ("left", "center", "right")
        }

        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))

        self.image_scaling = 0.25

        # F/T filter state
        self._filtered_force = np.zeros(3)

        self.get_logger().info("HybridACTInsertion initialized.")

    # =====================================================================
    # Observation helpers (from RunACT)
    # =====================================================================

    @staticmethod
    def _img_to_tensor(raw_img, device, scale, mean, std):
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation):
        obs = {}
        for cam, img_attr in [
            ("left", "left_image"),
            ("center", "center_image"),
            ("right", "right_image"),
        ]:
            obs[f"observation.images.{cam}_camera"] = self._img_to_tensor(
                getattr(obs_msg, img_attr),
                self.device,
                self.image_scaling,
                self.img_stats[cam]["mean"],
                self.img_stats[cam]["std"],
            )

        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity
        state_np = np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                *obs_msg.controller_state.tcp_error,
                *obs_msg.joint_states.position[:7],
            ],
            dtype=np.float32,
        )
        raw = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        obs["observation.state"] = (raw - self.state_mean) / self.state_std
        return obs

    # =====================================================================
    # F/T helpers
    # =====================================================================

    def _compute_tared_force(self, obs_msg: Observation):
        """Return (force_vec, magnitude) after subtracting tare offset."""
        raw = obs_msg.wrist_wrench.wrench.force
        tare = obs_msg.controller_state.fts_tare_offset.wrench.force
        f = np.array([raw.x - tare.x, raw.y - tare.y, raw.z - tare.z])
        return f, np.linalg.norm(f)

    def _update_force_filter(self, force_vec):
        """EMA filter on force vector; returns filtered magnitude."""
        self._filtered_force = (
            EMA_ALPHA * force_vec + (1.0 - EMA_ALPHA) * self._filtered_force
        )
        return np.linalg.norm(self._filtered_force)

    # =====================================================================
    # Motion command helpers
    # =====================================================================

    def _set_cartesian_twist_target(self, twist: Twist, frame_id="base_link"):
        """Build a velocity-mode MotionUpdate (same as RunACT)."""
        msg = MotionUpdate()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = twist
        msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def _create_insertion_motion_update(
        self, target_pose: Pose, spiral_offset=(0.0, 0.0)
    ):
        """Build a position-mode MotionUpdate for compliant insertion."""
        pose = copy.deepcopy(target_pose)
        pose.position.x += spiral_offset[0]
        pose.position.y += spiral_offset[1]

        msg = MotionUpdate()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = pose
        # Low z-stiffness for compliance along insertion axis,
        # moderate lateral stiffness to stay centred
        msg.target_stiffness = np.diag(
            [60.0, 60.0, 15.0, 30.0, 30.0, 30.0]
        ).flatten()
        # High damping for smooth motion (low jerk)
        msg.target_damping = np.diag(
            [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]
        ).flatten()
        # Feedforward force along z to push plug in
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=INSERTION_FEEDFORWARD_FORCE),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        # High wrench feedback gains for force-regulated compliance
        msg.wrench_feedback_gains_at_tip = [0.8, 0.8, 0.8, 0.3, 0.3, 0.3]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        return msg

    @staticmethod
    def _get_spiral_offset(elapsed):
        """Return (dx, dy) for a growing spiral search pattern."""
        angle = 2.0 * math.pi * SPIRAL_RATE * elapsed
        radius = min(SPIRAL_RADIUS_MAX, SPIRAL_RADIUS_MAX * elapsed / 3.0)
        return radius * math.cos(angle), radius * math.sin(angle)

    # =====================================================================
    # Phase 1: ACT approach
    # =====================================================================

    def _run_act_approach(self, get_observation, move_robot, send_feedback):
        """Run ACT model until contact detected or timeout.

        Returns the TCP pose at contact, or None on timeout.
        """
        self.get_logger().info("Phase 1: ACT approach starting")
        send_feedback("Phase 1: ACT approach")
        self._filtered_force = np.zeros(3)
        start = time.time()

        while time.time() - start < ACT_PHASE_TIMEOUT:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                continue

            # -- F/T check --
            force_vec, _ = self._compute_tared_force(obs_msg)
            filtered_mag = self._update_force_filter(force_vec)
            if filtered_mag > CONTACT_FORCE_THRESHOLD:
                contact_pose = copy.deepcopy(obs_msg.controller_state.tcp_pose)
                self.get_logger().info(
                    f"Contact detected: {filtered_mag:.2f} N. "
                    f"Pose: ({contact_pose.position.x:.4f}, "
                    f"{contact_pose.position.y:.4f}, "
                    f"{contact_pose.position.z:.4f})"
                )
                return contact_pose

            # -- ACT inference --
            obs_tensors = self.prepare_observations(obs_msg)
            with torch.inference_mode():
                norm_action = self.act_policy.select_action(obs_tensors)

            raw_action = (norm_action * self.action_std) + self.action_mean
            action = raw_action[0].cpu().numpy()

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            motion_update = self._set_cartesian_twist_target(twist)
            move_robot(motion_update=motion_update)
            send_feedback("Phase 1: approaching")

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1.0 / ACT_HZ - elapsed))

        self.get_logger().warn("ACT approach timed out without contact")
        return None

    # =====================================================================
    # Phase 2: Impedance-controlled insertion
    # =====================================================================

    def _run_impedance_insertion(
        self, contact_pose, get_observation, move_robot, send_feedback
    ):
        """Force-regulated insertion with spiral search fallback."""
        self.get_logger().info("Phase 2: impedance insertion starting")
        send_feedback("Phase 2: impedance insertion")

        target_pose = copy.deepcopy(contact_pose)
        z_advanced = 0.0
        start = time.time()
        last_z_progress_time = start
        last_tcp_z = contact_pose.position.z
        use_spiral = False
        spiral_start = None

        while time.time() - start < INSERTION_PHASE_TIMEOUT:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                continue

            # -- F/T monitoring --
            force_vec, force_mag = self._compute_tared_force(obs_msg)

            # Emergency retract
            if force_mag > RETRACT_FORCE:
                self.get_logger().warn(
                    f"Force {force_mag:.1f} N > {RETRACT_FORCE} N — retracting"
                )
                target_pose.position.z -= RETRACT_DISTANCE
                z_advanced = max(0.0, z_advanced - RETRACT_DISTANCE)
                msg = self._create_insertion_motion_update(target_pose)
                move_robot(motion_update=msg)
                self.sleep_for(0.5)
                continue

            # Advance z if force is within limits
            if force_mag < MAX_INSERTION_FORCE and z_advanced < INSERTION_DEPTH:
                target_pose.position.z += INSERTION_STEP
                z_advanced += INSERTION_STEP

            # -- Stall detection for spiral search --
            current_tcp_z = obs_msg.controller_state.tcp_pose.position.z
            if abs(current_tcp_z - last_tcp_z) > 0.0002:
                last_z_progress_time = time.time()
                last_tcp_z = current_tcp_z
                use_spiral = False
                spiral_start = None

            if (
                time.time() - last_z_progress_time > STALL_TIMEOUT
                and z_advanced < INSERTION_DEPTH
            ):
                if not use_spiral:
                    self.get_logger().info("Z-progress stalled — starting spiral search")
                    use_spiral = True
                    spiral_start = time.time()

            # -- Build and send motion command --
            spiral_offset = (0.0, 0.0)
            if use_spiral and spiral_start is not None:
                spiral_offset = self._get_spiral_offset(
                    time.time() - spiral_start
                )

            msg = self._create_insertion_motion_update(target_pose, spiral_offset)
            move_robot(motion_update=msg)

            # -- Check success --
            if z_advanced >= INSERTION_DEPTH:
                self.get_logger().info(
                    f"Insertion complete: {z_advanced * 1000:.1f} mm advanced"
                )
                send_feedback("Insertion complete")
                return True

            send_feedback(
                f"Inserting: {z_advanced * 1000:.1f}/{INSERTION_DEPTH * 1000:.0f} mm, "
                f"F={force_mag:.1f} N"
            )

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1.0 / INSERTION_HZ - elapsed))

        self.get_logger().warn(
            f"Insertion timed out at {z_advanced * 1000:.1f} mm"
        )
        return z_advanced > INSERTION_DEPTH * 0.5

    # =====================================================================
    # Main entry point
    # =====================================================================

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.act_policy.reset()
        self.get_logger().info(f"HybridACTInsertion.insert_cable() — Task: {task}")

        # Phase 1: ACT approach
        contact_pose = self._run_act_approach(
            get_observation, move_robot, send_feedback
        )
        if contact_pose is None:
            send_feedback("ACT approach failed — no contact detected")
            return False

        # Phase 2: Impedance insertion
        success = self._run_impedance_insertion(
            contact_pose, get_observation, move_robot, send_feedback
        )

        self.get_logger().info(
            f"HybridACTInsertion.insert_cable() done — success={success}"
        )
        return success
