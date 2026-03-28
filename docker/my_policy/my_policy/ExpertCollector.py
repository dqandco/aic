"""Expert demonstration collector using ground-truth TF data.

This policy uses ground-truth transforms (available during training with
``ground_truth:=true``) to drive the robot toward the target port and
insert the cable.  It outputs velocity twist commands in the same action
space as the ACT model so that the LeRobot recorder captures compatible
demonstration data.

Usage – run the simulation with ground truth enabled, then record::

    pixi run lerobot-record \
      --robot.type=aic_controller --robot.id=aic \
      --teleop.type=aic_keyboard_ee --teleop.id=aic \
      --robot.teleop_target_mode=cartesian \
      --robot.teleop_frame_id=base_link \
      --dataset.repo_id=${HF_USER}/aic_expert_demos \
      --dataset.single_task="insert cable into port" \
      --dataset.push_to_hub=false \
      --dataset.private=true \
      --play_sounds=false \
      --display_data=true

Or use the standalone collection mode (no LeRobot recorder needed)::

    pixi run ros2 run aic_model aic_model --ros-args \
      -p use_sim_time:=true \
      -p policy:=my_policy.ExpertCollector
"""

import copy
import math
import time

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

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
# Tunable constants
# ---------------------------------------------------------------------------
APPROACH_SPEED = 0.04  # m/s – max linear velocity during approach
ANGULAR_GAIN = 2.0  # proportional gain for orientation correction
LINEAR_GAIN = 1.5  # proportional gain for position correction
APPROACH_Z_OFFSET = 0.10  # m – hover above port before descending
DESCENT_SPEED = 0.015  # m/s – z descent velocity
INSERTION_DEPTH = 0.015  # m
POSITION_TOLERANCE = 0.003  # m – xy alignment tolerance before descent
CONTROL_HZ = 4.0  # must match ACT inference rate

# Collision recovery
STALL_DISTANCE = 0.002  # m – if TCP moves less than this per step, it's stalled
STALL_FORCE = 3.0  # N – minimum force magnitude to consider it a collision
STALL_COUNT_TRIGGER = 3  # consecutive stalled steps before recovery
RECOVERY_LIFT = 0.03  # m – how far to rise on each recovery
RECOVERY_STEPS = 6  # control steps spent rising during recovery


class ExpertCollector(Policy):
    """Ground-truth expert that outputs velocity twists for ACT training data."""

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self._task = None
        self._prev_tcp_pos: np.ndarray | None = None
        self._stall_count = 0
        self.get_logger().info("ExpertCollector initialized")

    # ------------------------------------------------------------------
    # TF helpers (adapted from CheatCode)
    # ------------------------------------------------------------------

    def _wait_for_tf(self, target_frame, source_frame, timeout_sec=10.0):
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time()
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for TF '{source_frame}' -> '{target_frame}' "
                        "— is ground_truth:=true?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"TF '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def _lookup_tf(self, target_frame, source_frame):
        """Return transform or None."""
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                target_frame, source_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None

    # ------------------------------------------------------------------
    # Velocity command helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_twist(lx=0.0, ly=0.0, lz=0.0, ax=0.0, ay=0.0, az=0.0):
        return Twist(
            linear=Vector3(x=lx, y=ly, z=lz),
            angular=Vector3(x=ax, y=ay, z=az),
        )

    def _send_twist(self, twist, move_robot, frame_id="base_link"):
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
        move_robot(motion_update=msg)

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_stall(self, tcp_tf, get_observation) -> bool:
        """Return True if the arm appears stalled against an obstacle.

        A stall is detected when the TCP barely moves between control
        steps while the F/T sensor reads significant force.
        """
        tcp_pos = np.array([
            tcp_tf.transform.translation.x,
            tcp_tf.transform.translation.y,
            tcp_tf.transform.translation.z,
        ])

        if self._prev_tcp_pos is None:
            self._prev_tcp_pos = tcp_pos
            self._stall_count = 0
            return False

        displacement = np.linalg.norm(tcp_pos - self._prev_tcp_pos)
        self._prev_tcp_pos = tcp_pos

        if displacement > STALL_DISTANCE:
            self._stall_count = 0
            return False

        # TCP barely moved — check if there is force (i.e. pushing on something)
        obs = get_observation()
        if obs is not None and hasattr(obs, "wrist_wrench"):
            w = obs.wrist_wrench
            force_mag = math.sqrt(w.force.x ** 2 + w.force.y ** 2 + w.force.z ** 2)
            if force_mag > STALL_FORCE:
                self._stall_count += 1
            else:
                self._stall_count = 0
        else:
            self._stall_count += 1  # no sensor data — be cautious

        if self._stall_count >= STALL_COUNT_TRIGGER:
            self.get_logger().warn(
                f"Stall detected (count={self._stall_count}, "
                f"disp={displacement:.4f}m) — recovering by lifting"
            )
            self._stall_count = 0
            return True

        return False

    # ------------------------------------------------------------------
    # Proportional controller producing velocity commands
    # ------------------------------------------------------------------

    def _compute_approach_twist(self, tcp_tf, port_tf, plug_tf, phase):
        """Compute a velocity twist to move the plug toward the port.

        Returns a Twist in base_link frame that matches the ACT action
        space (6D Cartesian velocity).

        *phase* is one of: "align", "hover", "descend".
        """
        # Current TCP position
        tcp_pos = np.array([
            tcp_tf.transform.translation.x,
            tcp_tf.transform.translation.y,
            tcp_tf.transform.translation.z,
        ])

        # Port position
        port_pos = np.array([
            port_tf.transform.translation.x,
            port_tf.transform.translation.y,
            port_tf.transform.translation.z,
        ])

        # Plug position
        plug_pos = np.array([
            plug_tf.transform.translation.x,
            plug_tf.transform.translation.y,
            plug_tf.transform.translation.z,
        ])

        # Offset from TCP to plug tip (gripper holds cable)
        tcp_to_plug = plug_pos - tcp_pos

        # Target TCP position = port - tcp_to_plug offset
        if phase == "descend":
            target_tcp = port_pos - tcp_to_plug
        else:
            # Hover above the port
            target_above = port_pos.copy()
            target_above[2] += APPROACH_Z_OFFSET
            target_tcp = target_above - tcp_to_plug

        # Position error
        pos_error = target_tcp - tcp_pos

        # Orientation error (simplified: use quaternion difference)
        q_port = np.array([
            port_tf.transform.rotation.w,
            port_tf.transform.rotation.x,
            port_tf.transform.rotation.y,
            port_tf.transform.rotation.z,
        ])
        q_plug = np.array([
            plug_tf.transform.rotation.w,
            plug_tf.transform.rotation.x,
            plug_tf.transform.rotation.y,
            plug_tf.transform.rotation.z,
        ])
        # Quaternion error: q_err = q_port * q_plug_inv
        q_plug_inv = np.array([-q_plug[0], q_plug[1], q_plug[2], q_plug[3]])
        q_err = quaternion_multiply(q_port, q_plug_inv)
        # Convert to axis-angle (small angle approx: angular_vel ~ 2 * vector part)
        if q_err[0] < 0:
            q_err = -q_err
        ang_vel = ANGULAR_GAIN * 2.0 * q_err[1:4]

        # Proportional velocity with clamping
        lin_vel = LINEAR_GAIN * pos_error
        if phase == "descend":
            # Clamp z and xy independently so xy correction isn't starved
            lin_vel[2] = np.clip(lin_vel[2], -DESCENT_SPEED, DESCENT_SPEED)
            xy_speed = np.linalg.norm(lin_vel[:2])
            if xy_speed > APPROACH_SPEED:
                lin_vel[:2] *= APPROACH_SPEED / xy_speed
        else:
            speed = np.linalg.norm(lin_vel)
            if speed > APPROACH_SPEED:
                lin_vel = lin_vel * (APPROACH_SPEED / speed)

        # Clamp angular velocity
        ang_speed = np.linalg.norm(ang_vel)
        max_ang = 0.3  # rad/s
        if ang_speed > max_ang:
            ang_vel = ang_vel * (max_ang / ang_speed)

        return self._make_twist(
            lx=float(lin_vel[0]),
            ly=float(lin_vel[1]),
            lz=float(lin_vel[2]),
            ax=float(ang_vel[0]),
            ay=float(ang_vel[1]),
            az=float(ang_vel[2]),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.get_logger().info(f"ExpertCollector.insert_cable() — Task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Wait for ground-truth TF
        for frame in [port_frame, plug_frame]:
            if not self._wait_for_tf("base_link", frame):
                send_feedback("Ground-truth TF not available")
                return False

        # ------- Phase 1: Safe approach above port -------
        # Sub-phases to avoid collisions with the task board:
        #   1a "lift"      – rise straight up to safe clearance height
        #   1b "translate"  – move laterally at safe height (no descent)
        #   1c "lower"     – descend to hover height once xy-aligned
        # If the arm stalls against an obstacle it lifts automatically.
        send_feedback("Expert: approaching port")
        self.get_logger().info("Phase 1: safe approach above port")
        self._prev_tcp_pos = None
        self._stall_count = 0
        recovery_steps_remaining = 0

        start = time.time()
        while time.time() - start < 25.0:
            loop_start = time.time()

            tcp_tf = self._lookup_tf("base_link", "gripper/tcp")
            port_tf = self._lookup_tf("base_link", port_frame)
            plug_tf = self._lookup_tf("base_link", plug_frame)
            if tcp_tf is None or port_tf is None or plug_tf is None:
                self.sleep_for(0.1)
                continue

            # --- Collision recovery: lift straight up ---
            if recovery_steps_remaining > 0:
                twist = self._make_twist(lz=APPROACH_SPEED)
                self._send_twist(twist, move_robot)
                recovery_steps_remaining -= 1
                send_feedback(
                    f"Expert: collision recovery, lifting "
                    f"({recovery_steps_remaining} steps left)"
                )
                elapsed = time.time() - loop_start
                time.sleep(max(0, 1.0 / CONTROL_HZ - elapsed))
                continue

            # --- Check for stall (collision) ---
            if self._check_stall(tcp_tf, get_observation):
                recovery_steps_remaining = RECOVERY_STEPS
                send_feedback("Expert: stall detected, lifting to recover")
                continue

            plug_pos = np.array([
                plug_tf.transform.translation.x,
                plug_tf.transform.translation.y,
                plug_tf.transform.translation.z,
            ])
            port_pos = np.array([
                port_tf.transform.translation.x,
                port_tf.transform.translation.y,
                port_tf.transform.translation.z,
            ])

            safe_z = port_pos[2] + APPROACH_Z_OFFSET
            xy_error = np.linalg.norm(plug_pos[:2] - port_pos[:2])

            if plug_pos[2] < safe_z - 0.005:
                # 1a: Below safe height — rise straight up first
                twist = self._make_twist(lz=APPROACH_SPEED)
                send_feedback(f"Expert: lifting, dz={safe_z - plug_pos[2]:.3f}m")
            elif xy_error > POSITION_TOLERANCE:
                # 1b: At safe height but not aligned — translate only
                twist = self._compute_approach_twist(
                    tcp_tf, port_tf, plug_tf, "hover"
                )
                # Zero out z velocity to stay at current height
                twist.linear.z = 0.0
                send_feedback(f"Expert: translating, xy_err={xy_error:.4f}m")
            else:
                # 1c: Aligned in xy — lower to hover height
                z_above = plug_pos[2] - port_pos[2]
                if z_above < APPROACH_Z_OFFSET + 0.01:
                    self.get_logger().info(
                        f"Aligned above port: xy_err={xy_error:.4f}m, "
                        f"z_above={z_above:.4f}m"
                    )
                    break
                twist = self._compute_approach_twist(
                    tcp_tf, port_tf, plug_tf, "hover"
                )
                send_feedback(
                    f"Expert: lowering, z_above={z_above:.4f}m"
                )

            self._send_twist(twist, move_robot)

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1.0 / CONTROL_HZ - elapsed))

        # ------- Phase 2: Descend and insert -------
        send_feedback("Expert: descending into port")
        self.get_logger().info("Phase 2: descending into port")

        start = time.time()
        while time.time() - start < 20.0:
            loop_start = time.time()

            tcp_tf = self._lookup_tf("base_link", "gripper/tcp")
            port_tf = self._lookup_tf("base_link", port_frame)
            plug_tf = self._lookup_tf("base_link", plug_frame)
            if tcp_tf is None or port_tf is None or plug_tf is None:
                self.sleep_for(0.1)
                continue

            plug_z = plug_tf.transform.translation.z
            port_z = port_tf.transform.translation.z
            z_remaining = plug_z - port_z

            twist = self._compute_approach_twist(tcp_tf, port_tf, plug_tf, "descend")
            self._send_twist(twist, move_robot)
            send_feedback(f"Expert: inserting, z_remaining={z_remaining:.4f}m")

            if z_remaining < -INSERTION_DEPTH:
                self.get_logger().info(
                    f"Insertion complete: plug {abs(z_remaining)*1000:.1f}mm past port"
                )
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1.0 / CONTROL_HZ - elapsed))

        # Hold position briefly to let connector settle
        self._send_twist(self._make_twist(), move_robot)
        self.sleep_for(2.0)

        self.get_logger().info("ExpertCollector.insert_cable() done")
        return True
