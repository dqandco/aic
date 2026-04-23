#
#  Copyright (C) 2026
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

import math

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
    Transform,
    Twist,
    Vector3,
    Wrench,
)
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class ExpertTwist(Policy):
    """Ground-truth cable-insertion expert that publishes velocity twists.

    Uses the same pose-planning logic as ``aic_example_policies.ros.CheatCode``
    (quaternion alignment, plug-tip offset compensation, XY error integrators)
    but converts the desired pose into a proportional velocity twist each
    tick so that the data-collection pipeline captures meaningful actions
    (``MotionUpdate.velocity`` with ``MODE_VELOCITY``).
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        self.control_hz = self._p_float(parent_node, "control_hz", 20.0)
        self.approach_duration_sec = self._p_float(
            parent_node, "approach_duration_sec", 5.0
        )
        self.z_offset_approach = self._p_float(parent_node, "z_offset_approach", 0.2)
        self.descent_speed_m_per_s = self._p_float(
            parent_node, "descent_speed_m_per_s", 0.01
        )
        self.insertion_depth_m = self._p_float(
            parent_node, "insertion_depth_m", 0.015
        )
        self.settle_sec = self._p_float(parent_node, "settle_sec", 5.0)
        self.kp_linear = self._p_float(parent_node, "kp_linear", 2.0)
        self.kp_angular = self._p_float(parent_node, "kp_angular", 2.0)
        self.max_linear_speed = self._p_float(parent_node, "max_linear_speed", 0.1)
        self.max_angular_speed = self._p_float(parent_node, "max_angular_speed", 1.0)
        self.max_ff_linear_jump = self._p_float(
            parent_node, "max_ff_linear_jump", 0.1
        )
        self.max_ff_angular_jump = self._p_float(
            parent_node, "max_ff_angular_jump", 1.0
        )
        self.i_gain = self._p_float(parent_node, "i_gain", 0.15)
        self._max_integrator_windup = self._p_float(
            parent_node, "max_integrator_windup", 0.05
        )

        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._task: Task | None = None

    @staticmethod
    def _p_float(node, name: str, default: float) -> float:
        return (
            node.declare_parameter(name, float(default))
            .get_parameter_value()
            .double_value
        )

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lerp_pose(a: Pose, b: Pose, frac: float) -> Pose:
        """Linearly interpolate positions and slerp orientations."""
        qa = (a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z)
        qb = (b.orientation.w, b.orientation.x, b.orientation.y, b.orientation.z)
        qs = quaternion_slerp(qa, qb, frac)
        return Pose(
            position=Point(
                x=a.position.x + frac * (b.position.x - a.position.x),
                y=a.position.y + frac * (b.position.y - a.position.y),
                z=a.position.z + frac * (b.position.z - a.position.z),
            ),
            orientation=Quaternion(w=qs[0], x=qs[1], y=qs[2], z=qs[3]),
        )

    # ------------------------------------------------------------------
    # TF helpers (pattern from CheatCode)
    # ------------------------------------------------------------------

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
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
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... "
                        "-- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    # ------------------------------------------------------------------
    # Desired gripper pose (verbatim from CheatCode.calc_gripper_pose)
    # ------------------------------------------------------------------

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link", "gripper/tcp", Time()
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (port_transform.translation.x, port_transform.translation.y)
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(
                np.clip(
                    self._tip_x_error_integrator + tip_x_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )
            self._tip_y_error_integrator = float(
                np.clip(
                    self._tip_y_error_integrator + tip_y_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )

        target_x = port_xy[0] + self.i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + self.i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(x=blend_xyz[0], y=blend_xyz[1], z=blend_xyz[2]),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    # ------------------------------------------------------------------
    # Tracking: feedforward (target velocity) + feedback (pose error)
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_delta_axis_angle(
        q_from: Quaternion, q_to: Quaternion
    ) -> tuple[float, float, float]:
        """Rotation axis*angle (3-vector) that rotates q_from to q_to."""
        qf = (q_from.w, q_from.x, q_from.y, q_from.z)
        qf_inv = (qf[0], -qf[1], -qf[2], -qf[3])
        qt = (q_to.w, q_to.x, q_to.y, q_to.z)
        qe = quaternion_multiply(qt, qf_inv)
        if qe[0] < 0.0:
            qe = tuple(-x for x in qe)
        vnorm = math.sqrt(qe[1] ** 2 + qe[2] ** 2 + qe[3] ** 2)
        if vnorm < 1e-9:
            return 0.0, 0.0, 0.0
        theta = 2.0 * math.atan2(vnorm, qe[0])
        return theta * qe[1] / vnorm, theta * qe[2] / vnorm, theta * qe[3] / vnorm

    @staticmethod
    def _clamp(v: float, lim: float) -> float:
        return max(-lim, min(lim, v))

    def _tracking_twist(
        self,
        current: Pose,
        target: Pose,
        prev_target: Pose | None,
        dt: float,
    ) -> Twist:
        """FF+FB: feedforward from target motion, feedback from pose error."""

        # Feedback: P * pose error
        fb_vx = self.kp_linear * (target.position.x - current.position.x)
        fb_vy = self.kp_linear * (target.position.y - current.position.y)
        fb_vz = self.kp_linear * (target.position.z - current.position.z)
        fb_wx, fb_wy, fb_wz = self._quat_delta_axis_angle(
            current.orientation, target.orientation
        )
        fb_wx *= self.kp_angular
        fb_wy *= self.kp_angular
        fb_wz *= self.kp_angular

        # Feedforward: (target - prev_target) / dt, clamped to reject TF glitches
        if prev_target is None or dt <= 0.0:
            ff_vx = ff_vy = ff_vz = 0.0
            ff_wx = ff_wy = ff_wz = 0.0
        else:
            ff_vx = self._clamp(
                (target.position.x - prev_target.position.x) / dt,
                self.max_ff_linear_jump,
            )
            ff_vy = self._clamp(
                (target.position.y - prev_target.position.y) / dt,
                self.max_ff_linear_jump,
            )
            ff_vz = self._clamp(
                (target.position.z - prev_target.position.z) / dt,
                self.max_ff_linear_jump,
            )
            raw_wx, raw_wy, raw_wz = self._quat_delta_axis_angle(
                prev_target.orientation, target.orientation
            )
            ff_wx = self._clamp(raw_wx / dt, self.max_ff_angular_jump)
            ff_wy = self._clamp(raw_wy / dt, self.max_ff_angular_jump)
            ff_wz = self._clamp(raw_wz / dt, self.max_ff_angular_jump)

        return Twist(
            linear=Vector3(
                x=self._clamp(ff_vx + fb_vx, self.max_linear_speed),
                y=self._clamp(ff_vy + fb_vy, self.max_linear_speed),
                z=self._clamp(ff_vz + fb_vz, self.max_linear_speed),
            ),
            angular=Vector3(
                x=self._clamp(ff_wx + fb_wx, self.max_angular_speed),
                y=self._clamp(ff_wy + fb_wy, self.max_angular_speed),
                z=self._clamp(ff_wz + fb_wz, self.max_angular_speed),
            ),
        )

    # ------------------------------------------------------------------
    # MotionUpdate builder (pattern from RunACT.set_cartesian_twist_target)
    # ------------------------------------------------------------------

    def _velocity_motion_update(
        self, twist: Twist, frame_id: str = "base_link"
    ) -> MotionUpdate:
        msg = MotionUpdate()
        msg.velocity = twist
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
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

    def _current_tcp_pose(self) -> Pose | None:
        try:
            t = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time()
            ).transform
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup for gripper/tcp failed: {ex}")
            return None
        return Pose(
            position=Point(
                x=t.translation.x, y=t.translation.y, z=t.translation.z
            ),
            orientation=Quaternion(
                x=t.rotation.x, y=t.rotation.y, z=t.rotation.z, w=t.rotation.w
            ),
        )

    def _send_twist_to_target(
        self,
        target_pose: Pose,
        move_robot: MoveRobotCallback,
        prev_target: Pose | None,
        dt: float,
    ) -> None:
        current_pose = self._current_tcp_pose()
        if current_pose is None:
            return
        twist = self._tracking_twist(current_pose, target_pose, prev_target, dt)
        try:
            move_robot(motion_update=self._velocity_motion_update(twist))
        except Exception as ex:
            self.get_logger().warn(f"move_robot failed: {ex}")

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
    ) -> bool:
        self.get_logger().info(f"ExpertTwist.insert_cable() task: {task}")
        self._task = task
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0

        port_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link"
        )
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"
        for frame in (port_frame, cable_tip_frame):
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        dt = 1.0 / self.control_hz
        n_approach = max(1, int(round(self.approach_duration_sec * self.control_hz)))
        prev_target: Pose | None = None

        # Snapshot initial TCP pose and compute absolute approach goal. The
        # approach trajectory is a plain lerp between these, so the reference
        # path is independent of arm state (no implicit feedback coupling).
        initial_tcp = self._current_tcp_pose()
        if initial_tcp is None:
            self.get_logger().error("Could not snapshot initial TCP pose.")
            return False
        try:
            approach_goal = self.calc_gripper_pose(
                port_transform,
                slerp_fraction=1.0,
                position_fraction=1.0,
                z_offset=self.z_offset_approach,
                reset_xy_integrator=True,
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not compute approach goal: {ex}")
            return False

        # Approach: snapshot-based lerp
        for t in range(n_approach):
            frac = (t + 1) / n_approach
            target = self._lerp_pose(initial_tcp, approach_goal, frac)
            self._send_twist_to_target(target, move_robot, prev_target, dt)
            prev_target = target
            send_feedback(f"Expert: approach {frac:.2f}")
            self.sleep_for(dt)

        # Descend. Terminates on whichever comes first:
        #   (a) actual plug progress past insertion depth, or
        #   (b) commanded z_offset past -insertion_depth_m (safety fallback).
        z_offset = self.z_offset_approach
        z_step = self.descent_speed_m_per_s * dt
        while z_offset > -self.insertion_depth_m:
            try:
                plug_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", cable_tip_frame, Time()
                ).transform
                if (
                    port_transform.translation.z - plug_tf.translation.z
                    >= self.insertion_depth_m
                ):
                    self.get_logger().info(
                        "Insertion detected via plug Z progress."
                    )
                    break
            except TransformException as ex:
                self.get_logger().warn(f"TF plug-progress check: {ex}")

            z_offset -= z_step
            try:
                target = self.calc_gripper_pose(
                    port_transform,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    z_offset=z_offset,
                    reset_xy_integrator=False,
                )
                self._send_twist_to_target(target, move_robot, prev_target, dt)
                prev_target = target
            except TransformException as ex:
                self.get_logger().warn(f"TF during descent: {ex}")
            send_feedback(f"Expert: descend z_offset={z_offset:.4f}")
            self.sleep_for(dt)

        # Settle
        self.get_logger().info("ExpertTwist: settling...")
        zero = Twist()
        n_settle = max(1, int(round(self.settle_sec * self.control_hz)))
        for _ in range(n_settle):
            try:
                move_robot(motion_update=self._velocity_motion_update(zero))
            except Exception as ex:
                self.get_logger().warn(f"move_robot failed during settle: {ex}")
            self.sleep_for(dt)

        self.get_logger().info("ExpertTwist.insert_cable() exiting.")
        return True
