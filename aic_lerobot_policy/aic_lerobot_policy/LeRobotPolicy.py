#
#  Copyright (C) 2026
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from geometry_msgs.msg import Twist, Vector3, Wrench
from rclpy.node import Node
from safetensors.torch import load_file

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

NORMALIZER_FILENAME = "policy_preprocessor_step_3_normalizer_processor.safetensors"


class LeRobotPolicy(Policy):
    """Generic LeRobot-policy runtime for the AIC framework.

    Loads any LeRobot-trained policy (ACT, Diffusion, SmolVLA, ...) from either
    a HuggingFace repo id or a local checkpoint directory, and drives the
    robot's Cartesian twist controller with the policy's action output.

    Observation layout (26D state, 3 cameras) matches the ground-truth dataset
    produced by ``ground_truth_collection/record_trials_v2_simrestart.py``.
    """

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        self.repo_id = self._declare_str(parent_node, "lerobot_repo_id", "")
        self.episode_duration = self._declare_float(
            parent_node, "episode_duration_sec", 30.0
        )
        self.control_rate_hz = self._declare_float(parent_node, "control_rate_hz", 4.0)
        self.image_scale_override = self._declare_float(
            parent_node, "image_scale_override", 0.0
        )
        device_str = self._declare_str(parent_node, "device", "")

        if not self.repo_id:
            raise RuntimeError(
                "lerobot_repo_id parameter is required (HF repo id or local path)."
            )

        self.device = torch.device(
            device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._ckpt_path = self._resolve_checkpoint(self.repo_id)
        self.get_logger().info(
            f"Loading LeRobot checkpoint from {self._ckpt_path} onto {self.device}"
        )

        config = PreTrainedConfig.from_pretrained(self._ckpt_path)
        config.device = str(self.device)
        policy_cls = get_policy_class(config.type)
        self.get_logger().info(
            f"Detected policy type '{config.type}' -> {policy_cls.__name__}"
        )

        self.policy = policy_cls.from_pretrained(self._ckpt_path, config=config)
        self.policy.to(self.device).eval()

        with torch.no_grad():
            total_params = sum(p.numel() for p in self.policy.parameters())
            total_abs = sum(p.abs().sum().item() for p in self.policy.parameters())
        self.get_logger().info(
            f"Loaded policy with {total_params} params, sum|p|={total_abs:.2f} "
            f"(expect much greater than 0 if weights loaded)."
        )

        self.input_features = self.policy.config.input_features
        self.output_features = self.policy.config.output_features

        self.image_feature_to_aic_field = self._map_cameras(self.input_features)
        self.get_logger().info(
            f"Camera mapping: {self.image_feature_to_aic_field}"
        )

        state_feat = self.input_features.get("observation.state")
        if state_feat is None or int(np.prod(state_feat.shape)) != 26:
            raise RuntimeError(
                "Policy observation.state shape "
                f"{None if state_feat is None else state_feat.shape} != (26,); "
                "LeRobotPolicy expects the 26D layout from record_trials_v2 "
                "(TCP pose 7 + TCP vel 6 + TCP error 6 + joint pos 7)."
            )

        self._load_stats()

        action_feat = self.output_features.get("action")
        if action_feat is None:
            raise RuntimeError("Policy does not expose an 'action' output feature.")
        self._action_dim = int(np.prod(action_feat.shape))
        if self._action_dim < 6:
            raise RuntimeError(
                f"Policy action dim {self._action_dim} < 6; cannot produce a 6D twist."
            )
        if self._action_dim > 6:
            self.get_logger().warning(
                f"Policy action dim {self._action_dim} > 6; using first 6 as twist "
                "and ignoring the rest."
            )

        self.get_logger().info("LeRobotPolicy initialized.")

    @staticmethod
    def _declare_str(node: Node, name: str, default: str) -> str:
        return (
            node.declare_parameter(name, default)
            .get_parameter_value()
            .string_value
        )

    @staticmethod
    def _declare_float(node: Node, name: str, default: float) -> float:
        return (
            node.declare_parameter(name, float(default))
            .get_parameter_value()
            .double_value
        )

    @staticmethod
    def _resolve_checkpoint(repo_id: str) -> Path:
        p = Path(repo_id).expanduser()
        if p.exists():
            return p.resolve()
        return Path(snapshot_download(repo_id=repo_id))

    @staticmethod
    def _map_cameras(input_features) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for key in input_features:
            if "image" not in key:
                continue
            low = key.lower()
            if "left" in low:
                mapping[key] = "left_image"
            elif "center" in low:
                mapping[key] = "center_image"
            elif "right" in low:
                mapping[key] = "right_image"
            else:
                raise RuntimeError(
                    f"Cannot map image feature '{key}' to an AIC camera "
                    "(expected 'left', 'center', or 'right' in the key name)."
                )
        if not mapping:
            raise RuntimeError(
                "Policy has no image input features; LeRobotPolicy requires "
                "at least one 'observation.images.*' input."
            )
        return mapping

    def _load_stats(self) -> None:
        stats_path = self._ckpt_path / NORMALIZER_FILENAME
        if not stats_path.exists():
            raise RuntimeError(
                f"Normalizer stats file not found at {stats_path}. "
                "LeRobotPolicy expects LeRobot's preprocessor pipeline artifacts."
            )
        stats = load_file(str(stats_path))

        def stat(key: str, shape):
            if key not in stats:
                raise RuntimeError(
                    f"Expected normalization key '{key}' missing from {stats_path.name}. "
                    f"Available keys: {sorted(stats.keys())}"
                )
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            key: {
                "mean": stat(f"{key}.mean", (1, 3, 1, 1)),
                "std": stat(f"{key}.std", (1, 3, 1, 1)),
            }
            for key in self.image_feature_to_aic_field
        }
        self.state_mean = stat("observation.state.mean", (1, -1))
        self.state_std = stat("observation.state.std", (1, -1))
        self.action_mean = stat("action.mean", (1, -1))
        self.action_std = stat("action.std", (1, -1))

    def _image_to_tensor(
        self,
        raw_img,
        expected_shape,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        expected_h, expected_w = int(expected_shape[-2]), int(expected_shape[-1])

        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

        if self.image_scale_override > 0.0:
            img_np = cv2.resize(
                img_np,
                None,
                fx=self.image_scale_override,
                fy=self.image_scale_override,
                interpolation=cv2.INTER_AREA,
            )
        if img_np.shape[0] != expected_h or img_np.shape[1] != expected_w:
            img_np = cv2.resize(
                img_np,
                (expected_w, expected_h),
                interpolation=cv2.INTER_AREA,
            )

        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, torch.Tensor]:
        obs: Dict[str, torch.Tensor] = {}

        for feature_key, aic_field in self.image_feature_to_aic_field.items():
            raw_img = getattr(obs_msg, aic_field)
            obs[feature_key] = self._image_to_tensor(
                raw_img,
                self.input_features[feature_key].shape,
                self.img_stats[feature_key]["mean"],
                self.img_stats[feature_key]["std"],
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
        raw_state = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        obs["observation.state"] = (raw_state - self.state_mean) / self.state_std
        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.policy.reset()
        self.get_logger().info(
            f"LeRobotPolicy.insert_cable() start; duration {self.episode_duration}s, "
            f"rate {self.control_rate_hz} Hz"
        )
        self.get_logger().info(
            f"action_mean={self.action_mean.cpu().numpy().flatten().tolist()}"
        )
        self.get_logger().info(
            f"action_std={self.action_std.cpu().numpy().flatten().tolist()}"
        )

        start = time.time()
        dt = 1.0 / self.control_rate_hz
        iteration = 0

        while time.time() - start < self.episode_duration:
            loop_t0 = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warning("No observation received.")
                time.sleep(max(0.0, dt - (time.time() - loop_t0)))
                continue

            obs_tensors = self.prepare_observations(obs_msg)

            if iteration == 0:
                for k, v in obs_tensors.items():
                    self.get_logger().info(
                        f"obs[{k}] shape={tuple(v.shape)} "
                        f"min={v.min().item():.4f} max={v.max().item():.4f} "
                        f"mean={v.mean().item():.4f}"
                    )

            with torch.inference_mode():
                norm_action = self.policy.select_action(obs_tensors)

            raw_action = (norm_action * self.action_std) + self.action_mean
            action = raw_action[0].cpu().numpy()

            if iteration % int(max(1, self.control_rate_hz)) == 0:
                self.get_logger().info(
                    f"[iter {iteration}] norm={norm_action[0].cpu().numpy().tolist()} "
                    f"action={action.tolist()}"
                )

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            move_robot(motion_update=self._velocity_motion_update(twist))
            send_feedback("in progress...")

            iteration += 1
            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, dt - elapsed))

        self.get_logger().info("LeRobotPolicy.insert_cable() exiting.")
        return True

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
