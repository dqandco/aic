import json
import os
import time
from pathlib import Path
from typing import Dict

import cv2
import draccus
import numpy as np
import torch
from geometry_msgs.msg import Twist, Vector3, Wrench
from huggingface_hub import snapshot_download
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
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

from my_policy.gazebo_rl.adapter import (
    ResidualACTAdapter,
    ResidualAdapterConfig,
    load_adapter_checkpoint,
)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

TASK_VECTOR_DIM = 12
PLUG_TYPE_TO_INDEX = {
    "sfp": 0,
    "sc": 1,
}
PORT_NAME_TO_INDEX = {
    "sfp_port_0": 0,
    "sfp_port_1": 1,
    "sc_port_base": 2,
    "sc_port": 2,
}
TARGET_MODULE_TO_INDEX = {
    **{f"nic_card_mount_{idx}": idx for idx in range(5)},
    **{f"nic_card_{idx}": idx for idx in range(5)},
    "sc_port_0": 5,
    "sc_port_1": 6,
}
TASK_CONDITIONED_CONFIG_FIELDS = {
    "task_embed_dim",
    "task_mlp_hidden_dim",
    "plug_type_vocab_size",
    "port_name_vocab_size",
    "target_module_vocab_size",
}

try:
    from aic_example_policies.task_conditioned_act import (
        TaskConditionedACTConfig,
        TaskConditionedACTPolicy,
    )
except ImportError:
    TaskConditionedACTConfig = None
    TaskConditionedACTPolicy = None


def is_task_conditioned_act_config(config_dict: dict) -> bool:
    if config_dict.get("type") == "task_conditioned_act":
        return True
    return any(field in config_dict for field in TASK_CONDITIONED_CONFIG_FIELDS)


class ResidualACT(Policy):
    """ACT policy with an optional learned residual adapter on the action output."""

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_repo_id = os.environ.get("ACT_MODEL_REPO", "grkw/aic_act_policy")
        self.adapter_path = os.environ.get("ACT_ADAPTER_PATH")
        self.adapter_scale = float(os.environ.get("ACT_ADAPTER_SCALE", "1.0"))
        self.image_scaling = 0.25

        policy_path = Path(
            snapshot_download(
                repo_id=self.base_repo_id,
                allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
            )
        )
        with open(policy_path / "config.json", "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
            self.task_conditioned = is_task_conditioned_act_config(config_dict)
            config_dict.pop("type", None)

        if self.task_conditioned and (
            TaskConditionedACTConfig is None or TaskConditionedACTPolicy is None
        ):
            raise ImportError(
                "This checkpoint requires task-conditioned ACT support, but "
                "`aic_example_policies.task_conditioned_act` is not available in the "
                "current environment. Reinstall `ros-kilted-aic-example-policies` or "
                "use a plain ACT checkpoint."
            )

        config_class = TaskConditionedACTConfig if self.task_conditioned else ACTConfig
        policy_class = TaskConditionedACTPolicy if self.task_conditioned else ACTPolicy
        config = draccus.decode(config_class, config_dict)

        self.policy = policy_class(config)
        self.policy.load_state_dict(load_file(policy_path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        stats = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        def get_stat(key: str, shape: tuple[int, ...]) -> torch.Tensor:
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

        self.adapter = self._load_adapter()
        self.get_logger().info(
            f"ResidualACT loaded from {policy_path} on {self.device}; "
            f"adapter={self.adapter_path or 'none'} scale={self.adapter_scale:.3f}"
        )

    def _load_adapter(self) -> ResidualACTAdapter:
        if self.adapter_path:
            adapter, _ = load_adapter_checkpoint(self.adapter_path, self.device)
        else:
            adapter = ResidualACTAdapter(ResidualAdapterConfig())
            adapter.to(self.device)
        adapter.eval()
        return adapter

    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        scale: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
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

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, torch.Tensor]:
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

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
        raw_state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        obs["observation.state"] = (raw_state - self.state_mean) / self.state_std
        return obs

    def prepare_task_vector(self, task: Task) -> torch.Tensor:
        task_vector = torch.zeros((1, TASK_VECTOR_DIM), dtype=torch.float32, device=self.device)
        task_vector[0, PLUG_TYPE_TO_INDEX[task.plug_type]] = 1.0
        task_vector[0, 2 + PORT_NAME_TO_INDEX[task.port_name]] = 1.0
        task_vector[0, 5 + TARGET_MODULE_TO_INDEX[task.target_module_name]] = 1.0
        return task_vector

    def _create_motion_update(self, action_xyzrpy: np.ndarray) -> MotionUpdate:
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = Twist(
            linear=Vector3(
                x=float(action_xyzrpy[0]),
                y=float(action_xyzrpy[1]),
                z=float(action_xyzrpy[2]),
            ),
            angular=Vector3(
                x=float(action_xyzrpy[3]),
                y=float(action_xyzrpy[4]),
                z=float(action_xyzrpy[5]),
            ),
        )
        motion_update_msg.header.frame_id = "base_link"
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()
        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()
        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )
        return motion_update_msg

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.policy.reset()
        start_time = time.time()

        while time.time() - start_time < 30.0:
            loop_start = time.time()
            observation_msg = get_observation()
            if observation_msg is None:
                continue

            obs_tensors = self.prepare_observations(observation_msg)
            task_vector = self.prepare_task_vector(task)

            if self.task_conditioned:
                obs_tensors.update(
                    {
                        "task.plug_type_index": torch.tensor(
                            [PLUG_TYPE_TO_INDEX[task.plug_type]],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        "task.port_name_index": torch.tensor(
                            [PORT_NAME_TO_INDEX[task.port_name]],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        "task.target_module_name_index": torch.tensor(
                            [TARGET_MODULE_TO_INDEX[task.target_module_name]],
                            dtype=torch.long,
                            device=self.device,
                        ),
                    }
                )

            with torch.inference_mode():
                normalized_action = self.policy.select_action(obs_tensors)
                raw_action = (normalized_action * self.action_std) + self.action_mean
                base_action = raw_action[:, :6]
                residual = self.adapter(
                    obs_tensors["observation.state"],
                    task_vector,
                    base_action,
                )
                adapted_action = raw_action.clone()
                adapted_action[:, :6] = base_action + residual * self.adapter_scale

            action_np = adapted_action[0].cpu().numpy()
            move_robot(motion_update=self._create_motion_update(action_np[:6]))
            send_feedback("Residual ACT in progress...")

            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.25 - elapsed))

        return True
