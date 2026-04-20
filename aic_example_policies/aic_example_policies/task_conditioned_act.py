from dataclasses import dataclass

import einops
import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

TASK_PLUG_TYPE_KEY = "task.plug_type_index"
TASK_PORT_NAME_KEY = "task.port_name_index"
TASK_TARGET_MODULE_KEY = "task.target_module_name_index"
TASK_INDEX_KEYS = (
    TASK_PLUG_TYPE_KEY,
    TASK_PORT_NAME_KEY,
    TASK_TARGET_MODULE_KEY,
)

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


def _normalize_task_value(value: str) -> str:
    return value.strip().lower()


def _lookup_index(name: str, value: str, mapping: dict[str, int]) -> int:
    normalized = _normalize_task_value(value)
    if normalized not in mapping:
        valid_values = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported {name} '{value}'. Expected one of: {valid_values}")
    return mapping[normalized]


def encode_task_fields(plug_type: str, port_name: str, target_module_name: str) -> dict[str, int]:
    return {
        TASK_PLUG_TYPE_KEY: _lookup_index("plug_type", plug_type, PLUG_TYPE_TO_INDEX),
        TASK_PORT_NAME_KEY: _lookup_index("port_name", port_name, PORT_NAME_TO_INDEX),
        TASK_TARGET_MODULE_KEY: _lookup_index(
            "target_module_name", target_module_name, TARGET_MODULE_TO_INDEX
        ),
    }


def task_indices_to_batch(task_indices: dict[str, int], device: torch.device) -> dict[str, Tensor]:
    return {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in task_indices.items()
    }


def is_task_conditioned_act_config(config_dict: dict) -> bool:
    if config_dict.get("type") == "task_conditioned_act":
        return True
    return any(field in config_dict for field in TASK_CONDITIONED_CONFIG_FIELDS)


@PreTrainedConfig.register_subclass("task_conditioned_act")
@dataclass
class TaskConditionedACTConfig(ACTConfig):
    task_embed_dim: int = 16
    task_mlp_hidden_dim: int = 128
    plug_type_vocab_size: int = 2
    port_name_vocab_size: int = 3
    target_module_vocab_size: int = 7

    def __post_init__(self):
        super().__post_init__()
        if self.task_embed_dim <= 0:
            raise ValueError("task_embed_dim must be positive.")
        if self.task_mlp_hidden_dim <= 0:
            raise ValueError("task_mlp_hidden_dim must be positive.")


class TaskEmbedding(nn.Module):
    def __init__(self, config: TaskConditionedACTConfig):
        super().__init__()
        embed_dim = config.task_embed_dim
        self.plug_type_embedding = nn.Embedding(config.plug_type_vocab_size, embed_dim)
        self.port_name_embedding = nn.Embedding(config.port_name_vocab_size, embed_dim)
        self.target_module_embedding = nn.Embedding(config.target_module_vocab_size, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 3, config.task_mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.task_mlp_hidden_dim, config.dim_model),
        )

    def forward(
        self,
        plug_type_indices: Tensor,
        port_name_indices: Tensor,
        target_module_indices: Tensor,
    ) -> Tensor:
        plug_type_indices = plug_type_indices.view(-1)
        port_name_indices = port_name_indices.view(-1)
        target_module_indices = target_module_indices.view(-1)

        task_embedding = torch.cat(
            [
                self.plug_type_embedding(plug_type_indices),
                self.port_name_embedding(port_name_indices),
                self.target_module_embedding(target_module_indices),
            ],
            dim=-1,
        )
        return self.projection(task_embedding)


class TaskConditionedACT(ACT):
    def __init__(self, config: TaskConditionedACTConfig):
        super().__init__(config)
        self.task_embedding = TaskEmbedding(config)
        self.encoder_1d_feature_pos_embed = nn.Embedding(
            self.encoder_1d_feature_pos_embed.num_embeddings + 1,
            config.dim_model,
        )

    def _reference_tensor(self, batch: dict[str, Tensor]) -> Tensor:
        if OBS_IMAGES in batch:
            return batch[OBS_IMAGES][0]
        if OBS_ENV_STATE in batch:
            return batch[OBS_ENV_STATE]
        return batch[OBS_STATE]

    def _task_token_from_batch(self, batch: dict[str, Tensor]) -> Tensor:
        missing_keys = [key for key in TASK_INDEX_KEYS if key not in batch]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise KeyError(f"Missing task conditioning inputs: {missing}")

        return self.task_embedding(
            batch[TASK_PLUG_TYPE_KEY],
            batch[TASK_PORT_NAME_KEY],
            batch[TASK_TARGET_MODULE_KEY],
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        reference_tensor = self._reference_tensor(batch)
        batch_size = reference_tensor.shape[0]

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=reference_tensor.device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                reference_tensor.device
            )

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        encoder_in_tokens.append(self._task_token_from_batch(batch))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


class TaskConditionedACTPolicy(ACTPolicy):
    config_class = TaskConditionedACTConfig
    name = "task_conditioned_act"

    def __init__(self, config: TaskConditionedACTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = TaskConditionedACT(config)
