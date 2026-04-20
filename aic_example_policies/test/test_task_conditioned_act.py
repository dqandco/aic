import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from aic_example_policies.task_conditioned_act import (
    TASK_PLUG_TYPE_KEY,
    TASK_PORT_NAME_KEY,
    TASK_TARGET_MODULE_KEY,
    TaskConditionedACTConfig,
    TaskConditionedACTPolicy,
    encode_task_fields,
)


def test_encode_task_fields_supports_current_task_names():
    assert encode_task_fields("sfp", "sfp_port_1", "nic_card_mount_4") == {
        TASK_PLUG_TYPE_KEY: 0,
        TASK_PORT_NAME_KEY: 1,
        TASK_TARGET_MODULE_KEY: 4,
    }
    assert encode_task_fields("sc", "sc_port_base", "sc_port_0") == {
        TASK_PLUG_TYPE_KEY: 1,
        TASK_PORT_NAME_KEY: 2,
        TASK_TARGET_MODULE_KEY: 5,
    }


def test_encode_task_fields_accepts_nic_card_alias():
    assert encode_task_fields("sfp", "sfp_port_0", "nic_card_2")[TASK_TARGET_MODULE_KEY] == 2


def test_task_conditioned_policy_predicts_chunk():
    config = TaskConditionedACTConfig(
        use_vae=False,
        chunk_size=4,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        task_mlp_hidden_dim=32,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )

    policy = TaskConditionedACTPolicy(config)
    batch = {
        OBS_STATE: torch.randn(2, 6),
        OBS_ENV_STATE: torch.randn(2, 3),
        TASK_PLUG_TYPE_KEY: torch.tensor([0, 1], dtype=torch.long),
        TASK_PORT_NAME_KEY: torch.tensor([1, 2], dtype=torch.long),
        TASK_TARGET_MODULE_KEY: torch.tensor([4, 5], dtype=torch.long),
    }

    actions = policy.predict_action_chunk(batch)

    assert actions.shape == (2, 4, 7)
