from pathlib import Path

import torch

from my_policy.gazebo_rl.adapter import (
    ResidualACTAdapter,
    ResidualAdapterConfig,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from my_policy.gazebo_rl.runner import (
    EpisodeSpec,
    _diagnose_launch_failure,
    _wrap_eval_command,
)
from my_policy.gazebo_rl.scoring import build_training_reward, load_score_summary


def test_load_score_summary_and_reward(tmp_path: Path):
    scoring_yaml = tmp_path / "scoring.yaml"
    scoring_yaml.write_text(
        """
total: 43.5
trial_1:
  tier_1:
    score: 1
    message: ok
  tier_2:
    score: 5.5
    message: partial progress
    categories:
      duration:
        score: 2.0
      trajectory smoothness:
        score: 1.5
      trajectory efficiency:
        score: 1.0
      insertion force:
        score: 0.0
      contacts:
        score: 1.0
  tier_3:
    score: 37.0
    message: close
""".strip(),
        encoding="utf-8",
    )

    summary = load_score_summary(scoring_yaml)

    assert summary.total == 43.5
    assert summary.trials["trial_1"].tier_3.score == 37.0
    assert build_training_reward(summary) > summary.total


def test_adapter_checkpoint_round_trip(tmp_path: Path):
    adapter = ResidualACTAdapter(ResidualAdapterConfig(hidden_dim=8))
    vector = torch.arange(adapter.parameter_vector().numel(), dtype=torch.float32) * 0.001
    adapter.load_parameter_vector(vector)

    checkpoint_path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(checkpoint_path, adapter, metadata={"tag": "roundtrip"})
    restored, metadata = load_adapter_checkpoint(checkpoint_path)

    assert metadata["tag"] == "roundtrip"
    assert torch.allclose(restored.parameter_vector(), vector)


def test_diagnose_launch_failure_missing_container():
    reason = _diagnose_launch_failure(
        "Create it now, out of image registry.fedoraproject.org/fedora-toolbox:latest?",
        "aic_eval",
    )
    assert "could not find" in reason
    assert "DBX_CONTAINER_MANAGER=docker" in reason


def test_wrap_eval_command_respects_inside_container_mode(tmp_path: Path):
    spec = EpisodeSpec(
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        inside_eval_container=True,
    )
    assert _wrap_eval_command(spec, "echo ok") == ["bash", "-lc", "echo ok"]
