from dataclasses import replace
from pathlib import Path

import torch

from my_policy.gazebo_rl.adapter import (
    ResidualACTAdapter,
    ResidualAdapterConfig,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from my_policy.gazebo_rl.runner import (
    EpisodeResult,
    EpisodeSpec,
    _check_clock_in_container,
    _check_existing_sim_alive,
    _diagnose_launch_failure,
    _distrobox_command,
    _model_command,
    is_infrastructure_failure,
    wait_for_model_ready,
)
from my_policy.gazebo_rl import runner as runner_module
from my_policy.gazebo_rl.scoring import (
    RewardWeights,
    build_training_reward,
    load_score_summary,
)


def _make_result(**overrides) -> EpisodeResult:
    base = dict(
        success=True,
        reward=10.0,
        summary=None,
        scoring_path=None,
        results_dir=Path("/tmp/results"),
        engine_return_code=0,
        model_return_code=0,
        elapsed_s=12.0,
        failure_reason=None,
    )
    base.update(overrides)
    return EpisodeResult(**base)


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


def test_distrobox_command_wraps_with_distrobox(tmp_path: Path):
    spec = EpisodeSpec(
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
    )
    cmd = _distrobox_command(spec, "echo ok")
    assert cmd[:4] == ["distrobox", "enter", "aic_eval", "--"]
    assert cmd[-3:] == ["bash", "-lc", "echo ok"]


def test_distrobox_command_use_root_adds_dash_r(tmp_path: Path):
    spec = EpisodeSpec(
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
    )
    cmd = _distrobox_command(spec, "echo ok", use_root=True)
    assert cmd[:5] == ["distrobox", "enter", "-r", "aic_eval", "--"]


def test_distrobox_command_respects_spec_use_root(tmp_path: Path):
    spec = EpisodeSpec(
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        distrobox_use_root=True,
    )
    cmd = _distrobox_command(spec, "echo ok")
    assert cmd[:5] == ["distrobox", "enter", "-r", "aic_eval", "--"]


def test_model_command_uses_pixi_run(tmp_path: Path):
    spec = EpisodeSpec(
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        policy="my_policy.ResidualACT",
    )
    cmd = _model_command(spec)
    assert cmd[:4] == ["pixi", "run", "ros2", "run"]
    assert "aic_model" in cmd
    assert "policy:=my_policy.ResidualACT" in cmd


def test_infrastructure_failure_engine_nonzero_exit():
    result = _make_result(success=False, engine_return_code=1)
    assert is_infrastructure_failure(result) is True


def test_infrastructure_failure_validation_failed_for_all_trials(tmp_path: Path):
    scoring_yaml = tmp_path / "scoring.yaml"
    scoring_yaml.write_text(
        """
total: 0
trial_1:
  tier_1:
    score: 0
    message: Validation failed
  tier_2:
    score: 0
    message: skipped
    categories:
      duration:
        score: 0
      trajectory smoothness:
        score: 0
      trajectory efficiency:
        score: 0
      insertion force:
        score: 0
      contacts:
        score: 0
  tier_3:
    score: 0
    message: skipped
""".strip(),
        encoding="utf-8",
    )
    summary = load_score_summary(scoring_yaml)
    result = _make_result(success=False, summary=summary)
    assert is_infrastructure_failure(result) is True


def test_infrastructure_failure_clean_episode_is_not_flagged(tmp_path: Path):
    scoring_yaml = tmp_path / "scoring.yaml"
    scoring_yaml.write_text(
        """
total: 12
trial_1:
  tier_1:
    score: 1
    message: ok
  tier_2:
    score: 4
    message: ok
    categories:
      duration:
        score: 1
      trajectory smoothness:
        score: 1
      trajectory efficiency:
        score: 1
      insertion force:
        score: 1
      contacts:
        score: 0
  tier_3:
    score: 7
    message: ok
""".strip(),
        encoding="utf-8",
    )
    summary = load_score_summary(scoring_yaml)
    result = _make_result(summary=summary)
    assert is_infrastructure_failure(result) is False


def test_infrastructure_failure_engine_log_marker():
    result = _make_result()
    assert (
        is_infrastructure_failure(
            result,
            engine_log_tail="GetState service call timed out for /aic_model.",
        )
        is True
    )


def test_episode_spec_defaults_use_existing_sim_true(tmp_path: Path):
    spec = EpisodeSpec(
        config_path=tmp_path / "c.yaml",
        results_dir=tmp_path / "r",
    )
    assert spec.use_existing_sim is True
    assert spec.eval_container == "aic_eval"
    assert spec.existing_sim_check_timeout_s > 0


def test_check_existing_sim_alive_returns_none_when_service_present(
    tmp_path: Path, monkeypatch
):
    spec = EpisodeSpec(
        config_path=tmp_path / "c.yaml",
        results_dir=tmp_path / "r",
        existing_sim_check_timeout_s=1.0,
    )
    monkeypatch.setattr(runner_module, "wait_for_sim", lambda spec, timeout: True)

    assert _check_existing_sim_alive(spec) is None


def test_check_existing_sim_alive_explains_when_missing(tmp_path: Path, monkeypatch):
    spec = EpisodeSpec(
        config_path=tmp_path / "c.yaml",
        results_dir=tmp_path / "r",
        existing_sim_check_timeout_s=1.0,
    )
    monkeypatch.setattr(runner_module, "wait_for_sim", lambda spec, timeout: False)

    reason = _check_existing_sim_alive(spec)
    assert reason is not None
    assert "No running sim detected" in reason
    assert "/entrypoint.sh" in reason
    assert "clean_sim.sh" in reason


def test_check_existing_sim_alive_skips_when_timeout_zero(tmp_path: Path, monkeypatch):
    spec = EpisodeSpec(
        config_path=tmp_path / "c.yaml",
        results_dir=tmp_path / "r",
        existing_sim_check_timeout_s=0.0,
    )

    def _should_not_be_called(spec, timeout):
        raise AssertionError("wait_for_sim should be skipped when timeout is 0")

    monkeypatch.setattr(runner_module, "wait_for_sim", _should_not_be_called)
    assert _check_existing_sim_alive(spec) is None


def test_no_running_sim_failure_is_marked_retryable():
    result = _make_result(
        success=False,
        engine_return_code=None,
        failure_reason="No running sim detected (service ... missing).",
    )
    assert is_infrastructure_failure(result) is True


def test_engine_command_exports_zenoh_rmw(tmp_path: Path):
    """Regression: without RMW=rmw_zenoh_cpp, the in-container ``aic_engine``
    falls back to FastDDS, never sees ``/clock``, and dies after 10 s with
    ``Failed to find a valid clock`` -> ``aic_engine completed without
    producing scoring.yaml``."""
    from my_policy.gazebo_rl import runner as runner_module

    src = Path(runner_module.__file__).read_text()
    assert "RMW_IMPLEMENTATION=rmw_zenoh_cpp" in src
    assert "ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5" in src


def test_check_clock_in_container_passes_when_publisher_present(
    tmp_path: Path, monkeypatch
):
    spec = EpisodeSpec(
        config_path=tmp_path / "c.yaml",
        results_dir=tmp_path / "r",
    )

    class _R:
        stdout = "Publisher count: 1\n"
        stderr = ""

    monkeypatch.setattr(runner_module.subprocess, "run", lambda *a, **kw: _R())
    assert _check_clock_in_container(spec, use_root=False) is None


def test_check_clock_in_container_explains_when_no_publisher(
    tmp_path: Path, monkeypatch
):
    spec = EpisodeSpec(
        config_path=tmp_path / "c.yaml",
        results_dir=tmp_path / "r",
    )

    class _R:
        stdout = "Publisher count: 0\n"
        stderr = ""

    monkeypatch.setattr(runner_module.subprocess, "run", lambda *a, **kw: _R())
    reason = _check_clock_in_container(spec, use_root=False)
    assert reason is not None
    assert "/clock has no publisher" in reason
    assert "clean_sim.sh" in reason


def test_clock_missing_failure_is_marked_retryable():
    result = _make_result(
        success=False,
        engine_return_code=None,
        failure_reason=(
            "Sim is up but /clock has no publisher inside the container "
            "('Publisher count: 0')."
        ),
    )
    assert is_infrastructure_failure(result) is True


def test_wait_for_model_ready_returns_true_on_response(monkeypatch):
    """A ``ros2 service call`` that returns ``response: ... current_state: ...``
    means aic_model's executor is spinning, so we can launch the engine."""

    class _R:
        stdout = (
            "waiting for service to become available...\n"
            "requester: making request: lifecycle_msgs.srv.GetState_Request()\n"
            "\nresponse:\n"
            "lifecycle_msgs.srv.GetState_Response(current_state=lifecycle_msgs.msg."
            "State(id=1, label='unconfigured'))\n"
        )
        stderr = ""

    monkeypatch.setattr(runner_module.subprocess, "run", lambda *a, **kw: _R())
    assert wait_for_model_ready(timeout=2.0) is True


def test_wait_for_model_ready_times_out_when_no_response(monkeypatch):
    """If the service is advertised but the executor never answers (e.g.
    aic_model is still loading PyTorch), we must time out and report it."""

    class _R:
        stdout = "waiting for service to become available...\n"
        stderr = ""

    monkeypatch.setattr(runner_module.subprocess, "run", lambda *a, **kw: _R())
    monkeypatch.setattr(runner_module.time, "sleep", lambda _s: None)
    assert wait_for_model_ready(timeout=0.1) is False


def test_model_ready_timeout_failure_is_marked_retryable():
    result = _make_result(
        success=False,
        engine_return_code=None,
        failure_reason=(
            "aic_model did not answer /aic_model/get_state within 90 s. "
            "The policy module is probably still loading; bump "
            "EpisodeSpec.model_ready_timeout, or check "
            "outputs/<run>/logs/aic_model.log for an import error."
        ),
    )
    assert is_infrastructure_failure(result) is True


def test_resolve_init_checkpoint_explicit_takes_precedence(tmp_path: Path):
    """``--init-checkpoint`` always wins over the auto-resume default."""
    import argparse

    from train_gazebo_residual_act import resolve_init_checkpoint

    explicit = tmp_path / "from_other_run.pt"
    explicit.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    auto = out / "best_adapter.pt"
    auto.write_bytes(b"y")

    args = argparse.Namespace(
        init_checkpoint=explicit, no_auto_resume=False, output_dir=out
    )
    assert resolve_init_checkpoint(args) == explicit


def test_resolve_init_checkpoint_auto_resumes_when_best_present(tmp_path: Path):
    """Default behavior: re-running on the same --output-dir picks up
    best_adapter.pt automatically so improvements compound."""
    import argparse

    from train_gazebo_residual_act import resolve_init_checkpoint

    out = tmp_path / "out"
    out.mkdir()
    best = out / "best_adapter.pt"
    best.write_bytes(b"z")

    args = argparse.Namespace(
        init_checkpoint=None, no_auto_resume=False, output_dir=out
    )
    assert resolve_init_checkpoint(args) == best


def test_resolve_init_checkpoint_no_auto_resume_returns_none(tmp_path: Path):
    """``--no-auto-resume`` opts out of the implicit warm-start so the user
    can deliberately start from zero again."""
    import argparse

    from train_gazebo_residual_act import resolve_init_checkpoint

    out = tmp_path / "out"
    out.mkdir()
    (out / "best_adapter.pt").write_bytes(b"z")

    args = argparse.Namespace(
        init_checkpoint=None, no_auto_resume=True, output_dir=out
    )
    assert resolve_init_checkpoint(args) is None


def test_resolve_init_checkpoint_returns_none_on_fresh_dir(tmp_path: Path):
    import argparse

    from train_gazebo_residual_act import resolve_init_checkpoint

    out = tmp_path / "out"
    out.mkdir()

    args = argparse.Namespace(
        init_checkpoint=None, no_auto_resume=False, output_dir=out
    )
    assert resolve_init_checkpoint(args) is None


def test_resolve_init_checkpoint_missing_explicit_raises(tmp_path: Path):
    import argparse

    import pytest

    from train_gazebo_residual_act import resolve_init_checkpoint

    args = argparse.Namespace(
        init_checkpoint=tmp_path / "does_not_exist.pt",
        no_auto_resume=False,
        output_dir=tmp_path / "out",
    )
    with pytest.raises(FileNotFoundError):
        resolve_init_checkpoint(args)


def test_sample_iteration_configs_is_deterministic_for_a_given_iteration(
    tmp_path: Path,
):
    """All candidates in one iteration must see identical configs.

    The training loop only calls sample_iteration_configs once per
    iteration; this test guards the contract by confirming that two calls
    with the *same* RNG state produce the same draw, while advancing the
    RNG between iterations does change the draw.
    """
    import random

    from train_gazebo_residual_act import sample_iteration_configs

    configs = [tmp_path / f"config_{i:02d}.yaml" for i in range(8)]

    rng_a = random.Random(42)
    rng_b = random.Random(42)
    assert sample_iteration_configs(configs, 3, rng_a) == sample_iteration_configs(
        configs, 3, rng_b
    )

    # A second draw on the same RNG (i.e., next iteration) must produce a
    # fresh sample so the population sees diverse tasks across the run.
    second_draw = sample_iteration_configs(configs, 3, rng_a)
    rng_b_again = random.Random(42)
    first_draw = sample_iteration_configs(configs, 3, rng_b_again)
    assert second_draw != first_draw


def test_sample_iteration_configs_returns_all_when_request_exceeds_pool(
    tmp_path: Path,
):
    import random

    from train_gazebo_residual_act import sample_iteration_configs

    configs = [tmp_path / f"config_{i:02d}.yaml" for i in range(3)]
    chosen = sample_iteration_configs(configs, 10, random.Random(0))
    assert sorted(chosen) == sorted(configs)
