"""Gazebo evaluation episode runner.

Mirrors the manual workflow from ``docs/getting_started.md``: the user
keeps a long-lived ``distrobox enter -r aic_eval`` shell running
``/entrypoint.sh ... start_aic_engine:=false``, and this runner spawns
``aic_model`` (host pixi env) and ``aic_engine`` (in-container) once per
episode against that shared ROS graph.

For the why-each-preflight-exists context (RMW mismatch, model startup
race, sudo / distrobox interactions, ...), see
``docs/postmortems/gazebo_cem_training.md``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .scoring import RewardWeights, ScoreSummary, build_training_reward, load_score_summary

SIM_READINESS_SERVICE = "/aic_controller/change_target_mode"

RETRYABLE_ENGINE_LOG_MARKERS = (
    "GetState service call timed out",
    "Participant model is not ready",
    "Engine Stopped with Errors",
    "Failed to transition model node",
    "Failed to spawn cable",
)

RETRYABLE_FAILURE_REASON_MARKERS = (
    "aic_model exited during startup",
    "aic_model exited before episode finished",
    "Simulation did not become ready in time",
    "Episode timed out waiting for aic_engine",
    "aic_engine completed without producing scoring.yaml",
    "engine stopped with errors",
    "participant model is not ready",
    "no running sim detected",
    "/clock has no publisher inside the container",
    "aic_model did not answer /aic_model/get_state",
)


@dataclass(frozen=True)
class EpisodeSpec:
    config_path: Path
    results_dir: Path
    policy: str = "my_policy.ResidualACT"
    ground_truth: bool = False
    episode_timeout: int = 240
    sim_ready_timeout: int = 120
    model_startup_delay: float = 1.0
    # Hard cap for waiting on /aic_model/get_state to answer. ResidualACT
    # routinely takes 25-35 s of PyTorch import + checkpoint load before
    # its executor starts spinning, so 90 s is the safe default.
    model_ready_timeout: float = 90.0
    eval_container: str = "aic_eval"
    use_existing_sim: bool = True
    retries: int = 0
    retry_backoff_s: float = 3.0
    model_env: dict[str, str] = field(default_factory=dict)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    # ``None`` -> auto-detect (try without ``-r`` first; fall back to
    # ``-r``/sudo only if that fails). Force True/False to override.
    distrobox_use_root: bool | None = None
    # Set to 0 to skip the "is the sim actually publishing the readiness
    # service?" preflight when ``use_existing_sim=True``.
    existing_sim_check_timeout_s: float = 5.0


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    reward: float
    summary: ScoreSummary | None
    scoring_path: Path | None
    results_dir: Path
    engine_return_code: int | None
    model_return_code: int | None
    elapsed_s: float
    failure_reason: str | None = None
    attempts: int = 1
    retryable: bool = False


def _host_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a host-side env that always pins distrobox to Docker.

    pixi activation can shadow user-exported variables; pinning
    ``DBX_CONTAINER_MANAGER=docker`` here means subprocesses don't fall
    back to podman/sudo and prompt to create a fedora-toolbox container.
    """
    env = dict(os.environ)
    env.setdefault("DBX_CONTAINER_MANAGER", "docker")
    if extra:
        env.update(extra)
    return env


def _launch_process(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env if env is not None else _host_env(),
        preexec_fn=os.setsid,
    )


def _kill_process_group(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _read_log_tail(log_path: Path, max_lines: int = 12) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _diagnose_launch_failure(log_tail: str, container_name: str) -> str:
    if "sudo: a password is required" in log_tail or "sudo: a terminal is required" in log_tail:
        return (
            f"distrobox could not enter '{container_name}' and got stuck on a sudo prompt. "
            "Make sure the eval container already exists and export DBX_CONTAINER_MANAGER=docker."
        )
    if "no such container" in log_tail.lower() or "Create it now" in log_tail:
        return (
            f"distrobox could not find the '{container_name}' container. "
            "Create it first with the getting-started flow and export DBX_CONTAINER_MANAGER=docker."
        )
    if log_tail.strip():
        return f"Simulation launch failed early. Log tail:\n{log_tail}"
    return "Simulation process exited before the ROS graph became ready."


def _summary_all_trials_validation_failed(summary: ScoreSummary | None) -> bool:
    if summary is None or not summary.trials:
        return False
    return all(
        "validation failed" in trial.tier_1.message.lower()
        for trial in summary.trials.values()
    )


def is_infrastructure_failure(
    result: EpisodeResult,
    engine_log_tail: str = "",
) -> bool:
    """Classify whether an episode failed due to transient infra issues.

    Genuine policy rollouts that finish and produce a valid score are
    never flagged here; only sim/engine plumbing problems are.
    """
    if result.engine_return_code not in (None, 0):
        return True

    if result.summary is not None and result.summary.total == 0.0:
        if _summary_all_trials_validation_failed(result.summary):
            return True

    reason = (result.failure_reason or "").lower()
    if any(marker in reason for marker in RETRYABLE_FAILURE_REASON_MARKERS):
        return True

    if engine_log_tail and any(
        marker in engine_log_tail for marker in RETRYABLE_ENGINE_LOG_MARKERS
    ):
        return True

    return False


def _distrobox_command(
    spec: EpisodeSpec,
    command: str,
    *,
    use_root: bool | None = None,
) -> list[str]:
    """Wrap a shell command so it runs inside the eval distrobox container."""
    if use_root is None:
        use_root = bool(spec.distrobox_use_root)
    cmd = ["distrobox", "enter"]
    if use_root:
        cmd.append("-r")
    cmd.extend([spec.eval_container, "--", "bash", "-lc", command])
    return cmd


def _try_distrobox_enter(spec: EpisodeSpec, *, use_root: bool) -> str | None:
    """Attempt ``distrobox enter`` once; return None on success, else the error.

    ``stdin=DEVNULL`` makes a stuck ``sudo`` prompt fail immediately
    instead of hanging until the timeout expires.
    """
    try:
        result = subprocess.run(
            _distrobox_command(spec, "echo aic_eval_ok", use_root=use_root),
            capture_output=True,
            text=True,
            timeout=20,
            env=_host_env(),
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"distrobox preflight failed: {exc!r}"
    if result.returncode != 0 or "aic_eval_ok" not in result.stdout:
        tail = (result.stdout + result.stderr).strip().splitlines()[-6:]
        return _diagnose_launch_failure("\n".join(tail), spec.eval_container)
    return None


def _detect_distrobox_root(spec: EpisodeSpec) -> tuple[bool, str | None]:
    """Pick whether to pass ``-r`` to ``distrobox enter`` for this host.

    Try without ``-r`` first (works whenever the user is in the
    ``docker`` group, which is the recommended Docker post-install
    setup); only fall back to ``-r`` (which uses ``sudo docker``, so it
    can hang in non-TTY subprocesses) if the unprivileged variant fails.
    Honours ``spec.distrobox_use_root`` when forced.
    """
    if spec.distrobox_use_root is not None:
        forced = bool(spec.distrobox_use_root)
        return forced, _try_distrobox_enter(spec, use_root=forced)

    no_root_err = _try_distrobox_enter(spec, use_root=False)
    if no_root_err is None:
        return False, None

    root_err = _try_distrobox_enter(spec, use_root=True)
    if root_err is None:
        return True, None

    return False, (
        "distrobox enter failed both with and without -r.\n"
        f"without -r: {no_root_err}\n"
        f"with -r:    {root_err}"
    )


def _model_command(spec: EpisodeSpec) -> list[str]:
    """Launch ``aic_model`` on the host pixi env (matches docs Step 3)."""
    return [
        "pixi",
        "run",
        "ros2",
        "run",
        "aic_model",
        "aic_model",
        "--ros-args",
        "-p",
        "use_sim_time:=true",
        "-p",
        f"policy:={spec.policy}",
    ]


def wait_for_sim(timeout: float) -> bool:
    """Poll for the AIC controller service via the host pixi ROS CLI."""
    deadline = time.monotonic() + timeout
    poll_env = _host_env()
    cli_timeout = max(2.0, min(10.0, timeout))
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["pixi", "run", "ros2", "service", "list"],
                capture_output=True,
                text=True,
                timeout=cli_timeout,
                env=poll_env,
            )
            if SIM_READINESS_SERVICE in result.stdout:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(min(2.0, max(0.5, timeout / 4)))
    return False


def wait_for_model_ready(timeout: float) -> bool:
    """Poll until ``/aic_model/get_state`` actually answers a request.

    The lifecycle service is advertised as soon as ``create_service``
    runs, but only handled once the executor starts spinning. Heavy
    policies (e.g. ResidualACT loading PyTorch + a checkpoint in
    ``__init__``) routinely delay that by ~30 s, which is well past
    ``aic_engine``'s 5 s ``GetState`` timeout. Returns ``True`` once the
    service responds within ``timeout`` s.
    """
    deadline = time.monotonic() + timeout
    poll_env = _host_env()
    cli_timeout = 4.0
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                [
                    "pixi", "run", "ros2", "service", "call",
                    "/aic_model/get_state",
                    "lifecycle_msgs/srv/GetState",
                    "{}",
                ],
                capture_output=True,
                text=True,
                timeout=cli_timeout,
                env=poll_env,
            )
            if "response:" in result.stdout and "current_state" in result.stdout:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1.0)
    return False


def _check_clock_in_container(spec: EpisodeSpec, *, use_root: bool) -> str | None:
    """Verify ``/clock`` has a publisher *as seen from inside* the container.

    Catches the case where the host-side readiness service is up but
    ``gz_server`` has crashed: without this check the engine sits at
    "Waiting for clock" for 10 s and dies with the very confusing
    "no scoring.yaml" error.
    """
    cmd = _distrobox_command(
        spec,
        "source /ws_aic/install/setup.bash && "
        "export RMW_IMPLEMENTATION=rmw_zenoh_cpp && "
        "export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5 && "
        "ros2 topic info /clock 2>/dev/null | grep -E '^Publisher count:'",
        use_root=use_root,
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            env=_host_env(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return None  # don't block the run on a flaky preflight
    text = (result.stdout + result.stderr).strip()
    if "Publisher count: 0" in text or not text:
        return (
            "Sim is up but /clock has no publisher inside the container "
            f"({text!r}). gz_server probably crashed; run "
            "docker/my_policy/scripts/clean_sim.sh and re-launch "
            "/entrypoint.sh in your eval shell."
        )
    return None


def _check_existing_sim_alive(spec: EpisodeSpec) -> str | None:
    """Return ``None`` if a sim is publishing the readiness service, else why."""
    if spec.existing_sim_check_timeout_s <= 0:
        return None
    if not wait_for_sim(spec.existing_sim_check_timeout_s):
        return (
            f"No running sim detected (service '{SIM_READINESS_SERVICE}' is not "
            f"on the ROS graph after {spec.existing_sim_check_timeout_s:.0f} s). "
            "Either start it inside your eval shell with\n"
            f"    distrobox enter {spec.eval_container} -- /entrypoint.sh "
            "ground_truth:=false start_aic_engine:=false\n"
            "or run docker/my_policy/scripts/clean_sim.sh --check to inspect "
            "what's actually alive, then re-launch the entrypoint."
        )
    return None


def _make_failure(
    spec: EpisodeSpec,
    start_time: float,
    *,
    failure_reason: str,
    summary: ScoreSummary | None = None,
    scoring_path: Path | None = None,
    engine_return_code: int | None = None,
    model_return_code: int | None = None,
) -> EpisodeResult:
    """Construct a failed ``EpisodeResult`` with the standard penalty reward."""
    return EpisodeResult(
        success=False,
        reward=spec.reward_weights.invalid_episode_penalty,
        summary=summary,
        scoring_path=scoring_path,
        results_dir=spec.results_dir,
        engine_return_code=engine_return_code,
        model_return_code=model_return_code,
        elapsed_s=time.monotonic() - start_time,
        failure_reason=failure_reason,
    )


class GazeboEpisodeRunner:
    """Run one ACT policy episode through the Gazebo evaluation stack."""

    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        max_attempts = 1 + max(0, spec.retries)
        if max_attempts == 1:
            return self._run_once(spec)

        spec.results_dir.mkdir(parents=True, exist_ok=True)
        last_result: EpisodeResult | None = None
        for attempt in range(1, max_attempts + 1):
            attempt_dir = spec.results_dir / f"attempt_{attempt:02d}"
            attempt_spec = replace(spec, results_dir=attempt_dir)
            result = self._run_once(attempt_spec)

            engine_log_tail = _read_log_tail(
                attempt_dir / "logs" / "aic_engine.log", max_lines=40
            )
            infra_failure = is_infrastructure_failure(result, engine_log_tail)

            if not infra_failure:
                return replace(result, attempts=attempt, retryable=False)

            penalty_reason = (
                result.failure_reason
                or "Infrastructure failure detected from engine/model state."
            )
            result = replace(
                result,
                success=False,
                reward=spec.reward_weights.invalid_episode_penalty,
                failure_reason=f"[attempt {attempt}] {penalty_reason}",
                attempts=attempt,
                retryable=True,
            )
            last_result = result

            if attempt < max_attempts:
                time.sleep(spec.retry_backoff_s)

        return last_result  # type: ignore[return-value]

    def _run_once(self, spec: EpisodeSpec) -> EpisodeResult:
        spec.results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = spec.results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        sim_proc = None
        model_proc = None
        engine_proc = None
        start_time = time.monotonic()
        sim_log_path = logs_dir / "sim.log"

        manifest_path = spec.results_dir / "episode_spec.json"
        manifest_path.write_text(
            json.dumps(
                {
                    **asdict(spec),
                    "config_path": str(spec.config_path),
                    "results_dir": str(spec.results_dir),
                    "reward_weights": asdict(spec.reward_weights),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # Inherit the parent (host pixi) env so aic_model picks up the
        # ACT_* checkpoint env vars and the right RMW / Zenoh config.
        model_env = _host_env(spec.model_env)

        use_root, preflight_error = _detect_distrobox_root(spec)
        if preflight_error is not None:
            return _make_failure(
                spec, start_time,
                failure_reason=f"Distrobox preflight failed: {preflight_error}",
            )

        try:
            if not spec.use_existing_sim:
                sim_proc = _launch_process(
                    _distrobox_command(
                        spec,
                        "/entrypoint.sh "
                        "spawn_task_board:=false "
                        "spawn_cable:=false "
                        f"ground_truth:={'true' if spec.ground_truth else 'false'} "
                        "start_aic_engine:=false",
                        use_root=use_root,
                    ),
                    sim_log_path,
                )
                time.sleep(2.0)
                if sim_proc.poll() is not None:
                    return _make_failure(
                        spec, start_time,
                        failure_reason=_diagnose_launch_failure(
                            _read_log_tail(sim_log_path), spec.eval_container
                        ),
                    )

                if not wait_for_sim(spec.sim_ready_timeout):
                    return _make_failure(
                        spec, start_time,
                        failure_reason=(
                            "Simulation did not become ready in time. "
                            + _diagnose_launch_failure(
                                _read_log_tail(sim_log_path), spec.eval_container
                            )
                        ),
                    )
            else:
                sim_missing = _check_existing_sim_alive(spec)
                if sim_missing is not None:
                    return _make_failure(
                        spec, start_time, failure_reason=sim_missing
                    )
                clock_missing = _check_clock_in_container(spec, use_root=use_root)
                if clock_missing is not None:
                    return _make_failure(
                        spec, start_time, failure_reason=clock_missing
                    )

            model_proc = _launch_process(
                _model_command(spec),
                logs_dir / "aic_model.log",
                env=model_env,
            )
            time.sleep(spec.model_startup_delay)

            if model_proc.poll() is not None:
                return _make_failure(
                    spec, start_time,
                    failure_reason="aic_model exited during startup.",
                    model_return_code=model_proc.returncode,
                )

            if not wait_for_model_ready(spec.model_ready_timeout):
                if model_proc.poll() is not None:
                    return _make_failure(
                        spec, start_time,
                        failure_reason="aic_model exited during startup.",
                        model_return_code=model_proc.returncode,
                    )
                return _make_failure(
                    spec, start_time,
                    failure_reason=(
                        f"aic_model did not answer /aic_model/get_state "
                        f"within {spec.model_ready_timeout:.0f} s. The "
                        "policy module is probably still loading; bump "
                        "EpisodeSpec.model_ready_timeout, or check "
                        "outputs/<run>/logs/aic_model.log for an import "
                        "error."
                    ),
                )

            # /entrypoint.sh exports RMW + Zenoh router only inside its
            # own process tree, so a fresh distrobox shell falls back to
            # FastDDS and can't see /clock. See postmortem #2.
            engine_cmd = (
                "source /ws_aic/install/setup.bash && "
                "export RMW_IMPLEMENTATION=rmw_zenoh_cpp && "
                "export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5 && "
                f"export AIC_RESULTS_DIR='{spec.results_dir}' && "
                "ros2 run aic_engine aic_engine --ros-args "
                f"-p config_file_path:={spec.config_path.resolve()} "
                "-p use_sim_time:=true "
                f"-p ground_truth:={'true' if spec.ground_truth else 'false'}"
            )
            engine_proc = _launch_process(
                _distrobox_command(spec, engine_cmd, use_root=use_root),
                logs_dir / "aic_engine.log",
            )

            deadline = time.monotonic() + spec.episode_timeout
            while time.monotonic() < deadline:
                if engine_proc.poll() is not None:
                    break
                if model_proc.poll() is not None:
                    return _make_failure(
                        spec, start_time,
                        failure_reason="aic_model exited before episode finished.",
                        engine_return_code=engine_proc.poll(),
                        model_return_code=model_proc.returncode,
                    )
                time.sleep(1.0)

            if engine_proc.poll() is None:
                return _make_failure(
                    spec, start_time,
                    failure_reason="Episode timed out waiting for aic_engine.",
                    model_return_code=model_proc.poll(),
                )

            scoring_path = spec.results_dir / "scoring.yaml"
            if not scoring_path.exists():
                return _make_failure(
                    spec, start_time,
                    failure_reason="aic_engine completed without producing scoring.yaml.",
                    engine_return_code=engine_proc.returncode,
                    model_return_code=model_proc.poll(),
                )

            summary = load_score_summary(scoring_path)
            reward = build_training_reward(summary, spec.reward_weights)
            engine_rc = engine_proc.returncode
            engine_errored = engine_rc not in (0, None)
            scoring_looks_infra = (
                summary.total == 0.0
                and _summary_all_trials_validation_failed(summary)
            )
            if engine_errored or scoring_looks_infra:
                return _make_failure(
                    spec, start_time,
                    failure_reason=(
                        "aic_engine exited with a non-zero return code."
                        if engine_errored
                        else "Model validation failed for all trials."
                    ),
                    summary=summary,
                    scoring_path=scoring_path,
                    engine_return_code=engine_rc,
                    model_return_code=model_proc.poll(),
                )
            return EpisodeResult(
                success=True,
                reward=reward,
                summary=summary,
                scoring_path=scoring_path,
                results_dir=spec.results_dir,
                engine_return_code=engine_rc,
                model_return_code=model_proc.poll(),
                elapsed_s=time.monotonic() - start_time,
            )
        finally:
            _kill_process_group(engine_proc)
            _kill_process_group(model_proc)
            _kill_process_group(sim_proc)
