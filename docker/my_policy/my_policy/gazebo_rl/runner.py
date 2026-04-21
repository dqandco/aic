from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .scoring import RewardWeights, ScoreSummary, build_training_reward, load_score_summary

SIM_READINESS_SERVICE = "/aic_controller/change_target_mode"
DEFAULT_ROS_ENV = {
    "RMW_IMPLEMENTATION": "rmw_zenoh_cpp",
    "ZENOH_ROUTER_CHECK_ATTEMPTS": "-1",
}


@dataclass(frozen=True)
class EpisodeSpec:
    config_path: Path
    results_dir: Path
    policy: str = "my_policy.ResidualACT"
    ground_truth: bool = False
    episode_timeout: int = 240
    sim_ready_timeout: int = 120
    model_startup_delay: float = 5.0
    eval_container: str = "aic_eval"
    inside_eval_container: bool = False
    use_existing_sim: bool = False
    model_env: dict[str, str] = field(default_factory=dict)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


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
        env=env,
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


def _merged_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(DEFAULT_ROS_ENV)
    if extra_env:
        env.update(extra_env)
    return env


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


def _ros2_service_list_cmd(spec: EpisodeSpec) -> list[str]:
    if spec.inside_eval_container:
        return ["bash", "-lc", "source /ws_aic/install/setup.bash && ros2 service list"]
    return ["pixi", "run", "ros2", "service", "list"]


def _wrap_eval_command(spec: EpisodeSpec, command: str) -> list[str]:
    if spec.inside_eval_container:
        return ["bash", "-lc", command]
    return [
        "distrobox",
        "enter",
        "-r",
        spec.eval_container,
        "--",
        "bash",
        "-lc",
        command,
    ]


def wait_for_sim(
    spec: EpisodeSpec,
    timeout: int,
    env: dict[str, str] | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    run_env = _merged_env(env)
    while time.monotonic() < deadline:
        result = subprocess.run(
            _ros2_service_list_cmd(spec),
            capture_output=True,
            text=True,
            timeout=10,
            env=run_env,
        )
        if SIM_READINESS_SERVICE in result.stdout:
            return True
        time.sleep(3)
    return False


class GazeboEpisodeRunner:
    """Run one ACT policy episode through the Gazebo evaluation stack."""

    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        spec.results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = spec.results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        model_env = _merged_env(spec.model_env)
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

        try:
            if not spec.use_existing_sim:
                sim_proc = _launch_process(
                    _wrap_eval_command(
                        spec,
                        "/entrypoint.sh "
                        "spawn_task_board:=false "
                        "spawn_cable:=false "
                        f"ground_truth:={'true' if spec.ground_truth else 'false'} "
                        "start_aic_engine:=false",
                    ),
                    sim_log_path,
                    env=model_env,
                )
                time.sleep(2.0)
                if sim_proc.poll() is not None:
                    return EpisodeResult(
                        success=False,
                        reward=spec.reward_weights.invalid_episode_penalty,
                        summary=None,
                        scoring_path=None,
                        results_dir=spec.results_dir,
                        engine_return_code=None,
                        model_return_code=None,
                        elapsed_s=time.monotonic() - start_time,
                        failure_reason=_diagnose_launch_failure(
                            _read_log_tail(sim_log_path), spec.eval_container
                        ),
                    )

            sim_ready_deadline = time.monotonic() + spec.sim_ready_timeout
            while time.monotonic() < sim_ready_deadline:
                if wait_for_sim(spec, 3, env=model_env):
                    break
                if sim_proc is not None and sim_proc.poll() is not None:
                    return EpisodeResult(
                        success=False,
                        reward=spec.reward_weights.invalid_episode_penalty,
                        summary=None,
                        scoring_path=None,
                        results_dir=spec.results_dir,
                        engine_return_code=None,
                        model_return_code=None,
                        elapsed_s=time.monotonic() - start_time,
                        failure_reason=_diagnose_launch_failure(
                            _read_log_tail(sim_log_path), spec.eval_container
                        ),
                    )
            else:
                return EpisodeResult(
                    success=False,
                    reward=spec.reward_weights.invalid_episode_penalty,
                    summary=None,
                    scoring_path=None,
                    results_dir=spec.results_dir,
                    engine_return_code=None,
                    model_return_code=None,
                    elapsed_s=time.monotonic() - start_time,
                    failure_reason=(
                        "Simulation did not become ready in time. "
                        + _diagnose_launch_failure(
                            _read_log_tail(sim_log_path), spec.eval_container
                        )
                    ),
                )

            model_proc = _launch_process(
                [
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
                ],
                logs_dir / "aic_model.log",
                env=model_env,
            )
            time.sleep(spec.model_startup_delay)

            if model_proc.poll() is not None:
                return EpisodeResult(
                    success=False,
                    reward=spec.reward_weights.invalid_episode_penalty,
                    summary=None,
                    scoring_path=None,
                    results_dir=spec.results_dir,
                    engine_return_code=None,
                    model_return_code=model_proc.returncode,
                    elapsed_s=time.monotonic() - start_time,
                    failure_reason="aic_model exited during startup.",
                )

            engine_cmd = (
                "source /ws_aic/install/setup.bash && "
                f"export AIC_RESULTS_DIR='{spec.results_dir}' && "
                "ros2 run aic_engine aic_engine --ros-args "
                f"-p config_file_path:={spec.config_path.resolve()} "
                "-p use_sim_time:=true "
                f"-p ground_truth:={'true' if spec.ground_truth else 'false'}"
            )
            engine_proc = _launch_process(
                _wrap_eval_command(spec, engine_cmd),
                logs_dir / "aic_engine.log",
                env=model_env,
            )

            deadline = time.monotonic() + spec.episode_timeout
            while time.monotonic() < deadline:
                if engine_proc.poll() is not None:
                    break
                if model_proc.poll() is not None:
                    return EpisodeResult(
                        success=False,
                        reward=spec.reward_weights.invalid_episode_penalty,
                        summary=None,
                        scoring_path=None,
                        results_dir=spec.results_dir,
                        engine_return_code=engine_proc.poll(),
                        model_return_code=model_proc.returncode,
                        elapsed_s=time.monotonic() - start_time,
                        failure_reason="aic_model exited before episode finished.",
                    )
                time.sleep(1.0)

            if engine_proc.poll() is None:
                return EpisodeResult(
                    success=False,
                    reward=spec.reward_weights.invalid_episode_penalty,
                    summary=None,
                    scoring_path=None,
                    results_dir=spec.results_dir,
                    engine_return_code=None,
                    model_return_code=model_proc.poll(),
                    elapsed_s=time.monotonic() - start_time,
                    failure_reason="Episode timed out waiting for aic_engine.",
                )

            scoring_path = spec.results_dir / "scoring.yaml"
            if not scoring_path.exists():
                return EpisodeResult(
                    success=False,
                    reward=spec.reward_weights.invalid_episode_penalty,
                    summary=None,
                    scoring_path=None,
                    results_dir=spec.results_dir,
                    engine_return_code=engine_proc.returncode,
                    model_return_code=model_proc.poll(),
                    elapsed_s=time.monotonic() - start_time,
                    failure_reason="aic_engine completed without producing scoring.yaml.",
                )

            summary = load_score_summary(scoring_path)
            reward = build_training_reward(summary, spec.reward_weights)
            return EpisodeResult(
                success=True,
                reward=reward,
                summary=summary,
                scoring_path=scoring_path,
                results_dir=spec.results_dir,
                engine_return_code=engine_proc.returncode,
                model_return_code=model_proc.poll(),
                elapsed_s=time.monotonic() - start_time,
            )
        finally:
            _kill_process_group(engine_proc)
            _kill_process_group(model_proc)
            _kill_process_group(sim_proc)
