#!/usr/bin/env python3
"""Score-driven residual ACT optimization with Gazebo evaluation episodes.

A small cross-entropy-method (CEM) loop optimises the residual ACT
adapter; the ACT base policy stays frozen. See
``docs/postmortems/gazebo_cem_training.md`` for the working setup.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch

from my_policy.gazebo_rl.adapter import (
    ResidualACTAdapter,
    ResidualAdapterConfig,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from my_policy.gazebo_rl.runner import EpisodeSpec, GazeboEpisodeRunner
from my_policy.gazebo_rl.scoring import RewardWeights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("training/configs"),
        help="Directory of AIC engine YAML configs used as rollout episodes.",
    )
    parser.add_argument(
        "--config-glob",
        type=str,
        default="*.yaml",
        help="Glob pattern inside --config-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gazebo_score_rl"),
        help="Where to store candidate checkpoints, logs, and summaries.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of CEM updates.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=4,
        help="Number of candidate adapters per iteration.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=2,
        help="How many top candidates update the search distribution.",
    )
    parser.add_argument(
        "--episodes-per-candidate",
        type=int,
        default=1,
        help=(
            "How many config rollouts each candidate is evaluated on per "
            "iteration. The same configs are used for every candidate within "
            "an iteration (re-sampled each iteration), so candidate scores "
            "are directly comparable instead of being confounded by task "
            "difficulty. Each iteration therefore costs "
            "population_size * episodes_per_candidate rollouts."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for candidate sampling.",
    )
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=0.05,
        help="Initial Gaussian stddev over adapter parameter vector.",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=0.005,
        help="Lower bound on search stddev.",
    )
    parser.add_argument(
        "--eval-container",
        type=str,
        default="aic_eval",
        help="Distrobox evaluation container name.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="my_policy.ResidualACT",
        help="Policy class launched through aic_model.",
    )
    parser.add_argument(
        "--episode-timeout",
        type=int,
        default=240,
        help="Maximum wall-clock seconds for one rollout episode.",
    )
    parser.add_argument(
        "--act-model-repo",
        type=str,
        default="grkw/aic_act_policy",
        help="Hugging Face repo id for the frozen ACT backbone.",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to the residual adapter during rollouts.",
    )
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Enable ground-truth TF during training episodes.",
    )
    parser.add_argument(
        "--episode-retries",
        type=int,
        default=2,
        help="Retry transient infrastructure failures this many times per episode.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=5.0,
        help="Sleep between retry attempts in seconds.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start the CEM mean from this adapter checkpoint instead of "
            "zero-init. Typical use: pass the best_adapter.pt produced by an "
            "earlier run (e.g. outputs/gazebo_score_rl/best_adapter.pt) so each "
            "new training session continues improving from the previous best."
        ),
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help=(
            "Disable the default behavior of warm-starting from "
            "<output-dir>/best_adapter.pt when it already exists. Has no "
            "effect if --init-checkpoint is given."
        ),
    )
    return parser.parse_args()


def resolve_init_checkpoint(args: argparse.Namespace) -> Path | None:
    """Pick the adapter to warm-start from.

    Precedence:
      1. ``--init-checkpoint`` if explicitly set (must exist).
      2. ``<output-dir>/best_adapter.pt`` if it exists and ``--no-auto-resume``
         was not passed (so each subsequent run on the same output dir
         continues improving on the previous best).
      3. ``None`` -> zero-init.
    """
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(
                f"--init-checkpoint {args.init_checkpoint} does not exist"
            )
        return args.init_checkpoint
    if args.no_auto_resume:
        return None
    auto = args.output_dir / "best_adapter.pt"
    return auto if auto.exists() else None


def discover_configs(config_dir: Path, pattern: str) -> list[Path]:
    configs = sorted(config_dir.glob(pattern))
    if not configs:
        raise FileNotFoundError(
            f"No configs matching '{pattern}' were found in {config_dir.resolve()}"
        )
    return configs


def sample_iteration_configs(
    configs: list[Path],
    episodes_per_iteration: int,
    rng: random.Random,
) -> list[Path]:
    """Pick the configs every candidate in this iteration will be scored on.

    Sampling once per iteration (not per candidate) makes candidate
    rewards directly comparable on identical tasks. The draw still
    rotates between iterations so the population sees a broad task mix.
    """
    if episodes_per_iteration >= len(configs):
        chosen = list(configs)
        rng.shuffle(chosen)
        return chosen[:episodes_per_iteration]
    return rng.sample(configs, episodes_per_iteration)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    configs = discover_configs(args.config_dir, args.config_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_adapter = ResidualACTAdapter(ResidualAdapterConfig())
    init_checkpoint = resolve_init_checkpoint(args)
    prior_metadata: dict = {}
    if init_checkpoint is not None:
        loaded_adapter, prior_metadata = load_adapter_checkpoint(init_checkpoint)
        base_adapter.load_state_dict(loaded_adapter.state_dict())
        log.info("Warm-starting CEM mean from %s", init_checkpoint)
    else:
        log.info("No init checkpoint; starting from zero-init adapter.")
    mean = base_adapter.parameter_vector()
    sigma = torch.full_like(mean, args.init_sigma)

    runner = GazeboEpisodeRunner()
    best_checkpoint = args.output_dir / "best_adapter.pt"
    # Carry the prior best_reward forward as a floor so an unlucky early
    # sample can't overwrite a known-good warm-start checkpoint.
    best_reward = float(prior_metadata.get("best_reward", float("-inf")))
    if best_reward > float("-inf"):
        log.info("Carrying over prior best_reward=%.3f as floor.", best_reward)

    for iteration in range(args.iterations):
        iter_dir = args.output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        candidate_records: list[dict] = []

        # Same config(s) for every candidate this iteration; recorded in
        # iter_NNN/summary.json for audit.
        config_batch = sample_iteration_configs(
            configs,
            args.episodes_per_candidate,
            rng,
        )

        for candidate_idx in range(args.population_size):
            candidate_dir = iter_dir / f"candidate_{candidate_idx:02d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            sample = mean + torch.randn_like(mean) * sigma
            adapter = ResidualACTAdapter(ResidualAdapterConfig())
            adapter.load_parameter_vector(sample)
            checkpoint_path = candidate_dir / "adapter.pt"
            save_adapter_checkpoint(
                checkpoint_path,
                adapter,
                metadata={
                    "iteration": iteration,
                    "candidate_index": candidate_idx,
                },
            )

            rewards: list[float] = []
            episode_records: list[dict] = []

            for episode_idx, config_path in enumerate(config_batch):
                episode_dir = candidate_dir / f"episode_{episode_idx:02d}"
                result = runner.run_episode(
                    EpisodeSpec(
                        config_path=config_path,
                        results_dir=episode_dir,
                        policy=args.policy,
                        ground_truth=args.ground_truth,
                        episode_timeout=args.episode_timeout,
                        eval_container=args.eval_container,
                        retries=args.episode_retries,
                        retry_backoff_s=args.retry_backoff_s,
                        model_env={
                            "ACT_MODEL_REPO": args.act_model_repo,
                            "ACT_ADAPTER_PATH": str(checkpoint_path.resolve()),
                            "ACT_ADAPTER_SCALE": str(args.adapter_scale),
                        },
                        reward_weights=RewardWeights(),
                    )
                )
                rewards.append(result.reward)
                episode_records.append(
                    {
                        "config": str(config_path),
                        "reward": result.reward,
                        "success": result.success,
                        "attempts": result.attempts,
                        "retryable": result.retryable,
                        "failure_reason": result.failure_reason,
                        "scoring_path": str(result.scoring_path)
                        if result.scoring_path
                        else None,
                        "total_score": result.summary.total if result.summary else None,
                    }
                )
                log.info(
                    "iter=%d cand=%d ep=%d reward=%.3f success=%s attempts=%d",
                    iteration,
                    candidate_idx,
                    episode_idx,
                    result.reward,
                    result.success,
                    result.attempts,
                )

            average_reward = sum(rewards) / len(rewards)
            candidate_record = {
                "candidate_index": candidate_idx,
                "average_reward": average_reward,
                "checkpoint_path": str(checkpoint_path),
                "episodes": episode_records,
                "parameter_vector": sample.tolist(),
            }
            candidate_records.append(candidate_record)

            if average_reward > best_reward:
                best_reward = average_reward
                save_adapter_checkpoint(
                    best_checkpoint,
                    adapter,
                    metadata={
                        "best_reward": best_reward,
                        "iteration": iteration,
                        "candidate_index": candidate_idx,
                    },
                )

        candidate_records.sort(key=lambda item: item["average_reward"], reverse=True)
        elites = candidate_records[: args.elite_count]
        elite_vectors = torch.tensor(
            [elite["parameter_vector"] for elite in elites], dtype=mean.dtype
        )
        mean = elite_vectors.mean(dim=0)
        sigma = elite_vectors.std(dim=0, unbiased=False).clamp(min=args.min_sigma)

        summary = {
            "iteration": iteration,
            "best_reward_so_far": best_reward,
            "elite_count": args.elite_count,
            "population_size": args.population_size,
            "iteration_configs": [str(p) for p in config_batch],
            "candidates": candidate_records,
            "mean_vector": mean.tolist(),
            "sigma_vector": sigma.tolist(),
        }
        (iter_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
