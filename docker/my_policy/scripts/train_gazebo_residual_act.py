#!/usr/bin/env python3
"""Score-driven residual ACT optimization with Gazebo evaluation episodes.

This trainer uses a small cross-entropy-method loop because full PPO-style
fine-tuning is awkward in the expensive Gazebo evaluation stack. The ACT base
policy stays frozen; only a tiny residual action head is optimized.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from my_policy.gazebo_rl.adapter import (
    ResidualACTAdapter,
    ResidualAdapterConfig,
    save_adapter_checkpoint,
)
from my_policy.gazebo_rl.runner import EpisodeSpec, GazeboEpisodeRunner
from my_policy.gazebo_rl.scoring import RewardWeights


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
        help="How many config rollouts to average for each sampled candidate.",
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
    return parser.parse_args()


def discover_configs(config_dir: Path, pattern: str) -> list[Path]:
    configs = sorted(config_dir.glob(pattern))
    if not configs:
        raise FileNotFoundError(
            f"No configs matching '{pattern}' were found in {config_dir.resolve()}"
        )
    return configs


def sample_episode_configs(
    population_index: int,
    configs: list[Path],
    episodes_per_candidate: int,
    rng: random.Random,
) -> list[Path]:
    if episodes_per_candidate >= len(configs):
        chosen = list(configs)
        rng.shuffle(chosen)
        return chosen[:episodes_per_candidate]
    return rng.sample(configs, episodes_per_candidate)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    configs = discover_configs(args.config_dir, args.config_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_adapter = ResidualACTAdapter(ResidualAdapterConfig())
    mean = base_adapter.parameter_vector()
    sigma = torch.full_like(mean, args.init_sigma)

    runner = GazeboEpisodeRunner()
    best_reward = float("-inf")
    best_checkpoint = args.output_dir / "best_adapter.pt"

    for iteration in range(args.iterations):
        iter_dir = args.output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        candidate_records: list[dict] = []

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

            config_batch = sample_episode_configs(
                candidate_idx,
                configs,
                args.episodes_per_candidate,
                rng,
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
                        "failure_reason": result.failure_reason,
                        "scoring_path": str(result.scoring_path)
                        if result.scoring_path
                        else None,
                        "total_score": result.summary.total if result.summary else None,
                    }
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
