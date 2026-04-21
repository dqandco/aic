#!/usr/bin/env python3
"""Run one Gazebo evaluation episode and convert official scores into reward."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from my_policy.gazebo_rl.runner import EpisodeSpec, GazeboEpisodeRunner
from my_policy.gazebo_rl.scoring import RewardWeights, summarize_score_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="AIC engine config YAML.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory for scoring artifacts and logs.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="my_policy.ResidualACT",
        help="Policy class for aic_model.",
    )
    parser.add_argument(
        "--eval-container",
        type=str,
        default="aic_eval",
        help="Distrobox container name for the evaluation environment.",
    )
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Expose ground-truth TF to the policy during the rollout.",
    )
    parser.add_argument(
        "--episode-timeout",
        type=int,
        default=240,
        help="Maximum wall-clock seconds for one episode.",
    )
    parser.add_argument(
        "--inside-eval-container",
        action="store_true",
        help="Run eval-side commands directly in the current shell instead of via distrobox enter.",
    )
    parser.add_argument(
        "--use-existing-sim",
        action="store_true",
        help="Assume the Gazebo eval environment is already running and skip launching /entrypoint.sh.",
    )
    parser.add_argument(
        "--act-model-repo",
        type=str,
        default=os.environ.get("ACT_MODEL_REPO", "grkw/aic_act_policy"),
        help="Hugging Face repo for the base ACT checkpoint.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional residual adapter checkpoint.",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=float(os.environ.get("ACT_ADAPTER_SCALE", "1.0")),
        help="Global multiplier applied to the residual adapter output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_env = {
        "ACT_MODEL_REPO": args.act_model_repo,
        "ACT_ADAPTER_SCALE": str(args.adapter_scale),
    }
    if args.adapter_path is not None:
        model_env["ACT_ADAPTER_PATH"] = str(args.adapter_path.resolve())

    spec = EpisodeSpec(
        config_path=args.config.resolve(),
        results_dir=args.results_dir.resolve(),
        policy=args.policy,
        ground_truth=args.ground_truth,
        episode_timeout=args.episode_timeout,
        eval_container=args.eval_container,
        inside_eval_container=args.inside_eval_container,
        use_existing_sim=args.use_existing_sim,
        model_env=model_env,
        reward_weights=RewardWeights(),
    )

    runner = GazeboEpisodeRunner()
    result = runner.run_episode(spec)

    summary_json = {
        "success": result.success,
        "reward": result.reward,
        "elapsed_s": result.elapsed_s,
        "failure_reason": result.failure_reason,
        "engine_return_code": result.engine_return_code,
        "model_return_code": result.model_return_code,
        "scoring_path": str(result.scoring_path) if result.scoring_path else None,
        "results_dir": str(result.results_dir),
        "total_score": result.summary.total if result.summary else None,
        "score_lines": summarize_score_lines(result.summary) if result.summary else [],
    }
    print(json.dumps(summary_json, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
