from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScoreCategory:
    score: float
    message: str = ""


@dataclass(frozen=True)
class TierScoreRecord:
    score: float
    message: str = ""
    categories: dict[str, ScoreCategory] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialScoreRecord:
    name: str
    tier_1: TierScoreRecord
    tier_2: TierScoreRecord
    tier_3: TierScoreRecord

    @property
    def total(self) -> float:
        return self.tier_1.score + self.tier_2.score + self.tier_3.score


@dataclass(frozen=True)
class ScoreSummary:
    total: float
    trials: dict[str, TrialScoreRecord]
    source_path: Path


@dataclass(frozen=True)
class RewardWeights:
    total: float = 1.0
    tier_3: float = 0.35
    duration: float = 0.10
    trajectory_smoothness: float = 0.05
    trajectory_efficiency: float = 0.05
    insertion_force: float = 0.25
    contacts: float = 0.25
    invalid_episode_penalty: float = -25.0


def _parse_tier_score(raw: dict | None) -> TierScoreRecord:
    raw = raw or {}
    categories: dict[str, ScoreCategory] = {}
    for name, category in (raw.get("categories") or {}).items():
        categories[name] = ScoreCategory(
            score=float(category.get("score", 0.0)),
            message=str(category.get("message", "")),
        )
    return TierScoreRecord(
        score=float(raw.get("score", 0.0)),
        message=str(raw.get("message", "")),
        categories=categories,
    )


def load_score_summary(path: str | Path) -> ScoreSummary:
    score_path = Path(path)
    with score_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    trials: dict[str, TrialScoreRecord] = {}
    for trial_name, trial_raw in raw.items():
        if trial_name == "total":
            continue
        trials[trial_name] = TrialScoreRecord(
            name=trial_name,
            tier_1=_parse_tier_score(trial_raw.get("tier_1")),
            tier_2=_parse_tier_score(trial_raw.get("tier_2")),
            tier_3=_parse_tier_score(trial_raw.get("tier_3")),
        )

    return ScoreSummary(
        total=float(raw.get("total", 0.0)),
        trials=trials,
        source_path=score_path,
    )


def _category_score(trial: TrialScoreRecord, name: str) -> float:
    return trial.tier_2.categories.get(name, ScoreCategory(0.0)).score


def build_training_reward(
    summary: ScoreSummary | None,
    weights: RewardWeights | None = None,
) -> float:
    """Build a training reward anchored on official scoring output.

    The official total score remains the dominant term. Smaller category-level
    contributions act as score-adjacent shaping so training can distinguish
    between episodes with similar totals but different failure modes.
    """
    if summary is None:
        weights = weights or RewardWeights()
        return weights.invalid_episode_penalty

    weights = weights or RewardWeights()
    reward = summary.total * weights.total

    for trial in summary.trials.values():
        reward += trial.tier_3.score * weights.tier_3
        reward += _category_score(trial, "duration") * weights.duration
        reward += _category_score(
            trial, "trajectory smoothness"
        ) * weights.trajectory_smoothness
        reward += _category_score(
            trial, "trajectory efficiency"
        ) * weights.trajectory_efficiency
        reward += _category_score(trial, "insertion force") * weights.insertion_force
        reward += _category_score(trial, "contacts") * weights.contacts

    return reward


def summarize_score_lines(summary: ScoreSummary) -> list[str]:
    lines = [f"total={summary.total:.3f}"]
    for trial in summary.trials.values():
        lines.append(
            f"{trial.name}: total={trial.total:.3f} "
            f"(tier1={trial.tier_1.score:.3f}, "
            f"tier2={trial.tier_2.score:.3f}, "
            f"tier3={trial.tier_3.score:.3f})"
        )
    return lines
