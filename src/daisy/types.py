from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PredictorArtifact:
    name: str
    instructions: str
    demos: tuple[dict[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OptimizationResult:
    predictors: tuple[PredictorArtifact, ...]
    baseline_score: float
    optimized_score: float
    improved: bool
    duration_seconds: float


__all__ = [
    "PredictorArtifact",
    "OptimizationResult",
]
