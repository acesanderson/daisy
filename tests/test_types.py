from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from daisy.types import OptimizationResult, PredictorArtifact


def test_predictor_artifact_is_frozen():
    artifact = PredictorArtifact(name="p", instructions="instr")
    with pytest.raises(FrozenInstanceError):
        artifact.name = "other"


def test_optimization_result_is_frozen():
    result = OptimizationResult(
        predictors=(),
        baseline_score=0.5,
        optimized_score=0.8,
        improved=True,
        duration_seconds=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        result.improved = False
