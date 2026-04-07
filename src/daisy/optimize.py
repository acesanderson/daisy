from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2

from daisy.types import OptimizationResult, PredictorArtifact

if TYPE_CHECKING:
    pass

logger = logging.getLogger("daisy")


def optimize(
    module: dspy.Module,
    trainset: list[dict],
    input_keys: list[str],
    metric: Callable,
    lm: str,
    api_base: str | None = None,
    api_key: str | None = None,
    auto: str = "light",
    num_candidates: int | None = None,
    num_trials: int | None = None,
    max_bootstrapped_demos: int | None = None,
    num_threads: int = 4,
) -> OptimizationResult:
    _validate(
        module, trainset, input_keys, metric, auto,
        num_candidates, num_trials, max_bootstrapped_demos, num_threads,
    )
    raise NotImplementedError("optimize() not yet implemented beyond validation")


def _validate(
    module, trainset, input_keys, metric, auto,
    num_candidates, num_trials, max_bootstrapped_demos, num_threads,
):
    if not trainset:
        raise ValueError("trainset must not be empty")
    if not list(module.named_predictors()):
        raise ValueError("module must have at least one named predictor")
    for ex in trainset:
        for key in input_keys:
            if key not in ex:
                raise ValueError(f"input_key '{key}' not found in all trainset examples")
    if not callable(metric):
        raise TypeError("metric must be callable")
    test_example = dspy.Example(**trainset[0]).with_inputs(*input_keys)
    try:
        test_result = metric(test_example, dspy.Prediction())
    except Exception as exc:
        raise TypeError(f"metric raised during test call: {exc}") from exc
    if (
        isinstance(test_result, bool)
        or not isinstance(test_result, float)
        or not math.isfinite(test_result)
    ):
        raise TypeError(
            f"metric must return a finite float, got {type(test_result).__name__}: {test_result!r}"
        )
