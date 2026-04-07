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

    start = time.monotonic()

    dspy_trainset = [dspy.Example(**ex).with_inputs(*input_keys) for ex in trainset]

    working_module = copy.deepcopy(module)

    lm_kwargs: dict = {"cache": False}
    if api_base is not None:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key
    lm_instance = dspy.LM(lm, **lm_kwargs)
    dspy.configure(lm=lm_instance)

    logger.info(
        "Starting optimization: module=%s trainset_size=%d lm=%s auto=%s num_threads=%d",
        type(module).__name__, len(trainset), lm, auto, num_threads,
    )

    evaluator = Evaluate(
        devset=dspy_trainset, metric=metric, num_threads=num_threads, display_progress=False,
    )

    baseline_score = float(evaluator(copy.deepcopy(working_module)))
    baseline_artifacts = _extract_artifacts(working_module)

    logger.info("Baseline score: %f", baseline_score)

    mipro_kwargs: dict = {"metric": metric, "auto": auto, "num_threads": num_threads}
    if num_candidates is not None:
        mipro_kwargs["num_candidates"] = num_candidates
    if max_bootstrapped_demos is not None:
        mipro_kwargs["max_bootstrapped_demos"] = max_bootstrapped_demos

    compile_kwargs: dict = {"trainset": dspy_trainset}
    if num_trials is not None:
        compile_kwargs["num_trials"] = num_trials

    optimizer = MIPROv2(**mipro_kwargs)
    optimized_module = optimizer.compile(working_module, **compile_kwargs)

    optimized_score = float(evaluator(optimized_module))
    duration = time.monotonic() - start

    improved = optimized_score >= baseline_score
    artifacts = _extract_artifacts(optimized_module) if improved else baseline_artifacts

    logger.info(
        "Optimization complete: optimized_score=%f improved=%s duration_seconds=%.2f",
        optimized_score, improved, duration,
    )

    return OptimizationResult(
        predictors=artifacts,
        baseline_score=baseline_score,
        optimized_score=optimized_score,
        improved=improved,
        duration_seconds=duration,
    )


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
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    if num_candidates is not None and num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
    if num_trials is not None and num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")
    if max_bootstrapped_demos is not None and max_bootstrapped_demos < 0:
        raise ValueError(f"max_bootstrapped_demos must be >= 0, got {max_bootstrapped_demos}")
    if auto not in {"light", "medium", "heavy"}:
        raise ValueError(f"auto must be 'light', 'medium', or 'heavy', got {auto!r}")
    for _name, predictor in module.named_predictors():
        if getattr(predictor, "lm", None) is not None:
            raise ValueError(
                "Per-predictor LMs are not supported. Set a single LM via the  parameter."
            )


def _extract_artifacts(module: dspy.Module) -> tuple[PredictorArtifact, ...]:
    return tuple(
        PredictorArtifact(
            name=name,
            instructions=predictor.signature.instructions,
            demos=tuple(
                {k: str(v) for k, v in demo.items()}
                for demo in (predictor.demos or [])
            ),
        )
        for name, predictor in module.named_predictors()
    )
