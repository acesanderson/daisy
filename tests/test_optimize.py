from __future__ import annotations

from unittest.mock import patch

import pytest

from daisy.optimize import optimize


def test_empty_trainset_raises_before_lm_call(simple_module, good_metric):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="trainset must not be empty"):
            optimize(
                module=simple_module,
                trainset=[],
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
            )
        mock_lm.assert_not_called()


def test_empty_module_raises_before_lm_call(empty_module, trainset, good_metric):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="at least one named predictor"):
            optimize(
                module=empty_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
            )
        mock_lm.assert_not_called()


def test_non_callable_metric_raises_before_lm_call(simple_module, trainset):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(TypeError, match="metric must be callable"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric="not_a_function",
                lm="openai/gpt-4o-mini",
            )
        mock_lm.assert_not_called()


def test_missing_input_key_raises_before_lm_call(simple_module, good_metric):
    trainset = [{"question": "Q1", "answer": "A1"}, {"answer": "A2"}]  # second example missing "question"
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="input_key 'question'"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
            )
        mock_lm.assert_not_called()


def test_metric_that_raises_on_call_raises_type_error(simple_module, trainset):
    def raising_metric(example, prediction):
        raise RuntimeError("deliberate error")

    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(TypeError, match="metric raised during test call"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=raising_metric,
                lm="openai/gpt-4o-mini",
            )
        mock_lm.assert_not_called()


import math as _math

@pytest.mark.parametrize("bad_return,label", [
    ("not a float", "str"),
    (None, "NoneType"),
    (float("nan"), "nan"),
    (float("inf"), "inf"),
    (True, "bool"),
    (1, "int"),
])
def test_metric_bad_return_raises_type_error(simple_module, trainset, bad_return, label):
    def bad_metric(example, prediction):
        return bad_return

    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(TypeError, match="metric must return a finite float"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=bad_metric,
                lm="openai/gpt-4o-mini",
            )
        mock_lm.assert_not_called()


def test_num_threads_less_than_one_raises(simple_module, trainset, good_metric):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="num_threads must be >= 1"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
                num_threads=0,
            )
        mock_lm.assert_not_called()


def test_num_candidates_less_than_one_raises(simple_module, trainset, good_metric):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="num_candidates must be >= 1"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
                num_candidates=0,
            )
        mock_lm.assert_not_called()


def test_num_trials_less_than_one_raises(simple_module, trainset, good_metric):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="num_trials must be >= 1"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
                num_trials=0,
            )
        mock_lm.assert_not_called()


def test_max_bootstrapped_demos_negative_raises(simple_module, trainset, good_metric):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="max_bootstrapped_demos must be >= 0"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
                max_bootstrapped_demos=-1,
            )
        mock_lm.assert_not_called()


def test_max_bootstrapped_demos_zero_is_valid(simple_module, trainset, good_metric, mock_dspy):
    # zero is valid (>= 0); verify it does not raise
    result = optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
        max_bootstrapped_demos=0,
    )
    from daisy.types import OptimizationResult
    assert isinstance(result, OptimizationResult)


def test_invalid_auto_raises(simple_module, trainset, good_metric):
    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="auto must be"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
                auto="turbo",
            )
        mock_lm.assert_not_called()


def test_per_predictor_lm_raises(simple_module, trainset, good_metric):
    from unittest.mock import MagicMock
    simple_module.generate.lm = MagicMock()

    with patch("daisy.optimize.dspy.LM") as mock_lm:
        with pytest.raises(ValueError, match="Per-predictor LMs are not supported"):
            optimize(
                module=simple_module,
                trainset=trainset,
                input_keys=["question"],
                metric=good_metric,
                lm="openai/gpt-4o-mini",
            )
        mock_lm.assert_not_called()


def test_caller_module_not_mutated(simple_module, trainset, good_metric, mock_dspy):
    original_instructions = simple_module.generate.signature.instructions

    optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )

    assert simple_module.generate.signature.instructions == original_instructions


# Task 16 (AC6)
def test_instructions_are_unmodified_string(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["predictor"].signature.instructions = "You are a precise answerer."
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert result.predictors[0].instructions == "You are a precise answerer."


# Task 17 (AC7)
def test_demo_values_cast_to_str(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["predictor"].demos = [{"question": "Q?", "scores": [1, 2, 3]}]
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    demos = result.predictors[0].demos
    assert len(demos) == 1
    assert demos[0]["scores"] == "[1, 2, 3]"
    assert isinstance(demos[0]["scores"], str)


# Task 18 (AC5)
def test_one_artifact_per_predictor_in_order(simple_module, trainset, good_metric, mock_dspy):
    from unittest.mock import MagicMock
    mock_pred_a = MagicMock()
    mock_pred_a.signature.instructions = "Instr A"
    mock_pred_a.demos = []
    mock_pred_a.lm = None
    mock_pred_b = MagicMock()
    mock_pred_b.signature.instructions = "Instr B"
    mock_pred_b.demos = []
    mock_pred_b.lm = None
    mock_dspy["compiled"].named_predictors.return_value = [
        ("step_one", mock_pred_a),
        ("step_two", mock_pred_b),
    ]
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert len(result.predictors) == 2
    assert result.predictors[0].name == "step_one"
    assert result.predictors[0].instructions == "Instr A"
    assert result.predictors[1].name == "step_two"
    assert result.predictors[1].instructions == "Instr B"


# Task 19 (AC2)
def test_improved_true_when_optimized_score_exceeds_baseline(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = [0.5, 0.8]
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert result.improved is True
    assert result.baseline_score == 0.5
    assert result.optimized_score == 0.8


def test_improved_true_when_scores_equal(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = [0.7, 0.7]
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert result.improved is True


def test_improved_false_when_optimized_score_below_baseline(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = [0.8, 0.5]
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert result.improved is False


# Task 20 (AC3)
def test_returns_baseline_artifacts_when_no_improvement(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = [0.8, 0.5]
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert result.improved is False
    assert result.predictors[0].instructions != "Test instructions"


# Task 21 (AC4)
def test_returns_optimized_artifacts_when_improved(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = [0.5, 0.8]
    mock_dspy["predictor"].signature.instructions = "Optimized by MIPROv2"
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert result.improved is True
    assert result.predictors[0].instructions == "Optimized by MIPROv2"


# Task 22 (AC1)
from daisy.types import OptimizationResult


def test_always_returns_optimization_result(simple_module, trainset, good_metric, mock_dspy):
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert isinstance(result, OptimizationResult)


# Task 23 (AC21)
import math


def test_duration_seconds_is_positive_finite_float(simple_module, trainset, good_metric, mock_dspy):
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert isinstance(result.duration_seconds, float)
    assert math.isfinite(result.duration_seconds)
    assert result.duration_seconds > 0


# Task 24 (AC19)
def test_mipro_exception_propagates_unchanged(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["mipro_cls"].return_value.compile.side_effect = RuntimeError("network failure")
    with pytest.raises(RuntimeError, match="network failure"):
        optimize(
            module=simple_module, trainset=trainset, input_keys=["question"],
            metric=good_metric, lm="openai/gpt-4o-mini",
        )


def test_evaluate_exception_propagates_unchanged(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = ConnectionError("rate limited")
    with pytest.raises(ConnectionError, match="rate limited"):
        optimize(
            module=simple_module, trainset=trainset, input_keys=["question"],
            metric=good_metric, lm="openai/gpt-4o-mini",
        )


# Task 25 (Logging)
import logging


def test_logs_start_baseline_and_completion(simple_module, trainset, good_metric, mock_dspy, caplog):
    with caplog.at_level(logging.INFO, logger="daisy"):
        optimize(
            module=simple_module, trainset=trainset, input_keys=["question"],
            metric=good_metric, lm="openai/gpt-4o-mini",
        )
    messages = [r.message for r in caplog.records]
    assert any("Starting optimization" in m for m in messages)
    assert any("Baseline score" in m for m in messages)
    assert any("Optimization complete" in m for m in messages)
