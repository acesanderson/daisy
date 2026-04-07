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
