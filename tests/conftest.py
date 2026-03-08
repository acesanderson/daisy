from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy
import pytest


class _SimpleModule(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.generate(question=question)


class _EmptyModule(dspy.Module):
    """A module with no predictors — used to test AC 12."""
    def forward(self, question):
        return question


@pytest.fixture
def simple_module():
    return _SimpleModule()


@pytest.fixture
def empty_module():
    return _EmptyModule()


@pytest.fixture
def trainset():
    return [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


@pytest.fixture
def good_metric():
    """Always returns 1.0 without accessing prediction fields — survives test call."""
    def _metric(example, prediction):
        return 1.0
    return _metric


@pytest.fixture
def mock_dspy():
    """
    Patches all DSPy infrastructure for a full optimize() run.
    Yields a dict of mock handles for per-test customisation.
    Default: baseline_score=0.5, optimized_score=0.8, one predictor named 'generate'.
    """
    mock_pred = MagicMock()
    mock_pred.signature.instructions = "Test instructions"
    mock_pred.demos = []
    mock_pred.lm = None  # no per-predictor LM

    mock_compiled = MagicMock()
    mock_compiled.named_predictors.return_value = [("generate", mock_pred)]

    with (
        patch("daisy.optimize.dspy.LM") as mock_lm,
        patch("daisy.optimize.dspy.configure"),
        patch("daisy.optimize.Evaluate") as mock_eval_cls,
        patch("daisy.optimize.MIPROv2") as mock_mipro_cls,
    ):
        mock_eval_cls.return_value.side_effect = [0.5, 0.8]  # baseline, optimized
        mock_mipro_cls.return_value.compile.return_value = mock_compiled

        yield {
            "lm": mock_lm,
            "eval_cls": mock_eval_cls,
            "mipro_cls": mock_mipro_cls,
            "compiled": mock_compiled,
            "predictor": mock_pred,
        }
