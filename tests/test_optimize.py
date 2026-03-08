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
