# Daisy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `daisy.optimize()` — a thin DSPy/MIPROv2 wrapper that returns portable prompt artifacts (plain strings + dicts) with no DSPy inference dependency.

**Architecture:** A single `optimize()` entry point in `optimize.py` delegates to a `_validate()` pre-flight function and a `_extract_artifacts()` helper. Types live in the already-created `types.py`. All DSPy infrastructure (`dspy.LM`, `dspy.configure`, `Evaluate`, `MIPROv2`) is patched in tests — no real LM calls during the test suite.

**Tech Stack:** Python 3.13, dspy-ai, pytest, unittest.mock (stdlib)

---

## Task 1: Add dependencies and test infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`

### Step 1: Add dspy-ai to pyproject.toml

```toml
[project]
dependencies = [
    "dspy-ai>=2.6",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Step 2: Verify install

```bash
cd /Users/bianders/vibe/daisy-project && uv sync
```

Expected: resolves without error

### Step 3: Create conftest.py with shared fixtures

```python
# tests/conftest.py
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
```

### Step 4: Run existing sanity test

```bash
cd /Users/bianders/vibe/daisy-project && uv run pytest tests/test_basic.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add pyproject.toml tests/conftest.py
git commit -m "chore: add dspy-ai dependency and test fixtures"
```

---

## Task 2: types.py — frozen mutation guard

> **AC 22:** Mutating any field of a returned `OptimizationResult` or `PredictorArtifact` raises `FrozenInstanceError`

**Files:**
- Create: `tests/test_types.py`

### Step 1: Write the failing test

```python
# tests/test_types.py
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_types.py -v
```

Expected: FAIL — `FrozenInstanceError` not in `daisy.types` (or wrong field on dataclass)

### Step 3: Verify types.py is already correct

`src/daisy/types.py` was created with `frozen=True` in an earlier session. Confirm:

```python
# src/daisy/types.py — should already contain:
@dataclass(frozen=True)
class PredictorArtifact: ...

@dataclass(frozen=True)
class OptimizationResult: ...
```

If `duration_seconds: float` is missing from `OptimizationResult`, add it now.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_types.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_types.py src/daisy/types.py
git commit -m "test: AC22 frozen dataclass mutation raises FrozenInstanceError"
```

---

## Task 3: Create optimize.py skeleton + AC 11 (empty trainset)

> **AC 11:** Calling `optimize()` with an empty `trainset` raises `ValueError` before any LM call is made

**Files:**
- Create: `src/daisy/optimize.py`
- Create: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
# tests/test_optimize.py
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_empty_trainset_raises_before_lm_call -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError` (optimize.py doesn't exist yet)

### Step 3: Write minimal implementation

```python
# src/daisy/optimize.py
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
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_empty_trainset_raises_before_lm_call -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/daisy/optimize.py tests/test_optimize.py
git commit -m "feat: optimize skeleton + AC11 empty trainset validation"
```

---

## Task 4: AC 13 — input_keys absent from example

> **AC 13:** Calling `optimize()` with `input_keys` absent from any trainset example raises `ValueError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_missing_input_key_raises_before_lm_call -v
```

Expected: FAIL

### Step 3: Add check to `_validate`

```python
# In _validate, after the empty trainset check:
for ex in trainset:
    for key in input_keys:
        if key not in ex:
            raise ValueError(f"input_key '{key}' not found in all trainset examples")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_missing_input_key_raises_before_lm_call -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC13 input_keys validation"
```

---

## Task 5: AC 12 — zero named predictors

> **AC 12:** Calling `optimize()` with a module that has zero named predictors raises `ValueError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_empty_module_raises_before_lm_call -v
```

Expected: FAIL

### Step 3: Add check to `_validate`

```python
# In _validate, after empty trainset check, before input_keys check:
if not list(module.named_predictors()):
    raise ValueError("module must have at least one named predictor")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_empty_module_raises_before_lm_call -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC12 zero predictors validation"
```

---

## Task 6: AC 8 — non-callable metric

> **AC 8:** Calling `optimize()` with a non-callable `metric` raises `TypeError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_non_callable_metric_raises_before_lm_call -v
```

Expected: FAIL

### Step 3: Add check to `_validate`

```python
# In _validate, after input_keys check:
if not callable(metric):
    raise TypeError("metric must be callable")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_non_callable_metric_raises_before_lm_call -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC8 non-callable metric validation"
```

---

## Task 7: AC 9 — metric test call raises

> **AC 9:** Calling `optimize()` with a metric whose test call raises raises `TypeError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_metric_that_raises_on_call_raises_type_error -v
```

Expected: FAIL

### Step 3: Add test-call to `_validate`

```python
# In _validate, after callable check:
test_example = dspy.Example(**trainset[0]).with_inputs(*input_keys)
try:
    test_result = metric(test_example, dspy.Prediction())
except Exception as exc:
    raise TypeError(f"metric raised during test call: {exc}") from exc
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_metric_that_raises_on_call_raises_type_error -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC9 metric test-call validation"
```

---

## Task 8: AC 10 — metric returns non-finite or non-float

> **AC 10:** Calling `optimize()` with a metric whose test call returns a non-finite float or non-float type raises `TypeError` before any LM call is made. `bool` and `int` are explicitly rejected.

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing tests

```python
import pytest

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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_metric_bad_return_raises_type_error -v
```

Expected: FAIL (all parametrized cases)

### Step 3: Add return-type check to `_validate`

```python
# In _validate, after the test_result = metric(...) call:
if (
    isinstance(test_result, bool)
    or not isinstance(test_result, float)
    or not math.isfinite(test_result)
):
    raise TypeError(
        f"metric must return a finite float, got {type(test_result).__name__}: {test_result!r}"
    )
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_metric_bad_return_raises_type_error -v
```

Expected: PASS (all 6 parametrized cases)

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC10 metric return-type validation"
```

---

## Task 9: AC 14 — num_threads < 1

> **AC 14:** Calling `optimize()` with `num_threads < 1` raises `ValueError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_num_threads_less_than_one_raises -v
```

Expected: FAIL

### Step 3: Add check to `_validate`

```python
if num_threads < 1:
    raise ValueError(f"num_threads must be >= 1, got {num_threads}")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_num_threads_less_than_one_raises -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC14 num_threads validation"
```

---

## Task 10: AC 15a — num_candidates < 1

> **AC 15a:** Calling `optimize()` with `num_candidates < 1` raises `ValueError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_num_candidates_less_than_one_raises -v
```

Expected: FAIL

### Step 3: Add check

```python
if num_candidates is not None and num_candidates < 1:
    raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_num_candidates_less_than_one_raises -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC15a num_candidates validation"
```

---

## Task 11: AC 15b — num_trials < 1

> **AC 15b:** Calling `optimize()` with `num_trials < 1` raises `ValueError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_num_trials_less_than_one_raises -v
```

Expected: FAIL

### Step 3: Add check

```python
if num_trials is not None and num_trials < 1:
    raise ValueError(f"num_trials must be >= 1, got {num_trials}")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_num_trials_less_than_one_raises -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC15b num_trials validation"
```

---

## Task 12: AC 16 — max_bootstrapped_demos < 0

> **AC 16:** Calling `optimize()` with `max_bootstrapped_demos < 0` raises `ValueError`. `0` is valid (instruction-only optimization).

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing tests

```python
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
    # 0 is valid — should not raise, reaches NotImplementedError
    with pytest.raises(NotImplementedError):
        optimize(
            module=simple_module,
            trainset=trainset,
            input_keys=["question"],
            metric=good_metric,
            lm="openai/gpt-4o-mini",
            max_bootstrapped_demos=0,
        )
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_max_bootstrapped_demos_negative_raises tests/test_optimize.py::test_max_bootstrapped_demos_zero_is_valid -v
```

Expected: FAIL

### Step 3: Add check

```python
if max_bootstrapped_demos is not None and max_bootstrapped_demos < 0:
    raise ValueError(f"max_bootstrapped_demos must be >= 0, got {max_bootstrapped_demos}")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_max_bootstrapped_demos_negative_raises tests/test_optimize.py::test_max_bootstrapped_demos_zero_is_valid -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC16 max_bootstrapped_demos validation"
```

---

## Task 13: AC 17 — auto outside allowed set

> **AC 17:** Calling `optimize()` with `auto` outside `{"light", "medium", "heavy"}` raises `ValueError` before any LM call is made

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_invalid_auto_raises -v
```

Expected: FAIL

### Step 3: Add check

```python
if auto not in {"light", "medium", "heavy"}:
    raise ValueError(f"auto must be 'light', 'medium', or 'heavy', got {auto!r}")
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_invalid_auto_raises -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC17 auto mode validation"
```

---

## Task 14: AC 18 — per-predictor LM

> **AC 18:** Calling `optimize()` with a module that has any per-predictor LM set raises `ValueError` with a specific message

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py`

### Step 1: Write the failing test

```python
from unittest.mock import MagicMock

def test_per_predictor_lm_raises(simple_module, trainset, good_metric):
    simple_module.generate.lm = MagicMock()  # simulate per-predictor LM being set

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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_per_predictor_lm_raises -v
```

Expected: FAIL

### Step 3: Add check

```python
# In _validate, at the end:
for _name, predictor in module.named_predictors():
    if getattr(predictor, "lm", None) is not None:
        raise ValueError(
            "Per-predictor LMs are not supported. Set a single LM via the `lm` parameter."
        )
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_per_predictor_lm_raises -v
```

Expected: PASS

### Step 5: Run all validation tests together

```bash
uv run pytest tests/test_optimize.py -v
```

Expected: All PASS

### Step 6: Commit

```bash
git add tests/test_optimize.py src/daisy/optimize.py
git commit -m "feat: AC18 per-predictor LM validation — all validation complete"
```

---

## Task 15: AC 20 — caller's module is not mutated

> **AC 20:** The caller's `module` object is in the same state after `optimize()` as it was before — Daisy operates on a deep copy

**Files:**
- Modify: `tests/test_optimize.py`
- Modify: `src/daisy/optimize.py` (add core optimization flow, replacing `NotImplementedError`)

### Step 1: Write the failing test

```python
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
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_caller_module_not_mutated -v
```

Expected: FAIL — `NotImplementedError`

### Step 3: Implement the core optimization flow

Replace the `NotImplementedError` in `optimize()` with the full implementation:

```python
def optimize(
    module,
    trainset,
    input_keys,
    metric,
    lm,
    api_base=None,
    api_key=None,
    auto="light",
    num_candidates=None,
    num_trials=None,
    max_bootstrapped_demos=None,
    num_threads=4,
):
    _validate(
        module, trainset, input_keys, metric, auto,
        num_candidates, num_trials, max_bootstrapped_demos, num_threads,
    )

    start = time.monotonic()

    dspy_trainset = [dspy.Example(**ex).with_inputs(*input_keys) for ex in trainset]

    # Deep copy — caller's module is never touched
    working_module = copy.deepcopy(module)

    # Configure LM with caching disabled
    lm_kwargs: dict = {"cache": False}
    if api_base is not None:
        lm_kwargs["api_base"] = api_base
    if api_key:  # treats "" and None identically — let LiteLLM use env var
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
```

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_caller_module_not_mutated -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/daisy/optimize.py tests/test_optimize.py
git commit -m "feat: AC20 core optimization flow with deep copy"
```

---

## Task 16: AC 6 — instructions are the unmodified string

> **AC 6:** `PredictorArtifact.instructions` is the unmodified string from `predictor.signature.instructions` after optimization

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
def test_instructions_are_unmodified_string(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["predictor"].signature.instructions = "You are a precise answerer."

    result = optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )

    assert result.predictors[0].instructions == "You are a precise answerer."
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_instructions_are_unmodified_string -v
```

Expected: FAIL

### Step 3: Verify implementation

`_extract_artifacts` already does `instructions=predictor.signature.instructions`. No change needed — the test should reveal whether the mock wiring is correct.

If failing: check that `mock_dspy["compiled"].named_predictors.return_value` returns the mock predictor with the patched instructions. The `mock_dspy` fixture in conftest sets this up.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_instructions_are_unmodified_string -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC6 instructions are unmodified from predictor.signature.instructions"
```

---

## Task 17: AC 7 — demo values cast to str

> **AC 7:** `PredictorArtifact.demos` contains only `str` values; non-str values from DSPy demos are cast via `str()`

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
def test_demo_values_cast_to_str(simple_module, trainset, good_metric, mock_dspy):
    # DSPy demo with a non-str value (a list)
    mock_dspy["predictor"].demos = [{"question": "Q?", "scores": [1, 2, 3]}]

    result = optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )

    demos = result.predictors[0].demos
    assert len(demos) == 1
    assert demos[0]["scores"] == "[1, 2, 3]"
    assert isinstance(demos[0]["scores"], str)
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_demo_values_cast_to_str -v
```

Expected: FAIL

### Step 3: Verify implementation

`_extract_artifacts` already does `{k: str(v) for k, v in demo.items()}`. No change needed. If failing, investigate mock wiring.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_demo_values_cast_to_str -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC7 non-str demo values cast to str"
```

---

## Task 18: AC 5 — one artifact per predictor in named_predictors() order

> **AC 5:** `result.predictors` contains exactly one `PredictorArtifact` per entry in `module.named_predictors()`, in the same order

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
def test_one_artifact_per_predictor_in_order(simple_module, trainset, good_metric, mock_dspy):
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
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )

    assert len(result.predictors) == 2
    assert result.predictors[0].name == "step_one"
    assert result.predictors[0].instructions == "Instr A"
    assert result.predictors[1].name == "step_two"
    assert result.predictors[1].instructions == "Instr B"
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_one_artifact_per_predictor_in_order -v
```

Expected: FAIL

### Step 3: Verify implementation

`_extract_artifacts` iterates `module.named_predictors()` in order. No change needed. If failing: investigate mock wiring.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_one_artifact_per_predictor_in_order -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC5 one artifact per predictor in named_predictors() order"
```

---

## Task 19: AC 2 — improved flag logic

> **AC 2:** `result.improved` is `True` when `optimized_score >= baseline_score`, `False` otherwise — always

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing tests

```python
def test_improved_true_when_optimized_score_exceeds_baseline(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = [0.5, 0.8]  # baseline, optimized
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
    mock_dspy["eval_cls"].return_value.side_effect = [0.8, 0.5]  # baseline > optimized
    result = optimize(
        module=simple_module, trainset=trainset, input_keys=["question"],
        metric=good_metric, lm="openai/gpt-4o-mini",
    )
    assert result.improved is False
```

### Step 2: Run to verify they fail

```bash
uv run pytest tests/test_optimize.py::test_improved_true_when_optimized_score_exceeds_baseline tests/test_optimize.py::test_improved_true_when_scores_equal tests/test_optimize.py::test_improved_false_when_optimized_score_below_baseline -v
```

Expected: FAIL

### Step 3: Verify implementation

`optimize()` already does `improved = optimized_score >= baseline_score`. No change needed. If failing, investigate Evaluate mock `side_effect` not being consumed correctly.

### Step 4: Run to verify they pass

```bash
uv run pytest tests/test_optimize.py::test_improved_true_when_optimized_score_exceeds_baseline tests/test_optimize.py::test_improved_true_when_scores_equal tests/test_optimize.py::test_improved_false_when_optimized_score_below_baseline -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC2 improved flag is True iff optimized_score >= baseline_score"
```

---

## Task 20: AC 3 — baseline artifacts returned when no improvement

> **AC 3:** When `result.improved` is `False`, `result.predictors` contains artifacts extracted from the module's state **before** `compile()` was called

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
def test_returns_baseline_artifacts_when_no_improvement(simple_module, trainset, good_metric, mock_dspy):
    # Baseline predictor has different instructions than the compiled one
    mock_dspy["eval_cls"].return_value.side_effect = [0.8, 0.5]  # optimized < baseline

    # The working_module (deep copy of simple_module) has its own predictor.
    # The compiled module returns mock_dspy["predictor"] with "Test instructions".
    # When no improvement, we should get the baseline (working_module) instructions, not the compiled ones.

    result = optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )

    assert result.improved is False
    # Baseline artifacts come from working_module (deep copy of simple_module),
    # not from the compiled mock which has "Test instructions"
    assert result.predictors[0].instructions != "Test instructions"
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_returns_baseline_artifacts_when_no_improvement -v
```

Expected: FAIL

### Step 3: Verify implementation

`optimize()` already does:
```python
baseline_artifacts = _extract_artifacts(working_module)  # captured before compile()
...
artifacts = _extract_artifacts(optimized_module) if improved else baseline_artifacts
```

No change needed. If failing: the working_module's predictor instructions differ from the mock compiled predictor's instructions — confirm this is true by checking what `dspy.Predict("question -> answer").signature.instructions` returns out of the box.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_returns_baseline_artifacts_when_no_improvement -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC3 baseline artifacts returned when no improvement"
```

---

## Task 21: AC 4 — optimized artifacts returned when improved

> **AC 4:** When `result.improved` is `True`, `result.predictors` contains artifacts extracted from the optimized module

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
def test_returns_optimized_artifacts_when_improved(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = [0.5, 0.8]  # improved
    mock_dspy["predictor"].signature.instructions = "Optimized by MIPROv2"

    result = optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )

    assert result.improved is True
    assert result.predictors[0].instructions == "Optimized by MIPROv2"
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_returns_optimized_artifacts_when_improved -v
```

Expected: FAIL

### Step 3: Verify implementation

Already handled. If failing, investigate mock wiring.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_returns_optimized_artifacts_when_improved -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC4 optimized artifacts returned when improved"
```

---

## Task 22: AC 1 — returns OptimizationResult

> **AC 1:** `optimize()` returns an `OptimizationResult` for every successful run, whether or not improvement was found

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
from daisy.types import OptimizationResult

def test_always_returns_optimization_result(simple_module, trainset, good_metric, mock_dspy):
    result = optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )
    assert isinstance(result, OptimizationResult)
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_always_returns_optimization_result -v
```

Expected: FAIL

### Step 3: Verify implementation

Already handled. If failing, investigate return path.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_always_returns_optimization_result -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC1 optimize always returns OptimizationResult"
```

---

## Task 23: AC 21 — duration_seconds is positive finite float

> **AC 21:** `result.duration_seconds` is a positive finite float

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
import math

def test_duration_seconds_is_positive_finite_float(simple_module, trainset, good_metric, mock_dspy):
    result = optimize(
        module=simple_module,
        trainset=trainset,
        input_keys=["question"],
        metric=good_metric,
        lm="openai/gpt-4o-mini",
    )
    assert isinstance(result.duration_seconds, float)
    assert math.isfinite(result.duration_seconds)
    assert result.duration_seconds > 0
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_duration_seconds_is_positive_finite_float -v
```

Expected: FAIL

### Step 3: Verify implementation

`optimize()` already does:
```python
start = time.monotonic()
...
duration = time.monotonic() - start
```

No change needed.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_duration_seconds_is_positive_finite_float -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC21 duration_seconds is positive finite float"
```

---

## Task 24: AC 19 — exceptions propagate unchanged

> **AC 19:** Any exception raised during optimization propagates with its original type and message unchanged — Daisy does not wrap exceptions

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
def test_mipro_exception_propagates_unchanged(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["mipro_cls"].return_value.compile.side_effect = RuntimeError("network failure")

    with pytest.raises(RuntimeError, match="network failure"):
        optimize(
            module=simple_module,
            trainset=trainset,
            input_keys=["question"],
            metric=good_metric,
            lm="openai/gpt-4o-mini",
        )


def test_evaluate_exception_propagates_unchanged(simple_module, trainset, good_metric, mock_dspy):
    mock_dspy["eval_cls"].return_value.side_effect = ConnectionError("rate limited")

    with pytest.raises(ConnectionError, match="rate limited"):
        optimize(
            module=simple_module,
            trainset=trainset,
            input_keys=["question"],
            metric=good_metric,
            lm="openai/gpt-4o-mini",
        )
```

### Step 2: Run to verify they fail

```bash
uv run pytest tests/test_optimize.py::test_mipro_exception_propagates_unchanged tests/test_optimize.py::test_evaluate_exception_propagates_unchanged -v
```

Expected: FAIL

### Step 3: Verify implementation

No `try/except` wrapping exists in `optimize()`. Exceptions propagate naturally. If failing, investigate whether something is swallowing them.

### Step 4: Run to verify they pass

```bash
uv run pytest tests/test_optimize.py::test_mipro_exception_propagates_unchanged tests/test_optimize.py::test_evaluate_exception_propagates_unchanged -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: AC19 exceptions propagate with original type and message"
```

---

## Task 25: Logging — observability

> Per design doc Section 6: log at INFO on call start, baseline score, completion

**Files:**
- Modify: `tests/test_optimize.py`

### Step 1: Write the failing test

```python
import logging

def test_logs_start_baseline_and_completion(simple_module, trainset, good_metric, mock_dspy, caplog):
    with caplog.at_level(logging.INFO, logger="daisy"):
        optimize(
            module=simple_module,
            trainset=trainset,
            input_keys=["question"],
            metric=good_metric,
            lm="openai/gpt-4o-mini",
        )

    messages = [r.message for r in caplog.records]
    assert any("Starting optimization" in m for m in messages)
    assert any("Baseline score" in m for m in messages)
    assert any("Optimization complete" in m for m in messages)
```

### Step 2: Run to verify it fails

```bash
uv run pytest tests/test_optimize.py::test_logs_start_baseline_and_completion -v
```

Expected: FAIL

### Step 3: Verify implementation

`optimize()` already has the three `logger.info(...)` calls. If failing, check that `logger = logging.getLogger("daisy")` is defined at module level in `optimize.py`.

### Step 4: Run to verify it passes

```bash
uv run pytest tests/test_optimize.py::test_logs_start_baseline_and_completion -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_optimize.py
git commit -m "test: observability — logging at start, baseline, and completion"
```

---

## Task 26: Wire __init__.py and run full suite

**Files:**
- Modify: `src/daisy/__init__.py`

### Step 1: Update __init__.py

```python
# src/daisy/__init__.py
from daisy.optimize import optimize
from daisy.types import OptimizationResult, PredictorArtifact

__all__ = [
    "optimize",
    "OptimizationResult",
    "PredictorArtifact",
]
```

### Step 2: Run the full test suite

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS

### Step 3: Commit

```bash
git add src/daisy/__init__.py
git commit -m "feat: wire public API in __init__.py — daisy v0.1.0 complete"
```
