# Daisy — Design Document

_2026-03-07_

---

## 1. Goal

Daisy is a thin wrapper around DSPy's MIPROv2 optimizer that accepts a DSPy module, a labeled dataset, a metric function, and an LM — and returns the optimized prompt instructions and few-shot demos as portable plain-Python artifacts. It is used offline as a prompt compiler inside a broader eval loop; it has no inference responsibilities.

---

## 2. Constraints and Non-Goals

**In scope:**
- Running MIPROv2 optimization over a caller-supplied `dspy.Module`
- Returning optimized instructions and demos as plain strings and dicts (no DSPy dependency required at inference time)
- Returning the baseline artifacts if optimization yields no improvement
- Accepting any LM string supported by LiteLLM

**Explicitly out of scope:**
- Inference / running the optimized module in production
- Embedding prompt optimization (treat embedding prefixes as hyperparameters, not prompt optimization)
- Assertion / retry logic (`dspy.Assert`, `dspy.Suggest`)
- Weight fine-tuning (`dspy.BootstrapFinetune`)
- Checkpointing or resuming partial optimization runs
- Metric construction (caller owns the metric function entirely)
- Multi-LM pipelines (modules with per-predictor LMs wired in are not supported)
- Async metric functions
- Any UI, CLI, or server layer — including the existing `__main__.py`, which must remain empty
- Routing LLM calls through headwater; if using local models, run Daisy on the same machine as Ollama
- Progress callbacks or streaming optimization status
- Persisting the DSPy program object to disk (Daisy returns artifacts, not the compiled program)
- Thread safety across concurrent `optimize()` calls in the same process — callers must not call `optimize()` concurrently; `dspy.configure()` is global state and will produce undefined behavior if stomped

**DSPy LM caching:** DSPy enables LM call caching by default. Daisy disables it for every run. Stale cached responses would silently corrupt optimization results.

---

## 3. Interface Contracts

### Core types

```python
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Protocol, runtime_checkable
import dspy


Example = dict[str, str | list | dict]   # raw input/output pair before DSPy wrapping
                                          # values may be any JSON-serializable type;
                                          # Daisy does not enforce str-only values


@runtime_checkable
class Metric(Protocol):
    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        ...
# Runtime enforcement: isinstance(metric, Metric) checks callability only.
# Daisy also performs a test call on the first trainset example before any
# LM call is made. If the test call raises or returns a non-finite float,
# Daisy raises TypeError immediately.


@dataclass(frozen=True)
class PredictorArtifact:
    name: str
    instructions: str                          # the raw string from predictor.signature.instructions;
                                               # no post-processing or markup stripping is applied
    demos: tuple[dict[str, str], ...] = field(default_factory=tuple)
    # demo field values are always str; non-str DSPy demo values are cast via str()


@dataclass(frozen=True)
class OptimizationResult:
    predictors: tuple[PredictorArtifact, ...]  # one per named predictor, in named_predictors() order
    baseline_score: float
    optimized_score: float
    improved: bool                             # True if optimized_score >= baseline_score
    duration_seconds: float                    # wall-clock time for the full optimize() call
```

### Parameter precedence rule

When `num_candidates`, `num_trials`, or `max_bootstrapped_demos` are explicitly provided alongside `auto`, the explicit values take precedence and `auto` is used only for any remaining unset parameters. This is implemented by passing explicit values directly to `MIPROv2` and letting DSPy's `auto` fill in the rest.

Known `auto="light"` defaults (from DSPy source as of 2.6): `num_candidates=7`, `num_trials=13`, `max_bootstrapped_demos=3`. These are informational; Daisy does not hardcode them.

### Primary entry point

```python
def optimize(
    module: dspy.Module,
    trainset: list[Example],
    input_keys: list[str],
    metric: Metric,
    lm: str,                                   # LiteLLM model string e.g. "anthropic/claude-sonnet-4-6"
    api_base: str | None = None,               # for Ollama or OpenAI-compatible endpoints
    api_key: str | None = None,                # None and "" are both treated as "not provided";
                                               # LiteLLM env var lookup proceeds normally
    auto: str = "light",                       # MIPROv2 auto mode: "light" | "medium" | "heavy"
    num_candidates: int | None = None,         # overrides auto if provided; must be >= 1
    num_trials: int | None = None,             # overrides auto if provided; must be >= 1
    max_bootstrapped_demos: int | None = None, # overrides auto if provided; 0 is valid
                                               # (instruction-only optimization, no few-shot)
    num_threads: int = 4,                      # concurrent module evaluations; must be >= 1
) -> OptimizationResult:
    ...
```

**Module mutation:** Daisy deep-copies the caller's module before passing it to MIPROv2. The caller's original module object is never modified.

### Data shape

```python
# Caller provides raw dicts; Daisy wraps them into dspy.Example internally
trainset = [
    {"question": "What is X?", "answer": "X is ..."},
    ...
]
input_keys = ["question"]   # keys not in input_keys are treated as labels
                            # all input_keys must be present in every example
```

### Metric signature

```python
def my_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    ...
# Daisy does not enforce the [0.0, 1.0] range.
# Daisy does enforce: return value must be a finite float (not NaN, not inf, not bool, not int).
# bool and int are explicitly rejected even though Python would accept them.
```

---

## 4. Acceptance Criteria

- `optimize()` returns an `OptimizationResult` for every successful run, whether or not improvement was found
- `result.improved` is `True` when `result.optimized_score >= result.baseline_score`, `False` otherwise — always
- When `result.improved` is `False`, `result.predictors` contains artifacts extracted from the module's state **before** `compile()` was called
- When `result.improved` is `True`, `result.predictors` contains artifacts extracted from the optimized module
- `result.predictors` contains exactly one `PredictorArtifact` per entry in `module.named_predictors()`, in the same order
- `PredictorArtifact.instructions` is the unmodified string from `predictor.signature.instructions` after optimization (or before, if no improvement)
- `PredictorArtifact.demos` contains only `str` values in each dict; non-str values from DSPy demos are cast via `str()`
- Calling `optimize()` with a non-callable `metric` raises `TypeError` before any LM call is made
- Calling `optimize()` with a metric whose test call raises raises `TypeError` before any LM call is made
- Calling `optimize()` with a metric whose test call returns a non-finite float or a non-float type raises `TypeError` before any LM call is made
- Calling `optimize()` with an empty `trainset` raises `ValueError` before any LM call is made
- Calling `optimize()` with a module that has zero named predictors raises `ValueError` before any LM call is made
- Calling `optimize()` with `input_keys` absent from any trainset example raises `ValueError` before any LM call is made
- Calling `optimize()` with `num_threads < 1` raises `ValueError` before any LM call is made
- Calling `optimize()` with `num_candidates < 1` or `num_trials < 1` raises `ValueError` before any LM call is made
- Calling `optimize()` with `max_bootstrapped_demos < 0` raises `ValueError` before any LM call is made (`0` is valid)
- Calling `optimize()` with `auto` outside `{"light", "medium", "heavy"}` raises `ValueError` before any LM call is made
- Calling `optimize()` with a module that has any per-predictor LM set raises `ValueError` before any LM call is made
- Any exception raised during optimization (network error, LM failure, metric exception) propagates with its original type and message unchanged — Daisy does not wrap exceptions
- The caller's `module` object is in the same state after `optimize()` as it was before — Daisy operates on a deep copy
- `result.duration_seconds` is a positive finite float
- Mutating any field of `OptimizationResult` or `PredictorArtifact` raises `FrozenInstanceError`

---

## 5. Error Handling / Failure Modes

| Failure | Behavior |
|---|---|
| Empty `trainset` | `ValueError` before any LM call |
| `input_keys` absent from any example | `ValueError` before any LM call |
| Module has zero named predictors | `ValueError` before any LM call |
| Non-callable `metric` | `TypeError` before any LM call |
| Metric test call raises | `TypeError` before any LM call |
| Metric test call returns non-finite or non-float | `TypeError` before any LM call |
| `num_threads < 1` | `ValueError` before any LM call |
| `num_candidates < 1` or `num_trials < 1` | `ValueError` before any LM call |
| `max_bootstrapped_demos < 0` | `ValueError` before any LM call |
| `auto` not in `{"light", "medium", "heavy"}` | `ValueError` before any LM call |
| Module has per-predictor LMs set | `ValueError` with message: "Per-predictor LMs are not supported. Set a single LM via the `lm` parameter." |
| LM call fails (rate limit, network) | Propagate exception as-is |
| Metric function raises during optimization | Propagate exception as-is |
| MIPROv2 internal error | Propagate exception as-is |
| Optimization finds no improvement | Return `OptimizationResult` with `improved=False` and pre-compile artifacts |
| `module.forward()` raises during baseline scoring | Propagate exception as-is |

No exception wrapping. No retries. No silent fallbacks except the no-improvement case.

---

## 6. Observability

Daisy logs at `INFO` level using the standard `logging` module under the logger name `"daisy"`. No third-party logging dependencies.

| Event | Level | Fields |
|---|---|---|
| `optimize()` called | `INFO` | module class name, trainset size, lm, auto, num_threads |
| Baseline scored | `INFO` | baseline_score |
| Optimization complete | `INFO` | optimized_score, improved, duration_seconds |
| No improvement found | `INFO` | baseline_score, optimized_score |

Daisy does **not** log: prompt content, instructions, demos, or any trainset values. These may contain sensitive data and are the caller's responsibility to log if needed.

`OptimizationResult.duration_seconds` is the only timing artifact returned to the caller. LM call counts and token usage are not tracked by Daisy.

---

## 7. Style Example

```python
import dspy
from daisy import optimize

class Summarize(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("document -> summary")

    def forward(self, document):
        return self.generate(document=document)

def metric(example, prediction):
    return 1.0 if example.summary.lower() in prediction.summary.lower() else 0.0

trainset = [
    {"document": "Apple released a new iPhone...", "summary": "Apple launched iPhone"},
    ...
]

# hosted API — picks up ANTHROPIC_API_KEY from env
result = optimize(
    module=Summarize(),
    trainset=trainset,
    input_keys=["document"],
    metric=metric,
    lm="anthropic/claude-sonnet-4-6",
    auto="light",
)

# local Ollama — run Daisy on the same machine as Ollama
result = optimize(
    module=Summarize(),
    trainset=trainset,
    input_keys=["document"],
    metric=metric,
    lm="ollama_chat/llama3.2:1b",
    api_base="http://localhost:11434",
    api_key="",
    num_threads=2,
)

print(result.improved)           # True / False
print(result.baseline_score)
print(result.optimized_score)
print(result.duration_seconds)

for artifact in result.predictors:
    print(artifact.name)
    print(artifact.instructions)  # plain string, ready to use anywhere
    print(artifact.demos)         # tuple of dicts
```

---

## 8. Domain Language

These are the only nouns the implementation is allowed to use:

| Term | Definition |
|---|---|
| **module** | A `dspy.Module` subclass supplied by the caller. Daisy deep-copies it before optimization and never modifies the original. |
| **predictor** | A named `dspy.Predict` or `dspy.ChainOfThought` instance within a module, as returned by `module.named_predictors()` |
| **trainset** | The caller-supplied list of raw `dict` examples used for optimization |
| **example** | A single `dict` in the trainset, or its `dspy.Example` wrapper inside Daisy |
| **metric** | The caller-supplied scoring function conforming to the `Metric` protocol |
| **instructions** | The raw string from `predictor.signature.instructions`; never post-processed |
| **demos** | The tuple of few-shot example dicts attached to a predictor after optimization; values are always `str` |
| **artifact** | A `PredictorArtifact`: the portable, DSPy-free output for one predictor |
| **result** | An `OptimizationResult`: the full output of one `optimize()` call |
| **baseline** | The module's score and predictor state captured before `compile()` is called |

---

## 9. Invalid State Transitions

These must raise errors — see Section 5 for full behavior:

- Calling `optimize()` with an empty `trainset`
- Calling `optimize()` with a module that has zero named predictors
- Calling `optimize()` with `input_keys` absent from any trainset example
- Calling `optimize()` with a non-callable or non-conforming `metric`
- Calling `optimize()` with a module that has per-predictor LMs set
- Calling `optimize()` with `auto` outside `{"light", "medium", "heavy"}`
- Calling `optimize()` with `num_threads < 1`
- Calling `optimize()` with `num_candidates < 1` or `num_trials < 1`
- Calling `optimize()` with `max_bootstrapped_demos < 0`
- Mutating any field of a returned `OptimizationResult` or `PredictorArtifact`
