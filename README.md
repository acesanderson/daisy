# Daisy

Daisy is a minimal wrapper around [DSPy](https://dspy.ai/)'s MIPROv2 optimizer. You give it a DSPy module, a labeled dataset, a metric function, and an LM. It gives back optimized prompt instructions and few-shot demos as plain Python — no DSPy dependency required at inference time.

---

## Install

```bash
uv add daisy
```

---

## Usage

```python
import dspy
from daisy import optimize

# 1. Define your DSPy module as normal
class Summarize(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("document -> summary")

    def forward(self, document):
        return self.generate(document=document)


# 2. Provide labeled examples as plain dicts
trainset = [
    {"document": "Apple released a new iPhone today...", "summary": "Apple launched iPhone"},
    {"document": "The Fed raised interest rates by 0.25%...", "summary": "Fed raises rates"},
    # ... ideally 50-200 examples
]


# 3. Define a metric function
def metric(example, prediction):
    return 1.0 if example.summary.lower() in prediction.summary.lower() else 0.0


# 4. Run optimization
result = optimize(
    module=Summarize(),
    trainset=trainset,
    input_keys=["document"],   # remaining keys are treated as labels
    metric=metric,
    lm="anthropic/claude-sonnet-4-6",
    auto="light",              # "light" | "medium" | "heavy"
)


# 5. Use the artifacts however you want
for artifact in result.predictors:
    print(artifact.name)
    print(artifact.instructions)   # plain string — paste into your pipeline
    print(artifact.demos)          # list of dicts — use as few-shot examples
```

### Result shape

```python
result.improved          # True if optimization beat the baseline
result.baseline_score    # float
result.optimized_score   # float

result.predictors        # list[PredictorArtifact]
artifact.name            # str  — predictor name as defined in your module
artifact.instructions    # str  — rewritten system instruction
artifact.demos           # list[dict[str, str]]  — few-shot examples; [] if none generated
```

---

## What Daisy abstracts away

Using DSPy directly for this workflow requires:

**Data wrangling.** DSPy requires examples to be wrapped in `dspy.Example` objects with `.with_inputs()` called on each one. Daisy accepts plain dicts and handles the wrapping internally.

**LM configuration.** DSPy requires a global `dspy.configure(lm=dspy.LM(...))` call before anything runs. Daisy accepts an LM string and handles configuration and teardown, scoped to the optimization run.

**Optimizer boilerplate.** Instantiating `MIPROv2`, passing the right kwargs, and calling `.compile()` correctly involves a handful of non-obvious parameters. Daisy exposes only `auto` mode, which covers the majority of use cases.

**Artifact extraction.** DSPy returns an optimized program object — a stateful Python class that requires DSPy to run. Extracting the actual instructions and demos as portable plain-Python values requires inspecting `predictor.signature.instructions` and `predictor.demos` across all named predictors. Daisy does this automatically and returns frozen, DSPy-free artifacts.

**Baseline comparison.** DSPy does not automatically tell you whether optimization improved on the baseline. Daisy scores the original module before optimizing and includes both scores in the result, returning the original artifacts if optimization yielded no improvement.

---

## What Daisy does not do

- Run your module at inference time
- Optimize embedding prompt prefixes
- Support async metric functions
- Support modules with per-predictor LMs
- Checkpoint or resume partial optimization runs
- Construct or validate your metric function beyond a basic callability check

These are deliberate non-goals. Daisy is an offline prompt compiler, not an inference framework.
