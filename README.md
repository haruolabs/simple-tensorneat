# simpleneat

Minimal research-oriented port of the essential feed-forward NEAT pieces from
[`EMI-Group/tensorneat`](https://github.com/EMI-Group/tensorneat), rebuilt as a
smaller starting point under `src/`.

## Included

- `pipeline`, `problem`, `genome`, `algorithm`, and `common` package layout
- Feed-forward `NEAT`
- `DefaultGenome`, `DefaultNode`, `BiasNode`, `DefaultConn`
- `func_fit` problems: `FuncFit`, `CustomFuncFit`, `XOR`
- `rl` problem: `SlimeVolleyEnv`
- Optional gradient descent inside `NEAT`, enabled by default for differentiable
  problems such as function fitting
- Default yEd Live JSON export for the best genome after `Pipeline.auto_run()`
- ONNX export for feed-forward genomes
- Examples for XOR and a SlimeVolley smoke test

## Not Included

- HyperNEAT
- Recurrent genomes
- Brax, Gymnax, Mujoco adapters
- `xor3d.py`

## Install

```bash
pip install -e .
```

For the SlimeVolley example:

```bash
pip install -e ".[rl]"
```

JAX installation varies by CPU/GPU setup. See the official JAX installation
guide for the correct wheel for your machine.

## Example

```python
from simpleneat.pipeline import Pipeline
from simpleneat.algorithm.neat import NEAT
from simpleneat.genome import DefaultGenome, BiasNode
from simpleneat.problem.func_fit import XOR
from simpleneat.common import ACT, AGG

pipeline = Pipeline(
    algorithm=NEAT(
        pop_size=256,
        species_size=12,
        genome=DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            node_gene=BiasNode(
                activation_options=[ACT.tanh, ACT.sigmoid],
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.sigmoid,
        ),
        use_backprop=True,
        backprop_steps=3,
        backprop_learning_rate=0.05,
    ),
    problem=XOR(),
    generation_limit=200,
    fitness_target=-1e-4,
    seed=42,
)

state = pipeline.setup()
state, best = pipeline.auto_run(state)
pipeline.show(state, best)
```

By default, `Pipeline.auto_run()` also writes a yEd Live-friendly JSON file for
the best genome. If `log_path="results/xor_history.csv"`, the export is written
to `results/xor_best.yed.json`.

## Upstream Notes

This repository is intentionally smaller than upstream TensorNEAT. The goal is
to preserve the core package boundaries and the essential feed-forward NEAT
workflow while keeping the code easy to modify for research. The source is
flattened directly under `src/`, while packaging exposes it as `simpleneat`.
