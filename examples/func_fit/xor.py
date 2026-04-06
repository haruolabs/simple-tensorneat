import jax

from simpleneat.pipeline import Pipeline
from simpleneat.algorithm.neat import NEAT
from simpleneat.genome import BiasNode, DefaultGenome
from simpleneat.problem.func_fit import XOR
from simpleneat.common import ACT, AGG
from simpleneat.visualizer import draw


def main():
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=256,
            species_size=12,
            survival_threshold=0.2,
            compatibility_threshold=1.5,
            genome=DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                max_nodes=16,
                max_conns=32,
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
    print(pipeline.algorithm.genome.repr(state, *best))
    draw(
        state,
        pipeline.algorithm.genome,
        individual=best,
        save_path="examples/func_fit/xor_network.svg",
        draw_weight_labels=True,
    )


if __name__ == "__main__":
    main()
