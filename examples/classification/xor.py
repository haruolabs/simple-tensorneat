from simpleneat.algorithm.neat import NEAT
from simpleneat.common import ACT, AGG
from simpleneat.genome import BiasNode, DefaultGenome
from simpleneat.pipeline import Pipeline
from simpleneat.problem.classification import (
    ClassificationProblem,
    generate_xor_dataset,
)


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
                max_nodes=32,
                max_conns=64,
                node_gene=BiasNode(
                    activation_options=[ACT.tanh, ACT.sigmoid, ACT.relu, ACT.sin],
                    aggregation_options=AGG.sum,
                ),
                output_transform=ACT.sigmoid,
            ),
            use_backprop=True,
            backprop_steps=100,
            backprop_learning_rate=0.01,
        ),
        problem=ClassificationProblem(
            dataset_generator=generate_xor_dataset,
            train_size=500,
            penalty_connection_factor=0.01,
            seed=42,
        ),
        generation_limit=200,
        fitness_target=-0.008,
        seed=42,
        log_path="results/xor_history.csv",
    )

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    pipeline.show(
        state,
        best,
        test_decision_boundary_path="results/xor_test_decision_boundary.svg",
        train_decision_boundary_path="results/xor_train_decision_boundary.svg",
        x_range=(-5.0, 5.0),
        y_range=(-5.0, 5.0),
    )
    print(pipeline.algorithm.genome.repr(state, *best))


if __name__ == "__main__":
    main()
