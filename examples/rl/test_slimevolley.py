import jax

from simpleneat import algorithm, genome, problem
from simpleneat.common import ACT, AGG, State


def main():
    state = State(randkey=jax.random.PRNGKey(42))

    neat = algorithm.NEAT(
        pop_size=8,
        species_size=4,
        genome=genome.DefaultGenome(
            num_inputs=12,
            num_outputs=3,
            max_nodes=24,
            max_conns=64,
            node_gene=genome.BiasNode(
                activation_options=[ACT.tanh, ACT.relu, ACT.identity],
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.tanh,
        ),
        use_backprop=False,
    )

    env = problem.SlimeVolleyEnv(max_step=300, repeat_times=1)
    state = neat.setup(state)
    state = env.setup(state)

    population = neat.ask(state)
    candidate = (population[0][0], population[1][0])
    transformed = neat.transform(state, candidate)
    fitness = env.evaluate(state, jax.random.PRNGKey(0), neat.forward, transformed)
    print(f"single rollout fitness: {fitness}")


if __name__ == "__main__":
    main()
