import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from simpleneat.algorithm.neat import NEAT
from simpleneat.common import ACT, AGG
from simpleneat.genome import BiasNode, DefaultConn, DefaultGenome
from simpleneat.genome.operations import DefaultMutation
from simpleneat.pipeline import Pipeline
from simpleneat.problem.rl import EvoJAXSlimeVolleyEnv


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evolve a NEAT policy for EvoJAX SlimeVolley using self-play tournaments."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=256)
    parser.add_argument("--species-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--repeat-times", type=int, default=4)
    parser.add_argument("--max-step", type=int, default=1000)
    parser.add_argument(
        "--tournament-rounds",
        type=int,
        default=4,
        help="Number of random-pairing tournament rounds per generation.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="relative_ball",
        choices=["raw", "relative_ball"],
        help="Observation feature transform for the policy input.",
    )
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.75,
        help="Button activation threshold for forward/backward/jump outputs.",
    )
    parser.add_argument(
        "--hit-reward-scale",
        type=float,
        default=0.0,
        help="Additional shaping reward for right-agent ball contacts during self-play training.",
    )
    parser.add_argument("--fitness-target", type=float, default=15.0)
    parser.add_argument(
        "--log-path",
        type=str,
        default="examples/rl/slimevolley_evojax_selfplay_history.csv",
        help="CSV path for generation statistics. Use an empty string to disable logging.",
    )
    parser.add_argument(
        "--gif-path",
        type=str,
        default="examples/rl/slimevolley_evojax_selfplay_best.gif",
        help="Optional output path for a GIF of the best-genome replay against the internal agent.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=30,
        help="Frame rate for GIF export.",
    )
    parser.add_argument(
        "--eval-stop-lives-lost",
        type=int,
        default=4,
        help="During final replay, ignore max-step termination and stop once either side has lost this many lives.",
    )
    return parser


def build_pipeline(args, log_path):
    algorithm = NEAT(
        pop_size=args.pop_size,
        species_size=args.species_size,
        max_stagnation=30,
        species_elitism=3,
        spawn_number_change_rate=0.3,
        genome_elitism=2,
        survival_threshold=0.25,
        min_species_size=4,
        compatibility_threshold=1.25,
        genome=DefaultGenome(
            num_inputs=6 if args.feature_mode == "relative_ball" else 12,
            num_outputs=3,
            max_nodes=64,
            max_conns=256,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=[ACT.tanh, ACT.identity],
                aggregation_options=AGG.sum,
            ),
            conn_gene=DefaultConn(
                weight_init_std=0.75,
                weight_mutate_power=0.3,
                weight_mutate_rate=0.4,
            ),
            mutation=DefaultMutation(
                conn_add=0.35,
                conn_delete=0.08,
                node_add=0.12,
                node_delete=0.03,
            ),
            output_transform=ACT.identity,
        ),
        use_backprop=False,
    )
    problem = EvoJAXSlimeVolleyEnv(
        max_step=args.max_step,
        repeat_times=args.repeat_times,
        test=False,
        action_threshold=args.action_threshold,
        hit_reward_scale=args.hit_reward_scale,
        feature_mode=args.feature_mode,
    )
    return Pipeline(
        algorithm=algorithm,
        problem=problem,
        generation_limit=args.generations,
        fitness_target=args.fitness_target,
        seed=args.seed,
        log_path=log_path,
    )


def main():
    args = build_parser().parse_args()
    log_path = args.log_path or None
    gif_path = args.gif_path or None

    pipeline = build_pipeline(args, log_path)

    print(
        "Starting EvoJAX SlimeVolley self-play evolution with "
        f"pop_size={args.pop_size}, species_size={args.species_size}, "
        f"generations={args.generations}, tournament_rounds={args.tournament_rounds}, "
        f"repeat_times={args.repeat_times}, max_step={args.max_step}, "
        f"feature_mode={args.feature_mode}, action_threshold={args.action_threshold}, "
        f"hit_reward_scale={args.hit_reward_scale}",
        flush=True,
    )

    state = pipeline.setup()

    def selfplay_step(state_):
        eval_key, next_key = jax.random.split(state_.randkey)
        state_ = pipeline.algorithm.adapt(state_, pipeline.problem)
        population = pipeline.algorithm.ask(state_)
        transformed = jax.vmap(pipeline.algorithm.transform, in_axes=(None, 0))(state_, population)
        fitnesses = pipeline.problem.evaluate_population_self_play(
            state_,
            eval_key,
            pipeline.algorithm.forward,
            transformed,
            tournament_rounds=args.tournament_rounds,
            swap_sides=True,
        )
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        state_ = pipeline.algorithm.tell(state_, fitnesses)
        return state_.update(randkey=next_key), population, fitnesses

    print("start compile", flush=True)
    tic = time.time()
    compiled_step = jax.jit(selfplay_step).lower(state).compile()
    print(f"compile finished, cost time: {time.time() - tic:.6f}s", flush=True)

    for _ in range(args.generations):
        tic = time.time()
        state, previous_pop, fitnesses = compiled_step(state)
        cpu_fitnesses = np.asarray(jax.device_get(fitnesses))
        pipeline.analysis(state, previous_pop, cpu_fitnesses, time.time() - tic)
        if np.max(cpu_fitnesses) >= args.fitness_target:
            print("Fitness limit reached!", flush=True)
            break
    if int(state.generation) >= args.generations:
        print("Generation limit reached!", flush=True)

    best = pipeline.best_genome
    if best is None:
        raise RuntimeError("Evolution finished without producing a valid best genome.")

    print(f"best fitness seen during self-play training: {pipeline.best_fitness:.4f}")

    transformed = pipeline.algorithm.transform(state, best)

    eval_problem = EvoJAXSlimeVolleyEnv(
        max_step=args.max_step,
        repeat_times=1,
        test=True,
        action_threshold=args.action_threshold,
        hit_reward_scale=0.0,
        feature_mode=args.feature_mode,
    )
    internal_eval_reward = eval_problem.evaluate(
        state,
        state.randkey,
        pipeline.algorithm.forward,
        transformed,
    )
    print(f"best genome internal-agent evaluation reward: {float(internal_eval_reward):.4f}")
    print(pipeline.algorithm.genome.repr(state, *best))

    replay_reward = eval_problem.show(
        state,
        state.randkey,
        pipeline.algorithm.forward,
        transformed,
        gif_path=gif_path,
        gif_fps=args.gif_fps,
        stop_lives_lost=args.eval_stop_lives_lost,
    )
    print(f"best genome replay reward vs internal agent: {float(replay_reward):.4f}")


if __name__ == "__main__":
    main()
