import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from simpleneat.algorithm.neat import NEAT
from simpleneat.common import ACT, AGG
from simpleneat.genome import BiasNode, DefaultConn, DefaultGenome
from simpleneat.genome.operations import DefaultMutation
from simpleneat.pipeline import Pipeline
from simpleneat.problem.rl import EvoJAXSlimeVolleyEnv


def save_best_genome(save_path, args, best, first_test_reward, extra_eval_rewards, pipeline):
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "best_nodes": np.asarray(jax.device_get(best[0])),
        "best_conns": np.asarray(jax.device_get(best[1])),
        "best_fitness_seen": float(pipeline.best_fitness),
        "first_test_reward": float(first_test_reward),
        "extra_eval_rewards": [float(v) for v in extra_eval_rewards],
        "extra_eval_mean": float(np.mean(extra_eval_rewards)) if extra_eval_rewards else None,
    }
    with output_path.open("wb") as fp:
        pickle.dump(payload, fp)
    print(f"saved best genome: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evolve a NEAT policy for the EvoJAX SlimeVolley task."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=256)
    parser.add_argument("--species-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=1000) # 400
    parser.add_argument("--repeat-times", type=int, default=4)
    parser.add_argument("--max-step", type=int, default=1000) # 1000
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
        default=0.9, # 0.75
        help="Button activation threshold for forward/backward/jump outputs.",
    )
    parser.add_argument(
        "--hit-reward-scale",
        type=float,
        default=0.0, # 0.05
        help="Additional shaping reward for right-agent ball contacts during training.",
    )
    parser.add_argument("--fitness-target", type=float, default=15.0)
    parser.add_argument(
        "--log-path",
        type=str,
        default="results/slimevolley_evojax_history.csv",
        help="CSV path for generation statistics. Use an empty string to disable logging.",
    )
    parser.add_argument(
        "--gif-path",
        type=str,
        default="results/slimevolley_evojax_best.gif",
        help="Optional output path for a GIF of the best genome replay. Use an empty string to disable.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=15, #25
        help="Frame rate for GIF export.",
    )
    parser.add_argument(
        "--eval-stop-lives-lost",
        type=int,
        default=4,
        help="During final replay, ignore max-step termination and stop once either side has lost this many lives.",
    )
    parser.add_argument(
        "--save-threshold",
        type=float,
        default=4.0,
        help="If the first test replay reward reaches this value, run extra evaluations and save the genome.",
    )
    parser.add_argument(
        "--extra-eval-games",
        type=int,
        default=8,
        help="Number of additional test=True evaluations to average when the save threshold is reached.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="results/slimevolley_evojax_best_genome.pkl",
        help="Output path for saving the best genome and metadata once the save threshold is reached.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    log_path = args.log_path or None
    gif_path = args.gif_path or None

    pipeline = Pipeline(
        algorithm=NEAT(
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
                max_nodes=32, # 64
                max_conns=128, # 256
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
        ),
        problem=EvoJAXSlimeVolleyEnv(
            max_step=args.max_step,
            repeat_times=args.repeat_times,
            test=False,
            action_threshold=args.action_threshold,
            hit_reward_scale=args.hit_reward_scale,
            feature_mode=args.feature_mode,
        ),
        generation_limit=args.generations,
        fitness_target=args.fitness_target,
        seed=args.seed,
        log_path=log_path,
    )

    print(
        "Starting EvoJAX SlimeVolley evolution with "
        f"pop_size={args.pop_size}, species_size={args.species_size}, "
        f"generations={args.generations}, repeat_times={args.repeat_times}, "
        f"max_step={args.max_step}, feature_mode={args.feature_mode}, "
        f"action_threshold={args.action_threshold}, "
        f"hit_reward_scale={args.hit_reward_scale}",
        flush=True,
    )

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)

    if best is None:
        raise RuntimeError("Evolution finished without producing a valid best genome.")

    transformed = pipeline.algorithm.transform(state, best)

    train_fitness = pipeline.problem.evaluate(
        state,
        state.randkey,
        pipeline.algorithm.forward,
        transformed,
    )
    print(f"best genome train-mode fitness: {float(train_fitness):.4f}")
    print(pipeline.algorithm.genome.repr(state, *best))

    eval_problem = EvoJAXSlimeVolleyEnv(
        max_step=args.max_step,
        repeat_times=1,
        test=True,
        action_threshold=args.action_threshold,
        hit_reward_scale=0.0,
        feature_mode=args.feature_mode,
    )
    eval_reward = eval_problem.show(
        state,
        state.randkey,
        pipeline.algorithm.forward,
        transformed,
        gif_path=gif_path,
        gif_fps=args.gif_fps,
        stop_lives_lost=args.eval_stop_lives_lost,
    )
    print(f"best genome test-mode reward: {float(eval_reward):.4f}")

    if float(eval_reward) >= args.save_threshold:
        extra_eval_rewards = []
        for i in range(args.extra_eval_games):
            eval_key = jax.random.fold_in(state.randkey, i + 1)
            reward = eval_problem.evaluate(
                state,
                eval_key,
                pipeline.algorithm.forward,
                transformed,
            )
            extra_eval_rewards.append(float(reward))
        extra_eval_mean = float(np.mean(extra_eval_rewards)) if extra_eval_rewards else float("nan")
        print("eval rewards:", extra_eval_rewards)
        print(
            f"additional test=True evaluation over {args.extra_eval_games} games: "
            f"mean reward {extra_eval_mean:.4f}"
        )
        save_best_genome(
            args.save_path,
            args,
            best,
            eval_reward,
            extra_eval_rewards,
            pipeline,
        )


if __name__ == "__main__":
    main()
