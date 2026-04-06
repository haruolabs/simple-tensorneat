import argparse

from simpleneat.algorithm.neat import NEAT
from simpleneat.common import ACT, AGG
from simpleneat.genome import BiasNode, DefaultGenome
from simpleneat.genome.operations import DefaultMutation
from simpleneat.pipeline import Pipeline
from simpleneat.problem.rl import SlimeVolleyEnv


def build_parser():
    parser = argparse.ArgumentParser(description="Evolve a NEAT policy for SlimeVolley.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=256) # 32
    parser.add_argument("--species-size", type=int, default=24) # 8
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--max-step", type=int, default=1000)
    parser.add_argument("--fitness-target", type=float, default=3.0)
    parser.add_argument(
        "--log-path",
        type=str,
        default="examples/rl/slimevolley_history.csv",
        help="CSV path for generation statistics. Use an empty string to disable logging.",
    )
    parser.add_argument(
        "--render-best",
        action="store_true",
        help="Render one rollout for the best genome after evolution finishes.",
    )
    parser.add_argument(
        "--gif-path",
        type=str,
        default=None,
        help="Optional output path for a GIF of the best genome replay.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=30,
        help="Frame rate for GIF export when --gif-path is set.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Delay between rendered frames when --render-best is enabled.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    log_path = args.log_path or None
    max_env_steps_per_generation = args.pop_size * args.repeat_times * args.max_step

    print(
        "Starting SlimeVolley evolution with "
        f"pop_size={args.pop_size}, species_size={args.species_size}, "
        f"generations={args.generations}, repeat_times={args.repeat_times}, "
        f"max_step={args.max_step}",
        flush=True,
    )
    print(
        "This RL problem is evaluated sequentially, so each generation may use up to "
        f"{max_env_steps_per_generation} environment steps.",
        flush=True,
    )
    if max_env_steps_per_generation > 100_000:
        print(
            "This workload is large. If the first generation feels too slow, reduce "
            "`--pop-size`, `--repeat-times`, or `--max-step`.",
            flush=True,
        )

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=args.pop_size,
            species_size=args.species_size,
            survival_threshold=0.2,
            compatibility_threshold=1.5,
            genome=DefaultGenome(
                num_inputs=12,
                num_outputs=3,
                max_nodes=32,
                max_conns=96,
                node_gene=BiasNode(
                    activation_options=[ACT.tanh, ACT.identity], # ACT.relu],
                    aggregation_options=AGG.sum,
                ),
                mutation=DefaultMutation(
                    conn_add=0.3,
                    conn_delete=0.08,
                    node_add=0.12, # 0.08
                    node_delete=0.03,
                ),
                output_transform=ACT.identity #ACT.tanh,
            ),
            use_backprop=False,
        ),
        problem=SlimeVolleyEnv(
            max_step=args.max_step,
            repeat_times=args.repeat_times,
        ),
        generation_limit=args.generations,
        fitness_target=args.fitness_target,
        seed=args.seed,
        log_path=log_path,
    )

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)

    if best is None:
        raise RuntimeError("Evolution finished without producing a valid best genome.")

    final_fitness = pipeline.problem.evaluate(
        state,
        state.randkey,
        pipeline.algorithm.forward,
        pipeline.algorithm.transform(state, best),
    )
    print(f"best genome reevaluated fitness: {final_fitness:.4f}")
    print(pipeline.algorithm.genome.repr(state, *best))

    if args.render_best or args.gif_path is not None:
        pipeline.show(
            state,
            best,
            render=args.render_best,
            sleep=args.sleep,
            gif_path=args.gif_path,
            gif_fps=args.gif_fps,
        )


if __name__ == "__main__":
    main()
