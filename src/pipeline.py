import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simpleneat.algorithm import BaseAlgorithm
from simpleneat.common import State, StatefulBaseClass
from simpleneat.problem import BaseProblem


class Pipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        problem: BaseProblem,
        seed: int = 42,
        fitness_target: float = 1.0,
        generation_limit: int = 1000,
        log_path: str | None = None,
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.seed = seed
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.log_path = Path(log_path) if log_path is not None else None
        self.pop_size = self.algorithm.pop_size

        if algorithm.num_inputs != self.problem.input_shape[-1]:
            raise ValueError(
                f"algorithm input size {algorithm.num_inputs} does not match problem input shape {self.problem.input_shape}"
            )
        if algorithm.num_outputs != self.problem.output_shape[-1]:
            raise ValueError(
                f"algorithm output size {algorithm.num_outputs} does not match problem output shape {self.problem.output_shape}"
            )

        self.best_genome = None
        self.best_fitness = float("-inf")

    def setup(self, state=State()):
        self.best_genome = None
        self.best_fitness = float("-inf")
        self._initialize_log_file()
        state = state.register(randkey=jax.random.PRNGKey(self.seed))
        state = self.algorithm.setup(state)
        state = self.problem.setup(state)
        return state

    def step(self, state):
        eval_key, next_key = jax.random.split(state.randkey)
        state = self.algorithm.adapt(state, self.problem)
        population = self.algorithm.ask(state)
        transformed = self._transform_population(state, population)
        fitnesses = self._evaluate_population(state, eval_key, transformed)
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        state = self.algorithm.tell(state, fitnesses)
        return state.update(randkey=next_key), population, fitnesses

    def auto_run(self, state):
        compiled_step = None
        if self.problem.jitable:
            print("start compile", flush=True)
            tic = time.time()
            compiled_step = jax.jit(self.step).lower(state).compile()
            print(f"compile finished, cost time: {time.time() - tic:.6f}s", flush=True)

        # Evolution loop
        for _ in range(self.generation_limit):
            tic = time.time()
            if compiled_step is None:
                state, previous_pop, fitnesses = self.step(state)
            else:
                state, previous_pop, fitnesses = compiled_step(state)
            cpu_fitnesses = np.asarray(jax.device_get(fitnesses))
            self.analysis(state, previous_pop, cpu_fitnesses, time.time() - tic)
            if np.max(cpu_fitnesses) >= self.fitness_target:
                print("Fitness limit reached!", flush=True)
                break
        if int(state.generation) >= self.generation_limit:
            print("Generation limit reached!", flush=True)
        return state, self.best_genome

    def analysis(self, state, pop, fitnesses, cost_time):
        generation = int(state.generation)
        valid_fitnesses = fitnesses[np.isfinite(fitnesses)]
        if valid_fitnesses.size == 0:
            max_f = min_f = mean_f = std_f = float("nan")
        else:
            max_f = float(np.max(valid_fitnesses))
            min_f = float(np.min(valid_fitnesses))
            mean_f = float(np.mean(valid_fitnesses))
            std_f = float(np.std(valid_fitnesses))

        max_idx = int(np.argmax(fitnesses))
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = float(fitnesses[max_idx])
            self.best_genome = (pop[0][max_idx], pop[1][max_idx])

        print(
            f"Generation: {generation}, Cost time: {cost_time * 1000:.2f}ms, "
            f"fitness max/min/mean/std: {max_f:.4f}/{min_f:.4f}/{mean_f:.4f}/{std_f:.4f}",
            flush=True,
        )
        self._log_generation(
            generation=generation,
            cost_time_ms=cost_time * 1000.0,
            fitness_max=max_f,
            fitness_min=min_f,
            fitness_mean=mean_f,
            fitness_std=std_f,
        )
        self.algorithm.show_details(state, fitnesses)

    def show(self, state, best, *args, **kwargs):
        transformed = self.algorithm.transform(state, best)
        return self.problem.show(
            state, state.randkey, self.algorithm.forward, transformed, *args, **kwargs
        )

    def _transform_population(self, state, population):
        if self.problem.jitable:
            return jax.vmap(self.algorithm.transform, in_axes=(None, 0))(state, population)

        pop_nodes, pop_conns = population
        return [
            self.algorithm.transform(state, (nodes, conns))
            for nodes, conns in zip(pop_nodes, pop_conns)
        ]

    def _evaluate_population(self, state, eval_key, transformed):
        # if jittable, return eval results with vmap
        if self.problem.jitable:
            keys = jax.random.split(eval_key, self.pop_size)
            return jax.vmap(self.problem.evaluate, in_axes=(None, 0, None, 0))(
                state, keys, self.algorithm.forward, transformed
            )

        keys = jax.random.split(eval_key, self.pop_size)
        return jnp.asarray(
            [
                self.problem.evaluate(state, key, self.algorithm.forward, params)
                for key, params in zip(keys, transformed)
            ],
            dtype=jnp.float32,
        )

    def _initialize_log_file(self):
        if self.log_path is None:
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "generation",
                    "cost_time_ms",
                    "fitness_max",
                    "fitness_min",
                    "fitness_mean",
                    "fitness_std",
                ]
            )

    def _log_generation(
        self,
        generation: int,
        cost_time_ms: float,
        fitness_max: float,
        fitness_min: float,
        fitness_mean: float,
        fitness_std: float,
    ):
        if self.log_path is None:
            return

        with self.log_path.open("a", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    generation,
                    f"{cost_time_ms:.6f}",
                    f"{fitness_max:.6f}",
                    f"{fitness_min:.6f}",
                    f"{fitness_mean:.6f}",
                    f"{fitness_std:.6f}",
                ]
            )
