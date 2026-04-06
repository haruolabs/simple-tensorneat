from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import State
from simpleneat.genome import BaseGenome

from ..base import BaseAlgorithm
from .species import SpeciesController


class NEAT(BaseAlgorithm):
    def __init__(
        self,
        genome: BaseGenome,
        pop_size: int,
        species_size: int = 10,
        max_stagnation: int = 15,
        species_elitism: int = 2,
        spawn_number_change_rate: float = 0.5,
        genome_elitism: int = 2,
        survival_threshold: float = 0.1,
        min_species_size: int = 1,
        compatibility_threshold: float = 2.0,
        species_fitness_func: Callable = jnp.max,
        species_number_calculate_by: str = "rank",
        use_backprop: bool = True,
        backprop_steps: int = 1,
        backprop_learning_rate: float = 0.05,
        backprop_clip_norm: float | None = None,
    ):
        if species_number_calculate_by not in {"rank", "fitness"}: # This shouldn't be assert for user error handling
            raise ValueError("species_number_calculate_by should be 'rank' or 'fitness'")

        self.genome = genome
        self.pop_size = pop_size
        self.use_backprop = use_backprop
        self.backprop_steps = backprop_steps
        self.backprop_learning_rate = backprop_learning_rate
        self.backprop_clip_norm = backprop_clip_norm
        self.species_controller = SpeciesController(
            pop_size=pop_size,
            species_size=species_size,
            max_stagnation=max_stagnation,
            species_elitism=species_elitism,
            spawn_number_change_rate=spawn_number_change_rate,
            genome_elitism=genome_elitism,
            survival_threshold=survival_threshold,
            min_species_size=min_species_size,
            compatibility_threshold=compatibility_threshold,
            species_fitness_func=species_fitness_func,
            species_number_calculate_by=species_number_calculate_by,
        )

    def setup(self, state=State()):
        state = self.genome.setup(state)
        k1, randkey = jax.random.split(state.randkey, 2)
        initialize_keys = jax.random.split(k1, self.pop_size)
        pop_nodes, pop_conns = vmap(self.genome.initialize, in_axes=(None, 0))(
            state, initialize_keys
        )
        state = state.register(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            generation=jnp.float32(0),
            backprop_losses=jnp.full((self.pop_size,), jnp.nan),
        )
        # Initialize species state
        state = self.species_controller.setup(state, pop_nodes[0], pop_conns[0])
        return state.update(randkey=randkey)

    def ask(self, state):
        return state.pop_nodes, state.pop_conns

    def tell(self, state, fitness):
        state = state.update(generation=state.generation + 1)
        state, winner, loser, elite_mask = self.species_controller.update_species(
            state, fitness
        )
        state = self._create_next_generation(state, winner, loser, elite_mask)
        state = self.species_controller.speciate(state, self.genome.execute_distance)
        return state

    def transform(self, state, individual):
        nodes, conns = individual
        return self.genome.transform(state, nodes, conns)

    def forward(self, state, transformed, inputs):
        return self.genome.forward(state, transformed, inputs)

    def adapt(self, state, problem):
        if (
            not self.use_backprop
            or self.backprop_steps <= 0
            or not getattr(problem, "supports_backprop", False)
        ):
            return state

        inputs = problem.backprop_inputs
        targets = problem.backprop_targets
        learning_rate = self.backprop_learning_rate

        def clip_grads(grad):
            if self.backprop_clip_norm is None:
                return grad
            grad = jnp.nan_to_num(grad, nan=0.0)
            norm = jnp.linalg.norm(grad.reshape(-1))
            scale = jnp.minimum(1.0, self.backprop_clip_norm / (norm + 1e-8))
            return grad * scale

        def single_step(nodes, conns):
            loss, (grad_nodes, grad_conns) = self.genome.grad(
                state,
                nodes,
                conns,
                inputs,
                lambda preds: problem.loss_from_outputs(preds, targets),
            )
            grad_nodes = clip_grads(grad_nodes)
            grad_conns = clip_grads(grad_conns)
            return nodes - learning_rate * grad_nodes, conns - learning_rate * grad_conns, loss

        batch_step = vmap(single_step, in_axes=(0, 0))
        pop_nodes = state.pop_nodes
        pop_conns = state.pop_conns
        losses = state.backprop_losses
        for _ in range(self.backprop_steps):
            pop_nodes, pop_conns, losses = batch_step(pop_nodes, pop_conns)
        return state.update(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            backprop_losses=losses,
        )

    @property
    def num_inputs(self):
        return self.genome.num_inputs

    @property
    def num_outputs(self):
        return self.genome.num_outputs

    def _create_next_generation(self, state, winner, loser, elite_mask):
        all_node_keys = state.pop_nodes[:, :, 0]
        max_node_key = jnp.max(all_node_keys, where=~jnp.isnan(all_node_keys), initial=0)
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.pop_size) + next_node_key
        new_conn_markers = jnp.full((self.pop_size, 3), 0)

        k1, k2, randkey = jax.random.split(state.randkey, 3)
        crossover_randkeys = jax.random.split(k1, self.pop_size)
        mutate_randkeys = jax.random.split(k2, self.pop_size)

        winner_nodes = state.pop_nodes[winner]
        winner_conns = state.pop_conns[winner]
        loser_nodes = state.pop_nodes[loser]
        loser_conns = state.pop_conns[loser]

        new_nodes, new_conns = vmap(
            self.genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0)
        )(
            state, crossover_randkeys, winner_nodes, winner_conns, loser_nodes, loser_conns
        )
        mutated_nodes, mutated_conns = vmap(
            self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0, 0)
        )(
            state, mutate_randkeys, new_nodes, new_conns, new_node_keys, new_conn_markers
        )

        pop_nodes = jnp.where(elite_mask[:, None, None], new_nodes, mutated_nodes)
        pop_conns = jnp.where(elite_mask[:, None, None], new_conns, mutated_conns)
        return state.update(randkey=randkey, pop_nodes=pop_nodes, pop_conns=pop_conns)

    def show_details(self, state, fitness):
        member_count = jax.device_get(state.species.member_count)
        species_sizes = [int(v) for v in member_count if v > 0]
        pop_nodes, pop_conns = jax.device_get([state.pop_nodes, state.pop_conns])
        node_counts = (~np.isnan(pop_nodes[:, :, 0])).sum(axis=1)
        conn_counts = (~np.isnan(pop_conns[:, :, 0])).sum(axis=1)
        backprop_losses = jax.device_get(state.backprop_losses)
        if np.isfinite(backprop_losses).any():
            mean_backprop = float(np.nanmean(backprop_losses))
            print(f"\tbackprop mean loss: {mean_backprop:.6f}")
        print(
            f"\tnode counts: max={int(node_counts.max())}, min={int(node_counts.min())}, mean={float(node_counts.mean()):.2f}"
        )
        print(
            f"\tconn counts: max={int(conn_counts.max())}, min={int(conn_counts.min())}, mean={float(conn_counts.mean()):.2f}"
        )
        print(f"\tspecies: {len(species_sizes)} {species_sizes}")
