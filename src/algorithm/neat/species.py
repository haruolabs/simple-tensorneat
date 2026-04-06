from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import (
    State,
    StatefulBaseClass,
    argmin_with_mask,
    fetch_first,
    rank_elements,
)


class SpeciesController(StatefulBaseClass):
    def __init__(
        self,
        pop_size,
        species_size,
        max_stagnation,
        species_elitism,
        spawn_number_change_rate,
        genome_elitism,
        survival_threshold,
        min_species_size,
        compatibility_threshold,
        species_fitness_func,
        species_number_calculate_by,
    ):
        self.pop_size = pop_size
        self.species_size = species_size
        self.species_arange = np.arange(self.species_size)
        self.max_stagnation = max_stagnation
        self.species_elitism = species_elitism
        self.spawn_number_change_rate = spawn_number_change_rate
        self.genome_elitism = genome_elitism
        self.survival_threshold = survival_threshold
        self.min_species_size = min_species_size
        self.compatibility_threshold = compatibility_threshold
        self.species_fitness_func = species_fitness_func
        self.species_number_calculate_by = species_number_calculate_by

    def setup(self, state, first_nodes, first_conns):
        species_state = State(
            species_keys=jnp.full((self.species_size,), jnp.nan).at[0].set(0),
            best_fitness=jnp.full((self.species_size,), jnp.nan).at[0].set(-jnp.inf),
            last_improved=jnp.full((self.species_size,), jnp.nan).at[0].set(0),
            member_count=jnp.full((self.species_size,), jnp.nan).at[0].set(self.pop_size),
            idx2species=jnp.zeros(self.pop_size),
            center_nodes=jnp.full((self.species_size, *first_nodes.shape), jnp.nan).at[0].set(first_nodes),
            center_conns=jnp.full((self.species_size, *first_conns.shape), jnp.nan).at[0].set(first_conns),
            next_species_key=jnp.float32(1),
        )
        return state.register(species=species_state)

    def update_species(self, state, fitness):
        species_state = state.species
        species_fitness = self._update_species_fitness(species_state, fitness)
        species_state, species_fitness = self._stagnation(
            species_state, species_fitness, state.generation
        )
        sort_indices = jnp.argsort(species_fitness)[::-1]
        species_state = species_state.update(
            species_keys=species_state.species_keys[sort_indices],
            best_fitness=species_state.best_fitness[sort_indices],
            last_improved=species_state.last_improved[sort_indices],
            member_count=species_state.member_count[sort_indices],
            center_nodes=species_state.center_nodes[sort_indices],
            center_conns=species_state.center_conns[sort_indices],
        )

        if self.species_number_calculate_by == "rank":
            spawn_number = self._cal_spawn_numbers_by_rank(species_state)
        else:
            spawn_number = self._cal_spawn_numbers_by_fitness(species_state)

        k1, k2 = jax.random.split(state.randkey)
        winner, loser, elite_mask = self._create_crossover_pair(
            species_state, k1, spawn_number, fitness
        )
        return (
            state.update(randkey=k2, species=species_state),
            winner,
            loser,
            elite_mask,
        )

    def _update_species_fitness(self, species_state, fitness):
        def aux_func(idx):
            s_fitness = jnp.where(
                species_state.idx2species == species_state.species_keys[idx],
                fitness,
                -jnp.inf,
            )
            return self.species_fitness_func(s_fitness)

        return vmap(aux_func)(self.species_arange)

    def _stagnation(self, species_state, species_fitness, generation):
        def check_stagnation(idx):
            stagnated = (species_fitness[idx] <= species_state.best_fitness[idx]) & (
                generation - species_state.last_improved[idx] > self.max_stagnation
            )
            last_improved, best_fitness = jax.lax.cond(
                species_fitness[idx] > species_state.best_fitness[idx],
                lambda: (generation, species_fitness[idx]),
                lambda: (
                    species_state.last_improved[idx],
                    species_state.best_fitness[idx],
                ),
            )
            return stagnated, best_fitness, last_improved

        stagnated, best_fitness, last_improved = vmap(check_stagnation)(self.species_arange)
        species_state = species_state.update(
            best_fitness=best_fitness,
            last_improved=last_improved,
        )
        species_rank = rank_elements(species_fitness)
        stagnated = jnp.where(species_rank < self.species_elitism, False, stagnated)

        def update_func(idx):
            return jax.lax.cond(
                stagnated[idx],
                lambda: (
                    jnp.nan,
                    jnp.nan,
                    jnp.nan,
                    jnp.nan,
                    jnp.full_like(species_state.center_nodes[idx], jnp.nan),
                    jnp.full_like(species_state.center_conns[idx], jnp.nan),
                    -jnp.inf,
                ),
                lambda: (
                    species_state.species_keys[idx],
                    species_state.best_fitness[idx],
                    species_state.last_improved[idx],
                    species_state.member_count[idx],
                    species_state.center_nodes[idx],
                    species_state.center_conns[idx],
                    species_fitness[idx],
                ),
            )

        (
            species_keys,
            best_fitness,
            last_improved,
            member_count,
            center_nodes,
            center_conns,
            species_fitness,
        ) = vmap(update_func)(self.species_arange)

        return (
            species_state.update(
                species_keys=species_keys,
                best_fitness=best_fitness,
                last_improved=last_improved,
                member_count=member_count,
                center_nodes=center_nodes,
                center_conns=center_conns,
            ),
            species_fitness,
        )

    def _cal_spawn_numbers_by_rank(self, species_state):
        is_valid = ~jnp.isnan(species_state.species_keys)
        valid_species_num = jnp.sum(is_valid)
        denominator = (valid_species_num + 1) * valid_species_num / 2
        rank_score = valid_species_num - self.species_arange
        target_spawn_number = jnp.floor(rank_score / denominator * self.pop_size)
        spawn_number = species_state.member_count + (
            target_spawn_number - species_state.member_count
        ) * self.spawn_number_change_rate
        spawn_number = jnp.where(
            spawn_number < self.min_species_size, self.min_species_size, spawn_number
        )
        spawn_number = spawn_number.astype(jnp.int32)
        error = self.pop_size - jnp.sum(spawn_number)
        return spawn_number.at[0].add(error)

    def _cal_spawn_numbers_by_fitness(self, species_state):
        species_fitness = species_state.best_fitness
        species_fitness = species_fitness - jnp.nanmin(species_fitness) + 1.0
        spawn_rate = species_fitness / jnp.sum(
            species_fitness, where=~jnp.isnan(species_fitness)
        )
        target_spawn_number = jnp.floor(spawn_rate * self.pop_size)
        spawn_number = species_state.member_count + (
            target_spawn_number - species_state.member_count
        ) * self.spawn_number_change_rate
        spawn_number = jnp.where(
            spawn_number < self.min_species_size, self.min_species_size, spawn_number
        )
        spawn_number = spawn_number.astype(jnp.int32)
        error = self.pop_size - jnp.sum(spawn_number)
        return spawn_number.at[0].add(error)

    def _create_crossover_pair(self, species_state, randkey, spawn_number, fitness):
        species_indices = self.species_arange
        pop_indices = jnp.arange(self.pop_size)

        def choose_parents(key, idx):
            members = species_state.idx2species == species_state.species_keys[idx]
            member_count = jnp.sum(members)
            member_fitness = jnp.where(members, fitness, -jnp.inf)
            sorted_member_indices = jnp.argsort(member_fitness)[::-1]
            survive_size = jnp.maximum(
                jnp.floor(self.survival_threshold * member_count).astype(jnp.int32), 1
            )
            select_pro = (pop_indices < survive_size) / survive_size
            father, mother = jax.random.choice(
                key,
                sorted_member_indices,
                shape=(2, self.pop_size),
                replace=True,
                p=select_pro,
            )
            father = jnp.where(pop_indices < self.genome_elitism, sorted_member_indices, father)
            mother = jnp.where(pop_indices < self.genome_elitism, sorted_member_indices, mother)
            elite = jnp.where(pop_indices < self.genome_elitism, True, False)
            return father, mother, elite

        fathers, mothers, elites = vmap(choose_parents)(
            jax.random.split(randkey, self.species_size), species_indices
        )
        spawn_number_cum = jnp.cumsum(spawn_number)

        def merge_for_population(idx):
            loc = jnp.argmax(idx < spawn_number_cum)
            idx_in_species = jnp.where(loc > 0, idx - spawn_number_cum[loc - 1], idx)
            return (
                fathers[loc, idx_in_species],
                mothers[loc, idx_in_species],
                elites[loc, idx_in_species],
            )

        part1, part2, elite_mask = vmap(merge_for_population)(pop_indices)
        part1_win = fitness[part1] >= fitness[part2]
        winner = jnp.where(part1_win, part1, part2)
        loser = jnp.where(part1_win, part2, part1)
        return winner, loser, elite_mask

    def speciate(self, state, genome_distance_func: Callable):
        o2p_distance_func = vmap(genome_distance_func, in_axes=(None, None, None, 0, 0))
        idx2species = jnp.full((self.pop_size,), jnp.nan)
        o2c_distances = jnp.full((self.pop_size,), jnp.inf)

        def cond_find_centers(carry):
            i, _, _, _, _ = carry
            return (i < self.species_size) & (~jnp.isnan(state.species.species_keys[i]))

        def body_find_centers(carry):
            i, i2s, cns, ccs, o2c = carry
            distances = o2p_distance_func(state, cns[i], ccs[i], state.pop_nodes, state.pop_conns)
            closest_idx = argmin_with_mask(distances, mask=jnp.isnan(i2s))
            i2s = i2s.at[closest_idx].set(state.species.species_keys[i])
            cns = cns.at[i].set(state.pop_nodes[closest_idx])
            ccs = ccs.at[i].set(state.pop_conns[closest_idx])
            o2c = o2c.at[closest_idx].set(0)
            return i + 1, i2s, cns, ccs, o2c

        _, idx2species, center_nodes, center_conns, o2c_distances = jax.lax.while_loop(
            cond_find_centers,
            body_find_centers,
            (
                0,
                idx2species,
                state.species.center_nodes,
                state.species.center_conns,
                o2c_distances,
            ),
        )

        state = state.update(
            species=state.species.update(
                idx2species=idx2species,
                center_nodes=center_nodes,
                center_conns=center_conns,
            ),
        )

        def speciate_by_threshold(i, i2s, cns, ccs, sk, o2c):
            o2p_distance = o2p_distance_func(state, cns[i], ccs[i], state.pop_nodes, state.pop_conns)
            close_enough = o2p_distance < self.compatibility_threshold
            catchable = jnp.isnan(i2s) | (o2p_distance < o2c)
            mask = close_enough & catchable
            i2s = jnp.where(mask, sk[i], i2s)
            o2c = jnp.where(mask, o2p_distance, o2c)
            return i2s, o2c

        def create_new_species(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry
            idx = fetch_first(jnp.isnan(i2s))
            sk = sk.at[i].set(nsk)
            i2s = i2s.at[idx].set(nsk)
            o2c = o2c.at[idx].set(0)
            cns = cns.at[i].set(state.pop_nodes[idx])
            ccs = ccs.at[i].set(state.pop_conns[idx])
            i2s, o2c = speciate_by_threshold(i, i2s, cns, ccs, sk, o2c)
            return i, i2s, cns, ccs, sk, o2c, nsk + 1

        def update_existing_species(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry
            i2s, o2c = speciate_by_threshold(i, i2s, cns, ccs, sk, o2c)
            return i + 1, i2s, cns, ccs, sk, o2c, nsk

        def cond_assign(carry):
            i, i2s, _, _, sk, _, _ = carry
            current_exists = ~jnp.isnan(sk[i])
            not_all_assigned = jnp.any(jnp.isnan(i2s))
            within_bounds = i < self.species_size
            return within_bounds & (current_exists | not_all_assigned)

        def body_assign(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry
            _, i2s, cns, ccs, sk, o2c, nsk = jax.lax.cond(
                jnp.isnan(sk[i]),
                create_new_species,
                update_existing_species,
                (i, i2s, cns, ccs, sk, o2c, nsk),
            )
            return i + 1, i2s, cns, ccs, sk, o2c, nsk

        (
            _,
            idx2species,
            center_nodes,
            center_conns,
            species_keys,
            _,
            next_species_key,
        ) = jax.lax.while_loop(
            cond_assign,
            body_assign,
            (
                0,
                state.species.idx2species,
                center_nodes,
                center_conns,
                state.species.species_keys,
                o2c_distances,
                state.species.next_species_key,
            ),
        )

        idx2species = jnp.where(jnp.isnan(idx2species), species_keys[-1], idx2species)
        new_created_mask = (~jnp.isnan(species_keys)) & jnp.isnan(state.species.best_fitness)
        best_fitness = jnp.where(new_created_mask, -jnp.inf, state.species.best_fitness)
        last_improved = jnp.where(
            new_created_mask, state.generation, state.species.last_improved
        )

        def count_members(idx):
            return jax.lax.cond(
                jnp.isnan(species_keys[idx]),
                lambda: jnp.nan,
                lambda: jnp.sum(idx2species == species_keys[idx], dtype=jnp.float32),
            )

        member_count = vmap(count_members)(self.species_arange)
        species_state = state.species.update(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=next_species_key,
        )
        return state.update(species=species_state)
