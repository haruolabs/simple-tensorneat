import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import I_INF, check_cycles, fetch_random

from ...utils import (
    add_conn,
    add_node,
    delete_conn_by_pos,
    delete_node_by_pos,
    extract_gene_attrs,
    set_gene_attrs,
    unflatten_conns,
)
from .base import BaseMutation


class DefaultMutation(BaseMutation):
    def __init__(
        self,
        conn_add: float = 0.2,
        conn_delete: float = 0.2,
        node_add: float = 0.1,
        node_delete: float = 0.1,
    ):
        self.conn_add = conn_add
        self.conn_delete = conn_delete
        self.node_add = node_add
        self.node_delete = node_delete

    def __call__(self, state, genome, randkey, nodes, conns, new_node_key, new_conn_key):
        k1, k2 = jax.random.split(randkey)
        nodes, conns = self.mutate_structure(
            state, genome, k1, nodes, conns, new_node_key, new_conn_key
        )
        nodes, conns = self.mutate_values(state, genome, k2, nodes, conns)
        return nodes, conns

    def mutate_structure(
        self, state, genome, randkey, nodes, conns, new_node_key, new_conn_key
    ):
        def mutate_add_node(key_, nodes_, conns_):
            remain_node_space = jnp.isnan(nodes_[:, 0]).sum()
            remain_conn_space = jnp.isnan(conns_[:, 0]).sum()
            in_key, out_key, idx = self.choose_connection_key(key_, conns_)

            def successful_add_node():
                original_attrs = extract_gene_attrs(genome.conn_gene, conns_[idx])
                new_conns = delete_conn_by_pos(conns_, idx)
                new_nodes = add_node(
                    nodes_,
                    jnp.array([new_node_key]),
                    genome.node_gene.new_identity_attrs(state),
                )
                fix_attrs1 = jnp.array([in_key, new_node_key])
                fix_attrs2 = jnp.array([new_node_key, out_key])
                new_conns = add_conn(
                    new_conns,
                    fix_attrs1,
                    genome.conn_gene.new_identity_attrs(state),
                )
                new_conns = add_conn(new_conns, fix_attrs2, original_attrs)
                return new_nodes, new_conns

            return jax.lax.cond(
                (idx == I_INF) | (remain_node_space < 1) | (remain_conn_space < 2),
                lambda: (nodes_, conns_),
                successful_add_node,
            )

        def mutate_delete_node(key_, nodes_, conns_):
            node_key, idx = self.choose_node_key(
                key_,
                nodes_,
                genome.input_idx,
                genome.output_idx,
                allow_input_keys=False,
                allow_output_keys=False,
            )

            def successful_delete_node():
                new_nodes = delete_node_by_pos(nodes_, idx)
                new_conns = jnp.where(
                    ((conns_[:, 0] == node_key) | (conns_[:, 1] == node_key))[:, None],
                    jnp.nan,
                    conns_,
                )
                return new_nodes, new_conns

            return jax.lax.cond(
                idx == I_INF,
                lambda: (nodes_, conns_),
                successful_delete_node,
            )

        def mutate_add_conn(key_, nodes_, conns_):
            remain_conn_space = jnp.isnan(conns_[:, 0]).sum()
            k1_, k2_ = jax.random.split(key_, num=2)
            in_key, from_idx = self.choose_node_key(
                k1_,
                nodes_,
                genome.input_idx,
                genome.output_idx,
                allow_input_keys=True,
                allow_output_keys=True,
            )
            out_key, to_idx = self.choose_node_key(
                k2_,
                nodes_,
                genome.input_idx,
                genome.output_idx,
                allow_input_keys=False,
                allow_output_keys=True,
            )
            conn_pos = jnp.argmax((conns_[:, 0] == in_key) & (conns_[:, 1] == out_key))
            exists = (conns_[conn_pos, 0] == in_key) & (conns_[conn_pos, 1] == out_key)

            def successful():
                return nodes_, add_conn(
                    conns_,
                    jnp.array([in_key, out_key]),
                    genome.conn_gene.new_zero_attrs(state),
                )

            u_conns = unflatten_conns(nodes_, conns_)
            conns_exist = u_conns != I_INF
            is_cycle = check_cycles(nodes_, conns_exist, from_idx, to_idx)

            return jax.lax.cond(
                exists | is_cycle | (remain_conn_space < 1),
                lambda: (nodes_, conns_),
                successful,
            )

        def mutate_delete_conn(key_, nodes_, conns_):
            _, _, idx = self.choose_connection_key(key_, conns_)
            return jax.lax.cond(
                idx == I_INF,
                lambda: (nodes_, conns_),
                lambda: (nodes_, delete_conn_by_pos(conns_, idx)),
            )

        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        r1, r2, r3, r4 = jax.random.uniform(k1, shape=(4,))

        def noop(_, nodes_, conns_):
            return nodes_, conns_

        if self.node_add > 0:
            nodes, conns = jax.lax.cond(
                r1 < self.node_add, mutate_add_node, noop, k1, nodes, conns
            )
        if self.node_delete > 0:
            nodes, conns = jax.lax.cond(
                r2 < self.node_delete, mutate_delete_node, noop, k2, nodes, conns
            )
        if self.conn_add > 0:
            nodes, conns = jax.lax.cond(
                r3 < self.conn_add, mutate_add_conn, noop, k3, nodes, conns
            )
        if self.conn_delete > 0:
            nodes, conns = jax.lax.cond(
                r4 < self.conn_delete, mutate_delete_conn, noop, k4, nodes, conns
            )
        return nodes, conns

    def mutate_values(self, state, genome, randkey, nodes, conns):
        k1, k2 = jax.random.split(randkey)
        node_randkeys = jax.random.split(k1, num=genome.max_nodes)
        conn_randkeys = jax.random.split(k2, num=genome.max_conns)

        node_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.node_gene, nodes)
        new_node_attrs = vmap(genome.node_gene.mutate, in_axes=(None, 0, 0))(
            state, node_randkeys, node_attrs
        )
        new_nodes = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.node_gene, nodes, new_node_attrs
        )

        conn_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.conn_gene, conns)
        new_conn_attrs = vmap(genome.conn_gene.mutate, in_axes=(None, 0, 0))(
            state, conn_randkeys, conn_attrs
        )
        new_conns = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.conn_gene, conns, new_conn_attrs
        )

        new_nodes = jnp.where(jnp.isnan(nodes), jnp.nan, new_nodes)
        new_conns = jnp.where(jnp.isnan(conns), jnp.nan, new_conns)
        return new_nodes, new_conns

    def choose_node_key(
        self,
        key,
        nodes,
        input_idx,
        output_idx,
        allow_input_keys=False,
        allow_output_keys=False,
    ):
        node_keys = nodes[:, 0]
        mask = ~jnp.isnan(node_keys)
        if not allow_input_keys:
            mask = jnp.logical_and(mask, ~jnp.isin(node_keys, input_idx))
        if not allow_output_keys:
            mask = jnp.logical_and(mask, ~jnp.isin(node_keys, output_idx))
        idx = fetch_random(key, mask)
        node_key = jnp.where(idx != I_INF, nodes[idx, 0], jnp.nan)
        return node_key, idx

    def choose_connection_key(self, key, conns):
        idx = fetch_random(key, ~jnp.isnan(conns[:, 0]))
        in_key = jnp.where(idx != I_INF, conns[idx, 0], jnp.nan)
        out_key = jnp.where(idx != I_INF, conns[idx, 1], jnp.nan)
        return in_key, out_key, idx
