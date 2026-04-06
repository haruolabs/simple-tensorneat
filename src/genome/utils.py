import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import I_INF, fetch_first

from .gene import BaseGene


def unflatten_conns(nodes, conns):
    node_keys = nodes[:, 0]
    input_keys = conns[:, 0]
    output_keys = conns[:, 1]

    def key_to_indices(key, keys):
        return fetch_first(key == keys)

    input_indices = vmap(key_to_indices, in_axes=(0, None))(input_keys, node_keys)
    output_indices = vmap(key_to_indices, in_axes=(0, None))(output_keys, node_keys)

    return (
        jnp.full((nodes.shape[0], nodes.shape[0]), I_INF, dtype=jnp.int32)
        .at[input_indices, output_indices]
        .set(jnp.arange(conns.shape[0], dtype=jnp.int32))
    )


def valid_cnt(nodes_or_conns):
    return jnp.sum(~jnp.isnan(nodes_or_conns[:, 0]))


def extract_gene_attrs(gene: BaseGene, gene_array):
    return gene_array[len(gene.fixed_attrs) :]


def set_gene_attrs(gene: BaseGene, gene_array, attrs):
    return gene_array.at[len(gene.fixed_attrs) :].set(attrs)


def add_node(nodes, fix_attrs, custom_attrs):
    pos = fetch_first(jnp.isnan(nodes[:, 0]))
    return nodes.at[pos].set(jnp.concatenate((fix_attrs, custom_attrs)))


def delete_node_by_pos(nodes, pos):
    return nodes.at[pos].set(jnp.nan)


def add_conn(conns, fix_attrs, custom_attrs):
    pos = fetch_first(jnp.isnan(conns[:, 0]))
    return conns.at[pos].set(jnp.concatenate((fix_attrs, custom_attrs)))


def delete_conn_by_pos(conns, pos):
    return conns.at[pos].set(jnp.nan)


def re_cound_idx(nodes, conns, input_idx, output_idx):
    nodes, conns = jax.device_get((nodes, conns))
    next_key = max(*input_idx, *output_idx) + 1
    old2new = {}

    for key in nodes[:, 0]:
        if np.isnan(key):
            continue
        if int(key) in input_idx + output_idx:
            continue
        old2new[int(key)] = next_key
        next_key += 1

    new_nodes = nodes.copy()
    for idx, key in enumerate(nodes[:, 0]):
        if not np.isnan(key) and int(key) in old2new:
            new_nodes[idx, 0] = old2new[int(key)]

    new_conns = conns.copy()
    for idx, (in_key, out_key) in enumerate(conns[:, :2]):
        if not np.isnan(in_key) and int(in_key) in old2new:
            new_conns[idx, 0] = old2new[int(in_key)]
        if not np.isnan(out_key) and int(out_key) in old2new:
            new_conns[idx, 1] = old2new[int(out_key)]

    return new_nodes, new_conns
