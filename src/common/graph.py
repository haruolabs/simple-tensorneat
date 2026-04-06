import jax
from jax import Array, jit, numpy as jnp

from .tools import I_INF, fetch_first


@jit
def topological_sort(nodes: Array, conns: Array) -> Array:
    in_degree = jnp.where(jnp.isnan(nodes[:, 0]), jnp.nan, jnp.sum(conns, axis=0))
    result = jnp.full(in_degree.shape, I_INF)

    def cond_fun(carry):
        _, _, in_degree_ = carry
        idx = fetch_first(in_degree_ == 0.0)
        return idx != I_INF

    def body_fun(carry):
        result_, write_idx, in_degree_ = carry
        idx = fetch_first(in_degree_ == 0.0)
        result_ = result_.at[write_idx].set(idx)
        in_degree_ = in_degree_.at[idx].set(-1)
        children = conns[idx, :]
        in_degree_ = jnp.where(children, in_degree_ - 1, in_degree_)
        return result_, write_idx + 1, in_degree_

    result, _, _ = jax.lax.while_loop(cond_fun, body_fun, (result, 0, in_degree))
    return result


def topological_sort_python(nodes, conns):
    nodes = nodes.copy()
    conns = conns.copy()
    in_degree = {node: 0 for node in nodes}
    for source, target in conns:
        in_degree[target] += 1

    topo_order = []
    topo_layers = []
    zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]

    while zero_in_degree_nodes:
        for node in zero_in_degree_nodes:
            nodes.remove(node)

        zero_in_degree_nodes = sorted(zero_in_degree_nodes)
        topo_layers.append(zero_in_degree_nodes.copy())

        for node in zero_in_degree_nodes:
            topo_order.append(node)
            for edge in list(conns):
                if edge[0] == node:
                    in_degree[edge[1]] -= 1
                    conns.remove(edge)

        zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]

    if conns or nodes:
        raise ValueError("Graph has at least one cycle")

    return topo_order, topo_layers


def find_useful_nodes(nodes, conns, output_idx):
    useful_nodes = set(output_idx)
    while True:
        new_nodes = set()
        for in_idx, out_idx in conns:
            if out_idx in useful_nodes and in_idx not in useful_nodes:
                new_nodes.add(in_idx)
        if not new_nodes:
            return useful_nodes
        useful_nodes |= new_nodes


@jit
def check_cycles(nodes: Array, conns: Array, from_idx, to_idx) -> Array:
    conns = conns.at[from_idx, to_idx].set(True)
    visited = jnp.full(nodes.shape[0], False)
    frontier = visited.at[to_idx].set(True)

    def cond_fun(carry):
        visited_, frontier_ = carry
        no_progress = jnp.all(visited_ == frontier_)
        reached_source = frontier_[from_idx]
        return jnp.logical_not(no_progress | reached_source)

    def body_fun(carry):
        _, frontier_ = carry
        next_frontier = jnp.dot(frontier_, conns)
        next_frontier = jnp.logical_or(frontier_, next_frontier)
        return frontier_, next_frontier

    _, visited = jax.lax.while_loop(cond_fun, body_fun, (visited, frontier))
    return visited[from_idx]

