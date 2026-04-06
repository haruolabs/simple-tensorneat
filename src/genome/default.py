import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import (
    ACT,
    AGG,
    I_INF,
    attach_with_inf,
    find_useful_nodes,
    topological_sort,
    topological_sort_python,
)

from .base import BaseGenome
from .gene import DefaultConn, DefaultNode
from .operations import DefaultCrossover, DefaultDistance, DefaultMutation
from .utils import extract_gene_attrs, unflatten_conns


class DefaultGenome(BaseGenome):
    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
    ):
        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=node_gene,
            conn_gene=conn_gene,
            mutation=mutation,
            crossover=crossover,
            distance=distance,
            output_transform=output_transform,
            input_transform=input_transform,
            init_hidden_layers=init_hidden_layers,
        )

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        conn_exist = u_conns != I_INF
        seqs = topological_sort(nodes, conn_exist)
        return seqs, nodes, conns, u_conns

    def forward(self, state, transformed, inputs):
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        cal_seqs, nodes, conns, u_conns = transformed
        values = jnp.full((self.max_nodes,), jnp.nan).at[self.input_idx].set(inputs)
        node_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.node_gene, nodes)
        conn_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.conn_gene, conns)

        def body_func(idx, values_):
            node_pos = cal_seqs[idx]

            def valid_node():
                def input_node():
                    return values_

                def compute_node():
                    conn_indices = u_conns[:, node_pos]
                    conn_exists = conn_indices != I_INF
                    src_valid = ~jnp.isnan(values_)
                    valid_mask = conn_exists & src_valid
                    hit_attrs = attach_with_inf(conn_attrs, conn_indices)
                    weighted_inputs = vmap(self.conn_gene.forward, in_axes=(None, 0, 0))(
                        state, hit_attrs, values_
                    )
                    z = self.node_gene.forward(
                        state,
                        node_attrs[node_pos],
                        weighted_inputs,
                        is_output_node=jnp.isin(nodes[node_pos, 0], self.output_idx),
                        valid_mask=valid_mask,
                    )
                    return jax.lax.cond(
                        jnp.any(valid_mask),
                        lambda: values_.at[node_pos].set(z),
                        lambda: values_,
                    )

                return jax.lax.cond(jnp.isin(node_pos, self.input_idx), input_node, compute_node)

            return jax.lax.cond(node_pos != I_INF, valid_node, lambda: values_)

        values = jax.lax.fori_loop(0, self.max_nodes, body_func, values)
        outputs = values[self.output_idx]
        return outputs if self.output_transform is None else self.output_transform(outputs)

    def network_dict(self, state, nodes, conns):
        network = super().network_dict(state, nodes, conns)
        topo_order, topo_layers = topological_sort_python(
            set(network["nodes"]), set(network["conns"])
        )
        network["topo_order"] = topo_order
        network["topo_layers"] = topo_layers
        network["useful_nodes"] = find_useful_nodes(
            set(network["nodes"]),
            set(network["conns"]),
            set(self.output_idx),
        )
        return network
