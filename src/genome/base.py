from typing import Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import State, StatefulBaseClass, hash_array

from .gene import BaseConn, BaseNode
from .operations import BaseCrossover, BaseDistance, BaseMutation
from .utils import re_cound_idx, valid_cnt


class BaseGenome(StatefulBaseClass):
    network_type = None

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int,
        max_conns: int,
        node_gene: BaseNode,
        conn_gene: BaseConn,
        mutation: BaseMutation,
        crossover: BaseCrossover,
        distance: BaseDistance,
        output_transform: Callable = None,
        input_transform: Callable = None,
        init_hidden_layers: Sequence[int] = (),
    ):
        if input_transform is not None:
            input_transform(jnp.zeros(num_inputs))
        if output_transform is not None:
            output_transform(jnp.zeros(num_outputs))

        all_layers = [num_inputs] + list(init_hidden_layers) + [num_outputs]
        layer_indices = []
        next_index = 0
        for layer_size in all_layers:
            layer_indices.append(list(range(next_index, next_index + layer_size)))
            next_index += layer_size

        init_nodes = []
        init_conns_in = []
        init_conns_out = []
        for i in range(len(layer_indices) - 1):
            in_layer = layer_indices[i]
            out_layer = layer_indices[i + 1]
            for in_idx in in_layer:
                for out_idx in out_layer:
                    init_conns_in.append(in_idx)
                    init_conns_out.append(out_idx)
            init_nodes.extend(in_layer)
        init_nodes.extend(layer_indices[-1])

        if max_nodes < len(init_nodes):
            raise ValueError("max_nodes is smaller than the initial node count")
        if max_conns < len(init_conns_in):
            raise ValueError("max_conns is smaller than the initial connection count")

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.node_gene = node_gene
        self.conn_gene = conn_gene
        self.mutation = mutation
        self.crossover = crossover
        self.distance = distance
        self.output_transform = output_transform
        self.input_transform = input_transform

        self.input_idx = np.array(layer_indices[0])
        self.output_idx = np.array(layer_indices[-1])
        self.all_init_nodes = np.array(init_nodes)
        self.all_init_conns = np.c_[init_conns_in, init_conns_out]

    def setup(self, state=State()):
        state = self.node_gene.setup(state)
        state = self.conn_gene.setup(state)
        state = self.mutation.setup(state)
        state = self.crossover.setup(state)
        state = self.distance.setup(state)
        return state

    def transform(self, state, nodes, conns):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def grad(self, state, nodes, conns, inputs, loss_fn):
        def inner(nodes_, conns_):
            transformed = self.transform(state, nodes_, conns_)
            if inputs.ndim == 1:
                outputs = self.forward(state, transformed, inputs)
            else:
                outputs = jax.vmap(self.forward, in_axes=(None, None, 0))(
                    state, transformed, inputs
                )
            return loss_fn(outputs)

        loss, (grad_nodes, grad_conns) = jax.value_and_grad(inner, argnums=(0, 1))(
            nodes, conns
        )
        grad_nodes = jnp.where(jnp.isnan(nodes), 0.0, grad_nodes)
        grad_conns = jnp.where(jnp.isnan(conns), 0.0, grad_conns)
        grad_nodes = jnp.where(self.node_gene.gradient_mask[None, :], grad_nodes, 0.0)
        grad_conns = jnp.where(self.conn_gene.gradient_mask[None, :], grad_conns, 0.0)
        return loss, (grad_nodes, grad_conns)

    def execute_mutation(self, state, randkey, nodes, conns, new_node_key, new_conn_keys):
        return self.mutation(
            state, self, randkey, nodes, conns, new_node_key, new_conn_keys
        )

    def execute_crossover(self, state, randkey, nodes1, conns1, nodes2, conns2):
        return self.crossover(state, self, randkey, nodes1, conns1, nodes2, conns2)

    def execute_distance(self, state, nodes1, conns1, nodes2, conns2):
        return self.distance(state, self, nodes1, conns1, nodes2, conns2)

    def initialize(self, state, randkey):
        k1, k2 = jax.random.split(randkey)
        node_count = len(self.all_init_nodes)
        conn_count = len(self.all_init_conns)

        nodes = jnp.full((self.max_nodes, self.node_gene.length), jnp.nan)
        node_indices = self.all_init_nodes
        node_randkeys = jax.random.split(k1, num=node_count)
        node_attr_func = vmap(self.node_gene.new_random_attrs, in_axes=(None, 0))
        node_attrs = node_attr_func(state, node_randkeys)
        nodes = nodes.at[:node_count, 0].set(node_indices)
        nodes = nodes.at[:node_count, 1:].set(node_attrs)

        conns = jnp.full((self.max_conns, self.conn_gene.length), jnp.nan)
        conn_randkeys = jax.random.split(k2, num=conn_count)
        conn_attrs = vmap(self.conn_gene.new_random_attrs, in_axes=(None, 0))(
            state, conn_randkeys
        )
        conns = conns.at[:conn_count, :2].set(self.all_init_conns)
        conns = conns.at[:conn_count, len(self.conn_gene.fixed_attrs) :].set(conn_attrs)
        return nodes, conns

    def get_input_idx(self):
        return self.input_idx.tolist()

    def get_output_idx(self):
        return self.output_idx.tolist()

    def hash(self, nodes, conns):
        return hash_array(
            jnp.concatenate([vmap(hash_array)(nodes), vmap(hash_array)(conns)])
        )

    def repr(self, state, nodes, conns, precision=2):
        nodes, conns = jax.device_get([nodes, conns])
        node_count = valid_cnt(nodes)
        conn_count = valid_cnt(conns)
        lines = [f"{self.__class__.__name__}(nodes={node_count}, conns={conn_count}):"]
        lines.append("\tNodes:")
        for node in nodes:
            if np.isnan(node[0]):
                break
            text = self.node_gene.repr(state, node, precision=precision)
            node_idx = int(node[0])
            if np.isin(node_idx, self.input_idx):
                text += " (input)"
            elif np.isin(node_idx, self.output_idx):
                text += " (output)"
            lines.append(f"\t\t{text}")
        lines.append("\tConns:")
        for conn in conns:
            if np.isnan(conn[0]):
                break
            lines.append(f"\t\t{self.conn_gene.repr(state, conn, precision=precision)}")
        return "\n".join(lines)

    def network_dict(self, state, nodes, conns, whether_re_cound_idx=True):
        if whether_re_cound_idx:
            nodes, conns = re_cound_idx(
                nodes, conns, self.get_input_idx(), self.get_output_idx()
            )
        return {
            "nodes": self._get_node_dict(state, nodes),
            "conns": self._get_conn_dict(state, conns),
        }

    def _get_conn_dict(self, state, conns):
        conns = jax.device_get(conns)
        conn_dict = {}
        for conn in conns:
            if np.isnan(conn[0]):
                continue
            cd = self.conn_gene.to_dict(state, conn)
            conn_dict[(cd["in"], cd["out"])] = cd
        return conn_dict

    def _get_node_dict(self, state, nodes):
        nodes = jax.device_get(nodes)
        node_dict = {}
        for node in nodes:
            if np.isnan(node[0]):
                continue
            nd = self.node_gene.to_dict(state, node)
            node_dict[nd["idx"]] = nd
        return node_dict
