import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import I_INF, fetch_first

from ...gene import BaseGene
from ...utils import extract_gene_attrs, set_gene_attrs
from .base import BaseCrossover


class DefaultCrossover(BaseCrossover):
    def __call__(self, state, genome, randkey, nodes1, conns1, nodes2, conns2):
        randkey1, randkey2 = jax.random.split(randkey, 2)
        node_randkeys = jax.random.split(randkey1, genome.max_nodes)
        conn_randkeys = jax.random.split(randkey2, genome.max_conns)
        batch_create_new_gene = jax.vmap(
            create_new_gene, in_axes=(None, 0, None, 0, 0, None, None)
        )

        node_keys1 = nodes1[:, : len(genome.node_gene.fixed_attrs)]
        node_keys2 = nodes2[:, : len(genome.node_gene.fixed_attrs)]
        node_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.node_gene, nodes1)
        node_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.node_gene, nodes2)
        new_node_attrs = batch_create_new_gene(
            state,
            node_randkeys,
            genome.node_gene,
            node_keys1,
            node_attrs1,
            node_keys2,
            node_attrs2,
        )
        new_nodes = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.node_gene, nodes1, new_node_attrs
        )

        conn_keys1 = conns1[:, : len(genome.conn_gene.fixed_attrs)]
        conn_keys2 = conns2[:, : len(genome.conn_gene.fixed_attrs)]
        conn_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.conn_gene, conns1)
        conn_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(genome.conn_gene, conns2)
        new_conn_attrs = batch_create_new_gene(
            state,
            conn_randkeys,
            genome.conn_gene,
            conn_keys1,
            conn_attrs1,
            conn_keys2,
            conn_attrs2,
        )
        new_conns = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.conn_gene, conns1, new_conn_attrs
        )

        return new_nodes, new_conns


def create_new_gene(state, randkey, gene: BaseGene, gene_key, gene_attrs, genes_keys, genes_attrs):
    homologous_idx = fetch_first(jnp.all(gene_key == genes_keys, axis=1))
    return jax.lax.cond(
        homologous_idx == I_INF,
        lambda: gene_attrs,
        lambda: gene.crossover(state, randkey, gene_attrs, genes_attrs[homologous_idx]),
    )
