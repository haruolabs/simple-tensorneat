import jax.numpy as jnp
from jax import vmap

from ...gene import BaseGene
from ...utils import extract_gene_attrs
from .base import BaseDistance


class DefaultDistance(BaseDistance):
    def __init__(
        self,
        compatibility_disjoint: float = 1.0,
        compatibility_weight: float = 0.4,
    ):
        self.compatibility_disjoint = compatibility_disjoint
        self.compatibility_weight = compatibility_weight

    def __call__(self, state, genome, nodes1, conns1, nodes2, conns2):
        node_distance = self.gene_distance(state, genome.node_gene, nodes1, nodes2)
        conn_distance = self.gene_distance(state, genome.conn_gene, conns1, conns2)
        return node_distance + conn_distance

    def gene_distance(self, state, gene: BaseGene, genes1, genes2):
        cnt1 = jnp.sum(~jnp.isnan(genes1[:, 0]))
        cnt2 = jnp.sum(~jnp.isnan(genes2[:, 0]))
        max_cnt = jnp.maximum(cnt1, cnt2)

        total_genes = jnp.concatenate((genes1, genes2), axis=0)
        identifiers = total_genes[:, : len(gene.fixed_attrs)]
        sorted_indices = jnp.lexsort(identifiers.T[::-1])
        total_genes = total_genes[sorted_indices]
        total_genes = jnp.concatenate(
            [total_genes, jnp.full((1, total_genes.shape[1]), jnp.nan)],
            axis=0,
        )
        first_row, second_row = total_genes[:-1], total_genes[1:]
        intersect_mask = jnp.all(
            first_row[:, : len(gene.fixed_attrs)] == second_row[:, : len(gene.fixed_attrs)],
            axis=1,
        ) & ~jnp.isnan(first_row[:, 0])

        non_homologous_cnt = cnt1 + cnt2 - 2 * jnp.sum(intersect_mask)
        first_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(gene, first_row)
        second_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(gene, second_row)
        homologous_distance = vmap(gene.distance, in_axes=(None, 0, 0))(
            state, first_attrs, second_attrs
        )
        homologous_distance = jnp.where(jnp.isnan(homologous_distance), 0.0, homologous_distance)
        homologous_distance = jnp.sum(homologous_distance * intersect_mask)

        total = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )
        return jnp.where(max_cnt == 0, 0.0, total / max_cnt)

