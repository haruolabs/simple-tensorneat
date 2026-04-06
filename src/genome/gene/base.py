import jax
import jax.numpy as jnp

from simpleneat.common import State, StatefulBaseClass, hash_array


class BaseGene(StatefulBaseClass):
    fixed_attrs = []
    custom_attrs = []
    trainable_custom_attrs = []

    def new_identity_attrs(self, state):
        raise NotImplementedError

    def new_random_attrs(self, state, randkey):
        raise NotImplementedError

    def mutate(self, state, randkey, attrs):
        raise NotImplementedError

    def crossover(self, state, randkey, attrs1, attrs2):
        return jnp.where(
            jax.random.normal(randkey, attrs1.shape) > 0,
            attrs1,
            attrs2,
        )

    def distance(self, state, attrs1, attrs2):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    @property
    def length(self):
        return len(self.fixed_attrs) + len(self.custom_attrs)

    @property
    def gradient_mask(self):
        if not self.trainable_custom_attrs:
            custom_mask = [True] * len(self.custom_attrs)
        else:
            custom_mask = list(self.trainable_custom_attrs)
        return jnp.asarray(
            [False] * len(self.fixed_attrs) + custom_mask,
            dtype=jnp.bool_,
        )

    def repr(self, state, gene, precision=2):
        raise NotImplementedError

    def hash(self, gene):
        return hash_array(gene)
