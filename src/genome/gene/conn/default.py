import jax
import jax.numpy as jnp

from simpleneat.common import mutate_float

from .base import BaseConn


class DefaultConn(BaseConn):
    custom_attrs = ["weight"]
    trainable_custom_attrs = [True]

    def __init__(
        self,
        weight_init_mean: float = 0.0,
        weight_init_std: float = 1.0,
        weight_mutate_power: float = 0.15,
        weight_mutate_rate: float = 0.2,
        weight_replace_rate: float = 0.015,
        weight_lower_bound: float = -5.0,
        weight_upper_bound: float = 5.0,
    ):
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate
        self.weight_lower_bound = weight_lower_bound
        self.weight_upper_bound = weight_upper_bound

    def new_zero_attrs(self, state):
        return jnp.array([0.0])

    def new_identity_attrs(self, state):
        return jnp.array([1.0])

    def new_random_attrs(self, state, randkey):
        weight = (
            jax.random.normal(randkey, ()) * self.weight_init_std
            + self.weight_init_mean
        )
        weight = jnp.clip(weight, self.weight_lower_bound, self.weight_upper_bound)
        return jnp.array([weight])

    def mutate(self, state, randkey, attrs):
        weight = mutate_float(
            randkey,
            attrs[0],
            self.weight_init_mean,
            self.weight_init_std,
            self.weight_mutate_power,
            self.weight_mutate_rate,
            self.weight_replace_rate,
        )
        weight = jnp.clip(weight, self.weight_lower_bound, self.weight_upper_bound)
        return jnp.array([weight])

    def distance(self, state, attrs1, attrs2):
        return jnp.abs(attrs1[0] - attrs2[0])

    def forward(self, state, attrs, inputs):
        weight = jnp.where(jnp.isnan(attrs[0]), 0.0, attrs[0])
        safe_inputs = jnp.where(jnp.isnan(inputs), 0.0, inputs)
        return safe_inputs * weight

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx, weight = conn
        return (
            f"{self.__class__.__name__}(in={int(in_idx):<{idx_width}}, "
            f"out={int(out_idx):<{idx_width}}, "
            f"weight={round(float(weight), precision):<{precision + 3}})"
        )

    def to_dict(self, state, conn):
        return {
            "in": int(conn[0]),
            "out": int(conn[1]),
            "weight": jnp.float32(conn[2]),
        }
