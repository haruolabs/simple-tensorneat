from typing import Callable, Optional, Sequence, Union

import numpy as np
import jax
import jax.numpy as jnp

from simpleneat.common import (
    ACT,
    AGG,
    apply_activation,
    apply_aggregation,
    get_func_name,
    mutate_float,
    mutate_int,
)

from .base import BaseNode


class BiasNode(BaseNode):
    custom_attrs = ["bias", "aggregation", "activation"]
    trainable_custom_attrs = [True, False, False]

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.15,
        bias_mutate_rate: float = 0.2,
        bias_replace_rate: float = 0.015,
        bias_lower_bound: float = -5.0,
        bias_upper_bound: float = 5.0,
        aggregation_default: Optional[Callable] = None,
        aggregation_options: Union[Callable, Sequence[Callable]] = AGG.sum,
        aggregation_replace_rate: float = 0.1,
        activation_default: Optional[Callable] = None,
        activation_options: Union[Callable, Sequence[Callable]] = ACT.sigmoid,
        activation_replace_rate: float = 0.1,
    ):
        if isinstance(aggregation_options, Callable):
            aggregation_options = [aggregation_options]
        if isinstance(activation_options, Callable):
            activation_options = [activation_options]

        if aggregation_default is None:
            aggregation_default = aggregation_options[0]
        if activation_default is None:
            activation_default = activation_options[0]

        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate
        self.bias_lower_bound = bias_lower_bound
        self.bias_upper_bound = bias_upper_bound

        self.aggregation_default = aggregation_options.index(aggregation_default)
        self.aggregation_options = aggregation_options
        self.aggregation_indices = np.arange(len(aggregation_options))
        self.aggregation_replace_rate = aggregation_replace_rate

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = activation_options
        self.activation_indices = np.arange(len(activation_options))
        self.activation_replace_rate = activation_replace_rate

    def new_identity_attrs(self, state):
        return jnp.array([0.0, float(self.aggregation_default), -1.0])

    def new_random_attrs(self, state, randkey):
        k1, k2, k3 = jax.random.split(randkey, num=3)
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        aggregation = jax.random.choice(k2, self.aggregation_indices)
        activation = jax.random.choice(k3, self.activation_indices)
        return jnp.array([bias, aggregation, activation])

    def mutate(self, state, randkey, attrs):
        k1, k2, k3 = jax.random.split(randkey, num=3)
        bias, aggregation, activation = attrs
        bias = mutate_float(
            k1,
            bias,
            self.bias_init_mean,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate,
        )
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        aggregation = mutate_int(
            k2, aggregation, self.aggregation_indices, self.aggregation_replace_rate
        )
        activation = mutate_int(
            k3, activation, self.activation_indices, self.activation_replace_rate
        )
        return jnp.array([bias, aggregation, activation])

    def distance(self, state, attrs1, attrs2):
        bias1, aggregation1, activation1 = attrs1
        bias2, aggregation2, activation2 = attrs2
        return (
            jnp.abs(bias1 - bias2)
            + (aggregation1 != aggregation2)
            + (activation1 != activation2)
        )

    def forward(self, state, attrs, inputs, is_output_node=False, valid_mask=None):
        bias, aggregation, activation = attrs
        if valid_mask is None:
            valid_mask = ~jnp.isnan(inputs)
        z = apply_aggregation(aggregation, inputs, self.aggregation_options, valid_mask)
        z = bias + z
        return jax.lax.cond(
            is_output_node,
            lambda: z,
            lambda: apply_activation(activation, z, self.activation_options),
        )

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx, bias, aggregation, activation = node
        act_func = (
            ACT.identity if int(activation) == -1 else self.activation_options[int(activation)]
        )
        return (
            f"{self.__class__.__name__}("
            f"idx={int(idx):<{idx_width}}, "
            f"bias={round(float(bias), precision):<{precision + 3}}, "
            f"aggregation={get_func_name(self.aggregation_options[int(aggregation)]):<{func_width}}, "
            f"activation={get_func_name(act_func):<{func_width}})"
        )

    def to_dict(self, state, node):
        idx, bias, aggregation, activation = node
        act_func = (
            ACT.identity if int(activation) == -1 else self.activation_options[int(activation)]
        )
        return {
            "idx": int(idx),
            "bias": jnp.float32(bias),
            "agg": get_func_name(self.aggregation_options[int(aggregation)]),
            "act": get_func_name(act_func),
        }
