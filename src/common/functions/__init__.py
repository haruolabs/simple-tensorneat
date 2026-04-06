import jax
import jax.numpy as jnp

from .act_jnp import (
    abs_,
    exp_,
    identity_,
    inv_,
    lelu_,
    log_,
    relu_,
    scaled_sigmoid_,
    scaled_tanh_,
    sigmoid_,
    sin_,
    tanh_,
)
from .agg_jnp import max_, maxabs_, mean_, min_, product_, sum_
from .manager import FunctionManager

act_name2jnp = {
    "scaled_sigmoid": scaled_sigmoid_,
    "sigmoid": sigmoid_,
    "scaled_tanh": scaled_tanh_,
    "tanh": tanh_,
    "sin": sin_,
    "relu": relu_,
    "lelu": lelu_,
    "identity": identity_,
    "inv": inv_,
    "log": log_,
    "exp": exp_,
    "abs": abs_,
}

agg_name2jnp = {
    "sum": sum_,
    "product": product_,
    "max": max_,
    "min": min_,
    "maxabs": maxabs_,
    "mean": mean_,
}

ACT = FunctionManager(act_name2jnp)
AGG = FunctionManager(agg_name2jnp)


def apply_activation(idx, z, act_funcs):
    idx = jnp.asarray(idx, dtype=jnp.int32)
    return jax.lax.cond(
        idx == -1,
        lambda: z,
        lambda: jax.lax.switch(idx, act_funcs, z),
    )


def apply_aggregation(idx, z, agg_funcs, mask):
    idx = jnp.asarray(idx, dtype=jnp.int32)
    return jax.lax.cond(
        jnp.all(~mask),
        lambda: 0.0,
        lambda: jax.lax.switch(idx, agg_funcs, z, mask),
    )


def get_func_name(func):
    name = func.__name__
    if name.endswith("_"):
        name = name[:-1]
    return name

