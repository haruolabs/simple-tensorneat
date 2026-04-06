from functools import partial

import numpy as np
import jax
from jax import Array, jit, numpy as jnp

I_INF = np.iinfo(jnp.int32).max


def attach_with_inf(arr, idx):
    target_dim = arr.ndim + idx.ndim - 1
    expand_idx = jnp.expand_dims(idx, axis=tuple(range(idx.ndim, target_dim)))
    return jnp.where(expand_idx == I_INF, jnp.nan, arr[idx])


@jit
def fetch_first(mask, default=I_INF) -> Array:
    idx = jnp.argmax(mask)
    return jnp.where(mask[idx], idx, default)


@jit
def fetch_random(randkey, mask, default=I_INF) -> Array:
    true_cnt = jnp.sum(mask)
    cumsum = jnp.cumsum(mask)
    target = jax.random.randint(randkey, shape=(), minval=1, maxval=true_cnt + 1)
    sampled_mask = jnp.where(true_cnt == 0, False, cumsum >= target)
    return fetch_first(sampled_mask, default)


@partial(jit, static_argnames=["reverse"])
def rank_elements(array, reverse=False):
    ranked = array if reverse else -array
    return jnp.argsort(jnp.argsort(ranked))


@jit
def mutate_float(
    randkey,
    val,
    init_mean,
    init_std,
    mutate_power,
    mutate_rate,
    replace_rate,
):
    k1, k2, k3 = jax.random.split(randkey, num=3)
    noise = jax.random.normal(k1, ()) * mutate_power
    replace = jax.random.normal(k2, ()) * init_std + init_mean
    r = jax.random.uniform(k3, ())
    return jnp.where(
        r < mutate_rate,
        val + noise,
        jnp.where((mutate_rate < r) & (r < mutate_rate + replace_rate), replace, val),
    )


@jit
def mutate_int(randkey, val, options, replace_rate):
    k1, k2 = jax.random.split(randkey, num=2)
    r = jax.random.uniform(k1, ())
    return jnp.where(r < replace_rate, jax.random.choice(k2, options), val)


def argmin_with_mask(arr, mask):
    masked_arr = jnp.where(mask, arr, jnp.inf)
    return jnp.argmin(masked_arr)


def hash_array(arr: Array):
    arr = jax.lax.bitcast_convert_type(arr, jnp.uint32)

    def update(i, hash_val):
        return hash_val ^ (
            arr[i] + jnp.uint32(0x9E3779B9) + (hash_val << 6) + (hash_val >> 2)
        )

    return jax.lax.fori_loop(0, arr.size, update, jnp.uint32(0))
