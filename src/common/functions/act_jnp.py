import jax.numpy as jnp

SCALE = 3.0


def scaled_sigmoid_(z):
    return SCALE / (1.0 + jnp.exp(-z))


def sigmoid_(z):
    return 1.0 / (1.0 + jnp.exp(-z))


def scaled_tanh_(z):
    return SCALE * jnp.tanh(z)


def tanh_(z):
    return jnp.tanh(z)


def sin_(z):
    return jnp.sin(z)


def relu_(z):
    return jnp.maximum(z, 0.0)


def lelu_(z):
    return jnp.where(z > 0.0, z, 0.005 * z)


def identity_(z):
    return z


def inv_(z):
    z = jnp.where(z > 0.0, jnp.maximum(z, 1e-7), jnp.minimum(z, -1e-7))
    return 1.0 / z


def log_(z):
    return jnp.log(jnp.maximum(z, 1e-7))


def exp_(z):
    return jnp.exp(z)


def abs_(z):
    return jnp.abs(z)

