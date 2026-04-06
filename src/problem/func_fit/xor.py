import jax.numpy as jnp

from .func_fit import FuncFit


class XOR(FuncFit):
    @property
    def inputs(self):
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=jnp.float32,
        )

    @property
    def targets(self):
        return jnp.array(
            [
                [0.0],
                [1.0],
                [1.0],
                [0.0],
            ],
            dtype=jnp.float32,
        )

    @property
    def input_shape(self):
        return self.inputs.shape

    @property
    def output_shape(self):
        return self.targets.shape

