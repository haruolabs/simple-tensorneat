from typing import Callable, List, Tuple, Union

import numpy as np
import jax.numpy as jnp
from jax import Array, vmap

from .func_fit import FuncFit


class CustomFuncFit(FuncFit):
    def __init__(
        self,
        func: Callable,
        low_bounds: Union[List, Tuple, Array],
        upper_bounds: Union[List, Tuple, Array],
        method: str = "sample",
        num_samples: int = 100,
        step_size: Array = None,
        *args,
        **kwargs,
    ):
        low_bounds = np.asarray(low_bounds, dtype=np.float32)
        upper_bounds = np.asarray(upper_bounds, dtype=np.float32)
        if low_bounds.shape != upper_bounds.shape:
            raise ValueError("Bounds must have the same shape")
        func(low_bounds)
        if method not in {"sample", "grid"}:
            raise ValueError("method must be 'sample' or 'grid'")

        self.func = func
        self.low_bounds = low_bounds
        self.upper_bounds = upper_bounds
        self.method = method
        self.num_samples = num_samples
        self.step_size = step_size
        self.generate_dataset()
        super().__init__(*args, **kwargs)

    def generate_dataset(self):
        if self.method == "sample":
            inputs = np.zeros((self.num_samples, self.low_bounds.shape[0]), dtype=np.float32)
            for i in range(self.low_bounds.shape[0]):
                inputs[:, i] = np.random.uniform(
                    self.low_bounds[i], self.upper_bounds[i], size=(self.num_samples,)
                )
        else:
            if self.step_size is None:
                raise ValueError("step_size must be provided for grid sampling")
            step_size = np.asarray(self.step_size, dtype=np.float32)
            inputs = np.zeros((1, 1), dtype=np.float32)
            for i in range(self.low_bounds.shape[0]):
                new_col = np.arange(self.low_bounds[i], self.upper_bounds[i], step_size[i])[:, None]
                inputs = cartesian_product(inputs, new_col)
            inputs = inputs[:, 1:]

        outputs = vmap(self.func)(inputs)
        self.data_inputs = jnp.asarray(inputs)
        self.data_outputs = jnp.asarray(outputs)

    @property
    def inputs(self):
        return self.data_inputs

    @property
    def targets(self):
        return self.data_outputs

    @property
    def input_shape(self):
        return self.data_inputs.shape

    @property
    def output_shape(self):
        return self.data_outputs.shape


def cartesian_product(arr1, arr2):
    repeated_arr1 = np.repeat(arr1, arr2.shape[0], axis=0)
    tiled_arr2 = np.tile(arr2, (arr1.shape[0], 1))
    return np.concatenate((repeated_arr1, tiled_arr2), axis=1)

