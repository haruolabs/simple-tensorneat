import jax
import jax.numpy as jnp
from jax import vmap

from simpleneat.common import State

from ..base import BaseProblem


class FuncFit(BaseProblem):
    jitable = True
    supports_backprop = True

    def __init__(self, error_method: str = "mse"):
        if error_method not in {"mse", "rmse", "mae", "mape"}:
            raise ValueError("Unsupported error method")
        self.error_method = error_method

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):
        predict = vmap(act_func, in_axes=(None, None, 0))(state, params, self.inputs)
        return -self.loss_from_outputs(predict, self.targets)

    def loss_from_outputs(self, predict, targets):
        if self.error_method == "mse":
            return jnp.mean((predict - targets) ** 2)
        if self.error_method == "rmse":
            return jnp.sqrt(jnp.mean((predict - targets) ** 2))
        if self.error_method == "mae":
            return jnp.mean(jnp.abs(predict - targets))
        if self.error_method == "mape":
            safe_targets = jnp.where(targets == 0, 1e-7, targets)
            return jnp.mean(jnp.abs((predict - targets) / safe_targets))
        raise NotImplementedError

    @property
    def backprop_inputs(self):
        return self.inputs

    @property
    def backprop_targets(self):
        return self.targets

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        predict = vmap(act_func, in_axes=(None, None, 0))(state, params, self.inputs)
        inputs, targets, predict = jax.device_get([self.inputs, self.targets, predict])
        fitness = self.evaluate(state, randkey, act_func, params)
        loss = -jax.device_get(fitness)
        lines = []
        for x, y, pred in zip(inputs, targets, predict):
            lines.append(f"input: {x}, target: {y}, predict: {pred}")
        lines.append(f"loss: {loss}")
        print("\n".join(lines))

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError
