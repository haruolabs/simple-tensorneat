from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from simpleneat.common import State

from ..base import BaseProblem
from .generators import DatasetGenerator


class ClassificationProblem(BaseProblem):
    jitable = True
    supports_backprop = True

    def __init__(
        self,
        dataset_generator: DatasetGenerator,
        train_size: int = 200,
        test_size: int = 200,
        noise: float = 0.5,
        penalty_connection_factor: float = 0.03,
        seed: int | None = None,
    ):
        if train_size <= 0:
            raise ValueError("train_size must be positive")
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        if penalty_connection_factor < 0:
            raise ValueError("penalty_connection_factor must be non-negative")

        self.dataset_generator = dataset_generator
        self.train_size = train_size
        self.test_size = test_size
        self.noise = noise
        self.penalty_connection_factor = penalty_connection_factor
        self.seed = seed

        rng = np.random.default_rng(seed)
        train_inputs, train_targets = dataset_generator(train_size, noise, rng)
        test_inputs, test_targets = dataset_generator(test_size, noise, rng)
        train_inputs, train_targets = self._shuffle_dataset(train_inputs, train_targets, rng)
        test_inputs, test_targets = self._shuffle_dataset(test_inputs, test_targets, rng)
        self.train_inputs, self.train_targets = self._validate_dataset(
            train_inputs, train_targets, split_name="train"
        )
        self.test_inputs, self.test_targets = self._validate_dataset(
            test_inputs, test_targets, split_name="test"
        )

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):
        del randkey
        predict = vmap(act_func, in_axes=(None, None, 0))(state, params, self.train_inputs)
        mse = self.loss_from_outputs(predict, self.train_targets)
        return -mse * self.network_complexity(params)

    def loss_from_outputs(self, predict, targets):
        predict = jnp.reshape(predict, targets.shape)
        return jnp.mean((predict - targets) ** 2)

    def network_complexity(self, params):
        conns = self._extract_conns(params)
        penalty_connection = jnp.sum(~jnp.isnan(conns[:, 0]), dtype=jnp.float32)
        return 1.0 + self.penalty_connection_factor * jnp.sqrt(penalty_connection)

    def classification_accuracy(self, predict, targets):
        predict = jnp.reshape(predict, targets.shape)
        labels = (predict >= 0.5).astype(targets.dtype)
        return jnp.mean((labels == targets).astype(jnp.float32))

    @property
    def backprop_inputs(self):
        return self.train_inputs

    @property
    def backprop_targets(self):
        return self.train_targets

    @property
    def inputs(self):
        return self.train_inputs

    @property
    def targets(self):
        return self.train_targets

    @property
    def input_shape(self):
        return self.train_inputs.shape

    @property
    def output_shape(self):
        return self.train_targets.shape

    def show(
        self,
        state,
        randkey,
        act_func,
        params,
        *args,
        decision_boundary_path: str | None = None,
        test_decision_boundary_path: str | None = None,
        train_decision_boundary_path: str | None = None,
        grid_size: int = 250,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        **kwargs,
    ):
        del args, kwargs
        train_predict = vmap(act_func, in_axes=(None, None, 0))(state, params, self.train_inputs)
        test_predict = vmap(act_func, in_axes=(None, None, 0))(state, params, self.test_inputs)
        train_fitness = self.evaluate(state, randkey, act_func, params)
        train_accuracy = self.classification_accuracy(train_predict, self.train_targets)
        test_accuracy = self.classification_accuracy(test_predict, self.test_targets)

        train_fitness, train_accuracy, test_accuracy = jax.device_get(
            [train_fitness, train_accuracy, test_accuracy]
        )
        print(f"train fitness: {float(train_fitness):.6f}")
        print(f"train accuracy: {float(train_accuracy):.6f}")
        print(f"test accuracy: {float(test_accuracy):.6f}")

        test_output_path = test_decision_boundary_path or decision_boundary_path

        if test_output_path is not None:
            self.draw_decision_boundary(
                state,
                act_func,
                params,
                split="test",
                save_path=test_output_path,
                grid_size=grid_size,
                x_range=x_range,
                y_range=y_range,
            )

        if train_decision_boundary_path is not None:
            self.draw_decision_boundary(
                state,
                act_func,
                params,
                split="train",
                save_path=train_decision_boundary_path,
                grid_size=grid_size,
                x_range=x_range,
                y_range=y_range,
            )

    def draw_decision_boundary(
        self,
        state,
        act_func,
        params,
        split: str = "test",
        save_path: str | None = None,
        grid_size: int = 250,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ):
        if grid_size <= 1:
            raise ValueError("grid_size must be greater than 1")
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")
        if x_range is not None and len(x_range) != 2:
            raise ValueError("x_range must be a tuple of length 2")
        if y_range is not None and len(y_range) != 2:
            raise ValueError("y_range must be a tuple of length 2")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for draw_decision_boundary(). Install it with `pip install matplotlib`."
            ) from exc

        split_inputs = self.train_inputs if split == "train" else self.test_inputs
        split_targets = self.train_targets if split == "train" else self.test_targets
        inputs = np.asarray(jax.device_get(split_inputs))
        targets = np.asarray(jax.device_get(split_targets)).reshape(-1)

        if x_range is None:
            x_min, x_max = float(inputs[:, 0].min()), float(inputs[:, 0].max())
            x_pad = max(0.2, (x_max - x_min) * 0.1)
            x_bounds = (x_min - x_pad, x_max + x_pad)
        else:
            x_bounds = (float(x_range[0]), float(x_range[1]))

        if y_range is None:
            y_min, y_max = float(inputs[:, 1].min()), float(inputs[:, 1].max())
            y_pad = max(0.2, (y_max - y_min) * 0.1)
            y_bounds = (y_min - y_pad, y_max + y_pad)
        else:
            y_bounds = (float(y_range[0]), float(y_range[1]))

        x_values = np.linspace(x_bounds[0], x_bounds[1], grid_size, dtype=np.float32)
        y_values = np.linspace(y_bounds[0], y_bounds[1], grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(x_values, y_values)
        grid_points = jnp.asarray(
            np.column_stack([xx.ravel(), yy.ravel()]),
            dtype=jnp.float32,
        )

        grid_predict = vmap(act_func, in_axes=(None, None, 0))(state, params, grid_points)
        grid_predict = np.asarray(jax.device_get(grid_predict)).reshape(grid_size, grid_size)

        split_predict = vmap(act_func, in_axes=(None, None, 0))(state, params, split_inputs)
        split_predict = np.asarray(jax.device_get(split_predict)).reshape(-1)
        split_accuracy = float(
            jax.device_get(
                self.classification_accuracy(jnp.asarray(split_predict[:, None]), split_targets)
            )
        )

        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        contour = ax.contourf(
            xx,
            yy,
            grid_predict,
            levels=np.linspace(0.0, 1.0, 21),
            cmap="RdBu",
            alpha=0.7,
        )
        ax.contour(
            xx,
            yy,
            grid_predict,
            levels=[0.5],
            colors="black",
            linewidths=1.5,
        )

        class_zero = targets == 0
        class_one = targets == 1
        ax.scatter(
            inputs[class_zero, 0],
            inputs[class_zero, 1],
            c="#1f77b4",
            edgecolors="black",
            linewidths=0.5,
            s=36,
            label=f"{split.capitalize()} class 0",
        )
        ax.scatter(
            inputs[class_one, 0],
            inputs[class_one, 1],
            c="#d62728",
            edgecolors="black",
            linewidths=0.5,
            s=36,
            label=f"{split.capitalize()} class 1",
        )

        ax.set_title(
            f"Decision Boundary ({split} accuracy={split_accuracy:.3f})"
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(*x_bounds)
        ax.set_ylim(*y_bounds)
        ax.legend(loc="upper right")
        colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
        colorbar.set_label("Predicted probability")
        plt.tight_layout()

        if save_path:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"{split.capitalize()} decision boundary saved to {output}")
            return None

        return fig

    def _extract_conns(self, params):
        if isinstance(params, tuple):
            if len(params) >= 3:
                return params[2]
            if len(params) >= 2:
                return params[1]
        raise TypeError("Unable to extract connections from transformed params")

    def _shuffle_dataset(self, inputs, targets, rng: np.random.Generator):
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        indices = rng.permutation(len(inputs))
        return inputs[indices], targets[indices]

    def _validate_dataset(self, inputs, targets, split_name: str):
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError(f"{split_name} inputs must have shape (N, 2)")
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.ndim != 2 or targets.shape[1] != 1:
            raise ValueError(f"{split_name} targets must have shape (N, 1)")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"{split_name} inputs and targets must have the same length")
        if not np.isin(targets, [0.0, 1.0]).all():
            raise ValueError(f"{split_name} targets must be binary labels in {{0, 1}}")

        return jnp.asarray(inputs), jnp.asarray(targets)
