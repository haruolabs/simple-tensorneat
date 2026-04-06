from __future__ import annotations

from typing import Callable

import numpy as np

DatasetGenerator = Callable[
    [int, float, np.random.Generator],
    tuple[np.ndarray, np.ndarray],
]


def generate_xor_dataset(
    num_points: int = 200,
    noise: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng

    xs = rng.uniform(-5.0, 5.0, size=num_points) + rng.normal(0.0, noise, size=num_points)
    ys = rng.uniform(-5.0, 5.0, size=num_points) + rng.normal(0.0, noise, size=num_points)
    inputs = np.column_stack((xs, ys)).astype(np.float32)

    labels = (
        ((inputs[:, 0] > 0.0) & (inputs[:, 1] > 0.0))
        | ((inputs[:, 0] < 0.0) & (inputs[:, 1] < 0.0))
    ).astype(np.float32)[:, None]
    return inputs, labels


def generate_spiral_dataset(
    num_points: int = 200,
    noise: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng

    first_count = num_points // 2
    second_count = num_points - first_count
    inputs = []
    labels = []

    def gen_spiral(count: int, delta_t: float, label: float):
        if count <= 0:
            return
        for i in range(count):
            r = i / count * 6.0
            t = 1.75 * i / count * 2.0 * np.pi + delta_t
            x = r * np.sin(t) + rng.uniform(-1.0, 1.0) * noise
            y = r * np.cos(t) + rng.uniform(-1.0, 1.0) * noise
            inputs.append((x, y))
            labels.append((label,))

    gen_spiral(first_count, 0.0, 0.0)
    gen_spiral(second_count, np.pi, 1.0)

    return np.asarray(inputs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def generate_circle_dataset(
    num_points: int = 200,
    noise: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng

    radius = 5.0
    inner_count = num_points // 2
    outer_count = num_points - inner_count
    inputs = []
    labels = []

    def get_circle_label(x: float, y: float) -> float:
        return 1.0 if x * x + y * y < (radius * 0.5) * (radius * 0.5) else 0.0

    for _ in range(inner_count):
        r = rng.uniform(0.0, radius * 0.5)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = rng.uniform(-radius, radius) * noise / 3.0
        noise_y = rng.uniform(-radius, radius) * noise / 3.0
        inputs.append((x + noise_x, y + noise_y))
        labels.append((get_circle_label(x, y),))

    for _ in range(outer_count):
        r = rng.uniform(radius * 0.75, radius)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = rng.uniform(-radius, radius) * noise / 3.0
        noise_y = rng.uniform(-radius, radius) * noise / 3.0
        inputs.append((x + noise_x, y + noise_y))
        labels.append((get_circle_label(x, y),))

    return np.asarray(inputs, dtype=np.float32), np.asarray(labels, dtype=np.float32)
