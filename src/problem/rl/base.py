import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simpleneat.common import State

from ..base import BaseProblem


class RLEnv(BaseProblem):
    jitable = False
    supports_backprop = False

    def __init__(self, max_step=3000, repeat_times=1):
        self.max_step = max_step
        self.repeat_times = repeat_times

    def evaluate(self, state: State, randkey, act_func, params):
        rewards = []
        for offset in range(self.repeat_times):
            seed = self._seed_from_key(randkey, offset)
            rewards.append(self.run_episode(state, act_func, params, seed=seed, render=False))
        return float(np.mean(rewards))

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        seed = self._seed_from_key(randkey, 0)
        render = kwargs.get("render", True)
        gif_path = kwargs.get("gif_path")
        total_reward = self.run_episode(
            state,
            act_func,
            params,
            seed=seed,
            render=render,
            sleep=kwargs.get("sleep", 0.0),
            gif_path=gif_path,
            gif_fps=kwargs.get("gif_fps", 30),
        )
        print(f"episode reward: {total_reward}")
        return total_reward

    def run_episode(
        self,
        state,
        act_func,
        params,
        seed: int,
        render: bool,
        sleep: float = 0.0,
        gif_path: str | None = None,
        gif_fps: int = 30,
    ):
        env = self.make_env()
        obs = self.reset_env(env, seed)
        total_reward = 0.0
        done = False
        steps = 0
        compiled_act_func = jax.jit(lambda obs_array: act_func(state, params, obs_array))
        frames = [] if gif_path is not None else None
        try:
            # Compile the policy once per episode instead of paying Python/JAX
            # dispatch overhead at every environment step.
            compiled_act_func(jnp.asarray(obs, dtype=jnp.float32))
            if frames is not None:
                frames.append(self.render_frame(env))
            while not done and steps < self.max_step:
                obs_array = jnp.asarray(obs, dtype=jnp.float32)
                raw_action = compiled_act_func(obs_array)
                action = self.action_from_output(jax.device_get(raw_action))
                obs, reward, done, _ = self.step_env(env, action)
                total_reward += float(reward)
                steps += 1
                if frames is not None:
                    frames.append(self.render_frame(env))
                if render:
                    env.render()
                    if sleep > 0:
                        time.sleep(sleep)
        finally:
            env.close()
        if frames is not None:
            self.save_gif(frames, gif_path, fps=gif_fps)
        return total_reward

    def _seed_from_key(self, randkey, offset: int):
        seed = int(np.asarray(jax.device_get(randkey)).sum()) + offset
        return abs(seed) % (2**31 - 1)

    def make_env(self):
        raise NotImplementedError

    def action_from_output(self, outputs):
        raise NotImplementedError

    def reset_env(self, env, seed):
        try:
            reset_result = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            reset_result = env.reset()
        return reset_result[0] if isinstance(reset_result, tuple) else reset_result

    def step_env(self, env, action):
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            return obs, reward, bool(terminated or truncated), info
        obs, reward, done, info = step_result
        return obs, reward, bool(done), info

    def render_frame(self, env):
        frame = env.render(mode="rgb_array")
        return np.asarray(frame, dtype=np.uint8)

    def save_gif(self, frames, gif_path: str, fps: int = 30):
        if not frames:
            raise ValueError("No frames were captured for GIF export.")

        output_path = Path(gif_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = max(int(round(1000 / max(fps, 1))), 1)

        try:
            import imageio.v2 as imageio

            imageio.mimsave(output_path, frames, duration=1.0 / max(fps, 1), loop=0)
        except ImportError:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError(
                    "GIF export requires `imageio` or `Pillow`. Install with `pip install imageio pillow`."
                ) from exc

            pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
            )
        print(f"saved gif: {output_path}", flush=True)
