import numpy as np

from .base import RLEnv


class SlimeVolleyEnv(RLEnv):
    def __init__(self, env_id: str = "SlimeVolley-v0", max_step: int = 3000, repeat_times: int = 1):
        self.env_id = env_id
        super().__init__(max_step=max_step, repeat_times=repeat_times)

    def make_env(self):
        try:
            import gym
            import slimevolleygym  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SlimeVolleyEnv requires `gym` and `slimevolleygym`. Install with `pip install -e \".[rl]\"`."
            ) from exc
        try:
            # Legacy Gym's passive env checker crashes with NumPy 2.x because it
            # references `np.bool8`, which was removed. We already normalize
            # reset/step outputs in `RLEnv`, so the extra checker is unnecessary.
            return gym.make(self.env_id, disable_env_checker=True)
        except TypeError:
            return gym.make(self.env_id)

    @property
    def input_shape(self):
        return (12,)

    @property
    def output_shape(self):
        return (3,)

    def action_from_output(self, outputs):
        outputs = np.asarray(outputs, dtype=np.float32).reshape(-1)
        return (outputs[:3] > 0.0).astype(np.int8)

    def render_frame(self, env):
        try:
            return super().render_frame(env)
        except Exception:
            import slimevolleygym.slimevolley as slimevolley

            slimevolley.setPixelObsMode()
            frame = env.unwrapped.game.display(None)
            return np.asarray(frame, dtype=np.uint8).copy()
