from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simpleneat.common import State

from ..base import BaseProblem


class EvoJAXSlimeVolleyEnv(BaseProblem):
    jitable = True
    supports_backprop = False

    def __init__(
        self,
        max_step: int = 3000,
        repeat_times: int = 1,
        test: bool = False,
        terminate_on_point: bool = False,
        action_threshold: float = 0.75,
        hit_reward_scale: float = 0.0,
        feature_mode: str = "raw",
    ):
        self.max_step = max_step
        self.repeat_times = repeat_times
        self.test = test
        self.terminate_on_point = terminate_on_point
        self.action_threshold = action_threshold
        self.hit_reward_scale = hit_reward_scale
        if feature_mode not in {"raw", "relative_ball"}:
            raise ValueError("feature_mode must be 'raw' or 'relative_ball'")
        self.feature_mode = feature_mode

        try:
            from evojax.task.slimevolley import (
                Game,
                GRAVITY,
                MAXLIVES,
                MAX_BALL_SPEED,
                NUDGE,
                PLAYER_SPEED_X,
                PLAYER_SPEED_Y,
                REF_U,
                REF_W,
                REF_WALL_WIDTH,
                TIMESTEP,
                SlimeVolley,
                detect_done,
                update_state_for_new_match,
            )
        except ImportError as exc:
            raise ImportError(
                "EvoJAXSlimeVolleyEnv requires `evojax`. Use the `evojax` conda env or install EvoJAX."
            ) from exc

        self.gravity = float(GRAVITY)
        self.Game = Game
        self.detect_done = detect_done
        self.max_lives = int(MAXLIVES)
        self.max_ball_speed = float(MAX_BALL_SPEED)
        self.nudge = float(NUDGE)
        self.player_speed_x = float(PLAYER_SPEED_X)
        self.player_speed_y = float(PLAYER_SPEED_Y)
        self.ref_u = float(REF_U)
        self.ref_w = float(REF_W)
        self.ref_wall_width = float(REF_WALL_WIDTH)
        self.timestep = float(TIMESTEP)
        self.update_state_for_new_match = update_state_for_new_match
        self.task = SlimeVolley(max_steps=max_step, test=test)

    def setup(self, state: State = State()):
        return state

    @property
    def input_shape(self):
        if self.feature_mode == "relative_ball":
            return (6,)
        return self.task.obs_shape

    @property
    def output_shape(self):
        return self.task.act_shape

    def action_from_output(self, outputs):
        outputs = jnp.asarray(outputs, dtype=jnp.float32).reshape(-1)
        return (outputs[:3] > self.action_threshold).astype(jnp.float32)

    def process_obs(self, obs):
        obs = jnp.asarray(obs, dtype=jnp.float32).reshape(-1)
        if self.feature_mode == "raw":
            return obs

        x, y, vx, vy, bx, by, bvx, bvy, ox, oy, ovx, ovy = obs
        del vx, vy, ox, oy, ovx, ovy
        return jnp.array([x, y, bx - x, by - y, bvx, bvy], dtype=jnp.float32)

    def evaluate(self, state, randkey, act_func, params):
        keys = jax.random.split(randkey, self.repeat_times)
        rewards = jax.vmap(
            lambda key: self._run_rollout_jax(
                state=state,
                randkey=key,
                act_func=act_func,
                params=params,
                terminate_on_point=self.terminate_on_point,
            )
        )(keys)
        return jnp.mean(rewards)

    def evaluate_population_self_play(
        self,
        state,
        randkey,
        act_func,
        transformed_population,
        tournament_rounds: int = 4,
        swap_sides: bool = True,
    ):
        if tournament_rounds <= 0:
            raise ValueError("tournament_rounds must be positive")

        leaf = jax.tree_util.tree_leaves(transformed_population)[0]
        pop_size = leaf.shape[0]
        pair_count = pop_size // 2
        scores = jnp.zeros((pop_size,), dtype=jnp.float32)
        counts = jnp.zeros((pop_size,), dtype=jnp.float32)
        round_keys = jax.random.split(randkey, tournament_rounds)

        def select_params(indices):
            return jax.tree_util.tree_map(lambda x: x[indices], transformed_population)

        for round_key in round_keys:
            perm = jax.random.permutation(round_key, pop_size)
            left_idx = perm[: 2 * pair_count : 2]
            right_idx = perm[1 : 2 * pair_count : 2]
            left_params = select_params(left_idx)
            right_params = select_params(right_idx)
            match_keys = jax.random.split(round_key, pair_count * (2 if swap_sides else 1))
            keys_a = match_keys[:pair_count]

            rewards_a = jax.vmap(
                lambda key, lp, rp: self.play_match(state, key, act_func, lp, rp),
                in_axes=(0, 0, 0),
            )(keys_a, left_params, right_params)
            scores = scores.at[left_idx].add(-rewards_a)
            scores = scores.at[right_idx].add(rewards_a)
            counts = counts.at[left_idx].add(1.0)
            counts = counts.at[right_idx].add(1.0)

            if swap_sides:
                keys_b = match_keys[pair_count:]
                rewards_b = jax.vmap(
                    lambda key, lp, rp: self.play_match(state, key, act_func, rp, lp),
                    in_axes=(0, 0, 0),
                )(keys_b, left_params, right_params)
                scores = scores.at[left_idx].add(rewards_b)
                scores = scores.at[right_idx].add(-rewards_b)
                counts = counts.at[left_idx].add(1.0)
                counts = counts.at[right_idx].add(1.0)

        return scores / jnp.maximum(counts, 1.0)

    def play_match(self, state, randkey, act_func, left_params, right_params):
        task_state = self.task.reset(randkey[None, :])
        game_state = jax.tree_util.tree_map(lambda x: x[0], task_state.game_state)

        def cond_fn(carry):
            _, _, steps, done, _ = carry
            return (steps < self.max_step) & (~done)

        def body_fn(carry):
            game_state_, total_reward, steps, _, key = carry
            obs_left, obs_right = self.get_both_obs(game_state_)
            left_raw = act_func(state, left_params, self.process_obs(obs_left))
            right_raw = act_func(state, right_params, self.process_obs(obs_right))
            left_action = self.action_from_output(left_raw)
            right_action = self.action_from_output(right_raw)
            step_key, next_key = jax.random.split(key)
            next_game_state, reward = self.step_self_play(
                game_state_, left_action, right_action, step_key
            )
            done = self.detect_done(next_game_state) if self.test else jnp.bool_(False)
            return next_game_state, total_reward + reward, steps + 1, done, next_key

        _, total_reward, _, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                game_state,
                jnp.float32(0.0),
                jnp.int32(0),
                jnp.bool_(False),
                randkey,
            ),
        )
        return total_reward

    def get_both_obs(self, game_state):
        game = self.Game(game_state)
        return game.agent_left.getObservation(), game.agent_right.getObservation()

    def step_self_play(self, game_state, left_action, right_action, key):
        game = self.Game(game_state)
        game.setLeftAction(left_action)
        game.setRightAction(right_action)
        game.setAction()
        reward = game.step()
        updated_game_state = game.getGameState()
        updated_game_state = self.update_state_for_new_match(updated_game_state, reward, key)
        return updated_game_state, reward

    def _run_rollout_jax(self, state, randkey, act_func, params, terminate_on_point: bool):
        task_state = self.task.reset(randkey[None, :])

        def cond_fn(carry):
            _, _, steps, done = carry
            return (steps < self.max_step) & (~done)

        def body_fn(carry):
            task_state_, total_reward, steps, _ = carry
            obs = self.process_obs(task_state_.obs[0])
            raw_action = act_func(state, params, obs)
            action = self.action_from_output(raw_action)[None, :]
            hit_reward = self._detect_right_hit(task_state_, action) * self.hit_reward_scale
            next_state, reward, env_done = self.task.step(task_state_, action)
            reward = reward[0].astype(jnp.float32)
            shaped_reward = reward + hit_reward.astype(jnp.float32)
            done = env_done[0]
            if terminate_on_point:
                done = done | (reward != 0.0)
            return next_state, total_reward + shaped_reward, steps + 1, done

        _, total_reward, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                task_state,
                jnp.float32(0.0),
                jnp.int32(0),
                jnp.bool_(False),
            ),
        )
        return total_reward

    def _detect_right_hit(self, task_state, action):
        game_state = task_state.game_state
        ball = jax.tree_util.tree_map(lambda x: x[0], game_state.ball)
        agent = jax.tree_util.tree_map(lambda x: x[0], game_state.agent_right)
        action = action[0]

        forward = action[0] > 0
        backward = action[1] > 0
        jump = action[2] > 0

        desired_vx = jnp.float32(0.0)
        desired_vx = jnp.where(forward & (~backward), -self.player_speed_x, desired_vx)
        desired_vx = jnp.where(backward & (~forward), self.player_speed_x, desired_vx)
        desired_vy = jnp.where(jump, self.player_speed_y, jnp.float32(0.0))

        moved_agent_vy = agent.vy + self.gravity * self.timestep
        moved_agent_vy = jnp.where(
            agent.y <= self.ref_u + self.nudge * self.timestep,
            desired_vy,
            moved_agent_vy,
        )
        moved_agent_vx = desired_vx * agent.direction.astype(jnp.float32)
        moved_agent_x = agent.x + moved_agent_vx * self.timestep
        moved_agent_y = agent.y + moved_agent_vy * self.timestep

        right_agent_x = moved_agent_x
        right_agent_y = jnp.where(moved_agent_y <= self.ref_u, self.ref_u, moved_agent_y)
        wall_x = self.ref_wall_width / 2.0 + agent.r
        max_x = self.ref_w / 2.0 - agent.r
        right_agent_x = jnp.where(
            moved_agent_x * agent.direction <= wall_x,
            agent.direction.astype(jnp.float32) * wall_x,
            right_agent_x,
        )
        right_agent_x = jnp.where(
            moved_agent_x * agent.direction >= max_x,
            agent.direction.astype(jnp.float32) * max_x,
            right_agent_x,
        )

        moved_ball_vx = ball.vx
        moved_ball_vy = ball.vy + self.gravity * self.timestep
        ball_speed_sq = moved_ball_vx * moved_ball_vx + moved_ball_vy * moved_ball_vy
        max_speed_sq = self.max_ball_speed * self.max_ball_speed
        speed = jnp.maximum(jnp.sqrt(ball_speed_sq), jnp.float32(1e-6))
        moved_ball_vx = jnp.where(ball_speed_sq > max_speed_sq, moved_ball_vx / speed, moved_ball_vx)
        moved_ball_vy = jnp.where(ball_speed_sq > max_speed_sq, moved_ball_vy / speed, moved_ball_vy)
        moved_ball_vx = jnp.where(ball_speed_sq > max_speed_sq, moved_ball_vx * self.max_ball_speed, moved_ball_vx)
        moved_ball_vy = jnp.where(ball_speed_sq > max_speed_sq, moved_ball_vy * self.max_ball_speed, moved_ball_vy)
        moved_ball_x = ball.x + moved_ball_vx * self.timestep
        moved_ball_y = ball.y + moved_ball_vy * self.timestep

        dx = moved_ball_x - right_agent_x
        dy = moved_ball_y - right_agent_y
        collision_radius = ball.r + agent.r
        return (dx * dx + dy * dy < collision_radius * collision_radius).astype(jnp.float32)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        del args
        gif_path = kwargs.get("gif_path")
        gif_fps = kwargs.get("gif_fps", 25)
        stop_lives_lost = kwargs.get("stop_lives_lost")

        if stop_lives_lost is not None:
            stop_lives_lost = int(stop_lives_lost)
            if stop_lives_lost <= 0 or stop_lives_lost > self.max_lives:
                raise ValueError(
                    f"stop_lives_lost must be in [1, {self.max_lives}]"
                )

        task_state = self.task.reset(randkey[None, :])
        total_reward = 0.0
        steps = 0
        frames = [] if gif_path is not None else None

        compiled_act_func = jax.jit(lambda obs_array: act_func(state, params, obs_array))
        compiled_act_func(self.process_obs(task_state.obs[0]))

        if frames is not None:
            frames.append(self.render_frame(task_state))

        done = False
        while not done and (stop_lives_lost is not None or steps < self.max_step):
            raw_action = compiled_act_func(self.process_obs(task_state.obs[0]))
            action = np.asarray(jax.device_get(self.action_from_output(raw_action)), dtype=np.float32)[
                None, :
            ]
            task_state, reward, env_done = self.task.step(task_state, jnp.asarray(action))
            reward = float(jax.device_get(reward[0]))
            total_reward += reward
            steps += 1
            if stop_lives_lost is None:
                done = bool(jax.device_get(env_done[0]))
            else:
                left_lives = int(jax.device_get(task_state.game_state.agent_left.life[0]))
                right_lives = int(jax.device_get(task_state.game_state.agent_right.life[0]))
                left_lost = self.max_lives - left_lives
                right_lost = self.max_lives - right_lives
                done = max(left_lost, right_lost) >= stop_lives_lost
            if self.terminate_on_point and reward != 0.0:
                done = True
            if frames is not None and not done:
                frames.append(self.render_frame(task_state))

        if frames is not None:
            self.save_gif(frames, gif_path, fps=gif_fps)

        print(f"episode reward: {total_reward}")
        return total_reward

    def render_frame(self, task_state):
        single_state = jax.tree_util.tree_map(lambda x: x[0], task_state)
        return self.task.render(single_state)

    def save_gif(self, frames, gif_path: str, fps: int = 25):
        if not frames:
            raise ValueError("No frames were captured for GIF export.")

        output_path = Path(gif_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = max(int(round(1000 / max(fps, 1))), 1)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"saved gif: {output_path}", flush=True)
