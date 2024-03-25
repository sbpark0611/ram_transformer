import sys
import time
from typing import TypeVar

from sb3_contrib import RecurrentPPO as RPPO
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import safe_mean


SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")


class RecurrentPPO(RPPO):
    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
                )
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed
                )
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    avg_rew = safe_mean(
                        [ep_info["r"] for ep_info in self.ep_info_buffer]
                    )
                    self.logger.record("rollout/ep_rew_mean", avg_rew)
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                    oracle_reward = (
                        self.env.envs[0].max_episode_steps
                        / self.env.envs[0].oracle_min_num_actions
                    )
                    self.logger.record(
                        "rollout/fraction_of_oracle", avg_rew / oracle_reward
                    )
                    self.logger.record("rollout/oracle_reward", oracle_reward)
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
                )
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self