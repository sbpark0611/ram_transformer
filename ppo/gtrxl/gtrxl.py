import math
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from utils.misc import eval_memory_correct

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    safe_mean,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv

from ppo.gtrxl.buffers import XLDictRolloutBuffer
from ppo.gtrxl.policies import GTRxLActorCriticPolicy


class GTRxL_PPO(OnPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "GTRxLActorCriticPolicy": GTRxLActorCriticPolicy
    }

    def __init__(
        self,
        policy: Union[str, Type[GTRxLActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        latent_detach: bool = False,
        normalize_advantage: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        stats_window_size: int = 100,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = (
            spaces.Box,
            spaces.Discrete,
            spaces.MultiDiscrete,
            spaces.MultiBinary,
        ),
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            stats_window_size=stats_window_size,
            supported_action_spaces=supported_action_spaces,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.latent_detach = latent_detach
        self.target_kl = target_kl
        self._last_mems = None
        self._last_act = None
        self._criterion = th.nn.CrossEntropyLoss()
        self.mem_len = policy_kwargs["model_config"].mem_len
        self.prev_rollout_pos = []

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            self.latent_detach,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # We assume that xl for the actor and the critic
        # have the same architecture
        xl = self.policy.model

        if not isinstance(self.policy, GTRxLActorCriticPolicy):
            raise ValueError("Policy must subclass GTRxLActorCriticPolicy")

        # hidden and cell states for actor and critic
        self._last_mems = None

        self.rollout_buffer = XLDictRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert (
                    self.clip_range_vf > 0
                ), "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            max_rew_ep_info = info.get("expected_max_rwd")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                maybe_ep_info["max_r"] = max_rew_ep_info
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: XLDictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        last_mems = deepcopy(self._last_mems)
        rollout_buffer.state_reset(last_mems)

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = (
                    th.tensor(self._last_episode_starts).float().to(self.device)
                )
                actions, values, log_probs, last_mems = self.policy.forward(
                    obs_tensor,
                    last_mems,
                    episode_starts,
                    False,
                )

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions.flatten())

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            if dones.any() and infos[0].get("terminal_observation") is not None:
                terminal_obs = {
                    "position": np.array(
                        [info["terminal_observation"]["position"] for info in infos]
                    ),
                    "prev_position": np.array(
                        [
                            info["terminal_observation"]["prev_position"]
                            for info in infos
                        ]
                    ),
                    "goal": np.array(
                        [info["terminal_observation"]["goal"] for info in infos]
                    ),
                    "prev_action": np.array(
                        [info["terminal_observation"]["prev_action"] for info in infos]
                    ),
                }
                terminal_obs = obs_as_tensor(terminal_obs, self.device)
                terminal_mems = last_mems

                with th.no_grad():
                    terminal_value = self.policy.predict_values(
                        terminal_obs, terminal_mems, 
                    )[..., 0]
                rewards += self.gamma * terminal_value.cpu().numpy()


            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

            self._last_obs = new_obs
            self._last_act = actions
            self._last_episode_starts = dones
            self._last_mems = last_mems

        with th.no_grad():
            # Compute value for the last timestep
            new_obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(
                new_obs_tensor,
                last_mems
            )
        if dones.any():
            # hidden and cell states for actor and critic
            self._last_mems = None
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.mems,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages.flatten()
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob.flatten())

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2))

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                # TODO: vales_pred shape is not (bsz, seq_len)!! fix it
                value_loss = th.mean(((rollout_data.returns.flatten() - values_pred) ** 2))

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean(((th.exp(log_ratio) - 1) - log_ratio)).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "GTRxL_PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "GTRxL_PPO":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
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
                    self.logger.record(
                        "rollout/ep_rew_mean",
                        avg_rew,
                    )
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
