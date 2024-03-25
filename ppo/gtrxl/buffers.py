from functools import partial
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

from ppo.gtrxl.type_aliases import XLDictRolloutBufferSamples


class XLDictRolloutBuffer(DictRolloutBuffer):
    """
    Rollout buffer that also stores the xlTEM RNN hidden states and memories.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

    def state_reset(self, last_mems):
        super().reset()
        self.last_mems = last_mems

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[XLDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        # Batch size should be muliples of n_steps (= buffer_size)

        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = obs.swapaxes(0, 1)
            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(0, 1)
            self.generator_ready = True

        idx = np.arange(self.buffer_size)
        np.random.shuffle(idx)

        start_idx = 0
        while start_idx < self.buffer_size:
            batch_inds = idx[start_idx] + 1
            yield self._get_samples(batch_inds)
            start_idx += 1

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> XLDictRolloutBufferSamples:
        to_tensor = lambda x: th.tensor(x, device=self.device)
        observations = {
            key: to_tensor(obs[:, :batch_inds]).squeeze(-1)
            for (key, obs) in self.observations.items()
        }

        return XLDictRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length)
            observations=observations,
            actions=to_tensor(self.actions[:, batch_inds - 1, -1]),
            old_values=to_tensor(self.values[:, batch_inds - 1]),
            old_log_prob=to_tensor(self.log_probs[:, batch_inds - 1]),
            advantages=to_tensor(self.advantages[:, batch_inds - 1]),
            returns=to_tensor(self.returns[:, batch_inds - 1]),
            mems=self.state_indexing(self.last_mems),
            episode_starts=to_tensor(self.episode_starts[:, batch_inds - 1]),
        )

    @staticmethod
    def state_indexing(mems):
        if mems is None:
            return None
        else:
            return mems