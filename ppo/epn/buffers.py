from typing import Generator, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer

from ppo.epn.type_aliases import EpisodicDictRolloutBufferSamples, TensorDict


class EPNDictRolloutBuffer(DictRolloutBuffer):
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

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[EpisodicDictRolloutBufferSamples, None, None]:
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
            ]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(0, 1)
            self.generator_ready = True

        # TODO: batch_size implementation
        batch_size = self.n_envs

        indices = np.repeat(np.arange(self.buffer_size)[None, :], self.n_envs, axis=0)
        for i in range(self.n_envs):
            np.random.shuffle(indices[i])

        start_idx = 0
        while start_idx < self.buffer_size:
            batch_inds = indices[:, start_idx]
            yield self._get_samples(batch_inds)
            start_idx += 1

    def episodic_storage(self) -> TensorDict:
        if self.pos >= self.buffer_size:
            _pos = self.buffer_size - 1
        else:
            _pos = self.pos
        _observations = {}
        for key, obs in self.observations.items():
            _observations[key] = obs.swapaxes(0, 1)

        indices = np.repeat(np.arange(self.buffer_size)[None, :], self.n_envs, axis=0)
        to_tensor = lambda x: th.tensor(x, device=self.device).long()
        to_zero = lambda x: th.zeros_like(x).long()
        batch_inds = indices[:, _pos]
        zeros = {key: to_zero(to_tensor(obs)) for (key, obs) in _observations.items()}
        observations = {}
        masks = indices <= batch_inds[:, None]

        for key, obs in zeros.items():
            obs[masks, :] = to_tensor(_observations[key][masks])
            observations[key] = obs.squeeze(-1)
        return observations

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> EpisodicDictRolloutBufferSamples:
        to_tensor = lambda x: th.tensor(x, device=self.device)
        to_zero = lambda x: th.zeros_like(x).long()
        indices = np.repeat(np.arange(self.buffer_size)[None, :], self.n_envs, axis=0)
        zeros = {
            key: to_zero(to_tensor(obs)) for (key, obs) in self.observations.items()
        }
        memory = {}
        masks = indices < batch_inds[:, None]

        for key, obs in zeros.items():
            obs[masks, :] = to_tensor(self.observations[key][masks]).long()
            memory[key] = obs.squeeze(-1)

        take_along_axis = lambda x: to_tensor(
            np.take_along_axis(x, batch_inds[:, None], axis=1)
        )

        observations = {}
        for key in self.observations.keys():
            observations[key] = (
                take_along_axis(self.observations[key][..., -1]).squeeze(-1).long()
            )

        return EpisodicDictRolloutBufferSamples(
            memory=memory,
            observations=observations,
            actions=take_along_axis(self.actions[..., -1]),
            old_values=take_along_axis(self.values),
            old_log_prob=take_along_axis(self.log_probs),
            advantages=take_along_axis(self.advantages),
            returns=take_along_axis(self.returns),
        )


def test_get_samples():
    from envs.memory_planning_game import MemoryPlanningGame

    env = MemoryPlanningGame()
    # Create an instance of EPNDictRolloutBuffer
    buffer_size = 5
    observation_space = env.observation_space
    action_space = env.action_space
    device = "cpu"
    gae_lambda = 0.95
    gamma = 0.99
    n_envs = 3
    buffer = EPNDictRolloutBuffer(
        buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs
    )

    # Generate some sample batch_inds
    batch_inds = np.repeat(np.arange(buffer_size)[None, :], n_envs, axis=0)
    for i in buffer.get():
        print(i)


# Run the test
if __name__ == "__main__":
    test_get_samples()
