from typing import NamedTuple

import torch as th
from stable_baselines3.common.type_aliases import TensorDict


class EpisodicRolloutBufferSamples(NamedTuple):
    memory: TensorDict
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class EpisodicDictRolloutBufferSamples(EpisodicRolloutBufferSamples):
    memory: TensorDict
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
