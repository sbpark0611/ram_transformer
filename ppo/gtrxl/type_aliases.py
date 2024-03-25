from typing import List, NamedTuple, Union

import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import TensorDict

class XLRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    mems: List[Union[th.FloatTensor, np.ndarray]]
    episode_starts: th.Tensor


class XLDictRolloutBufferSamples(XLRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    mems: List[Union[th.FloatTensor, np.ndarray]]
    episode_starts: th.Tensor
