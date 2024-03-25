from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from ppo.epn.model import EPN
from ppo.epn.type_aliases import TensorDict


class ActorCriticEpnPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = False,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        epn_kwargs: Optional[Dict[str, Any]] = {
            "embedding_size": 16,
            "num_heads": 1,
            "hidden_size": 64,
            "num_iterations": 1,
        },
    ):
        self.output_dim = epn_kwargs["hidden_size"]
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self._num_labels = self.observation_space.get("position").n
        self._num_action = self.action_space.n
        self.model = EPN(self._num_labels, self._num_action, **epn_kwargs)
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(
        self,
        memory: Dict,
        obs: TensorDict,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        combined_embedding = self.model(memory, obs)
        # Evaluate the values for the given observations
        values = self.value_net(combined_embedding)
        distribution = self._get_action_dist_from_latent(combined_embedding)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions).squeeze(-1)
        return actions, values, log_prob

    def get_distribution(
        self,
        memory: Dict,
        obs: TensorDict,
    ) -> Distribution:
        combined_embedding = self.model(memory, obs)
        dist = self._get_action_dist_from_latent(combined_embedding)
        return dist

    def predict_values(
        self,
        memory: Dict,
        obs: TensorDict,
    ) -> th.Tensor:
        combined_embedding = self.model(memory, obs)
        return self.value_net(combined_embedding)

    def evaluate_actions(
        self,
        memory: Dict,
        obs: TensorDict,
        act: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Preprocess the observation if needed
        combined_embedding = self.model(memory, obs)
        distribution = self._get_action_dist_from_latent(combined_embedding)
        log_prob = distribution.log_prob(act)
        entropy = distribution.entropy()
        values = self.value_net(combined_embedding)
        return values, log_prob, entropy

    def _predict(
        self,
        memory: Dict,
        obs: TensorDict,
        deterministic: bool = False,
    ) -> th.Tensor:
        distribution = self.get_distribution(memory, obs)
        return distribution.get_actions(deterministic=deterministic)

    def predict(
        self,
        memory: Dict,
        obs: TensorDict,
        deterministic: bool = False,
    ) -> np.ndarray:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # observation, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(
                memory,
                obs,
                deterministic=deterministic,
            )

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        return actions
