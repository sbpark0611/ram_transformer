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
from omegaconf import DictConfig

# from configuration_transfo_xl import TransfoXLConfig
from ppo.gtrxl.model import GTransfoXLModelOutput, GTRxL
from ppo.gtrxl.type_aliases import TensorDict


class GTRxLActorCriticPolicy(ActorCriticPolicy):
    """
    TransformerXL policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic transformerXL
    have the same architecture.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        latent_detach: bool,
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
        shared: bool = True,
        model_config: Optional[DictConfig] = None,
    ):
        self.output_dim = model_config.d_model
        self.model_config = model_config
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

        self.shared = shared
        self.latent_detach = latent_detach
        self.model = GTRxL(model_config)

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
        self._num_labels = self.observation_space.get("position").n
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
        self.next_obs_predict_net = nn.Linear(
            self.mlp_extractor.latent_dim_vf, self._num_labels
        )
        nn.init.normal_(
            self.next_obs_predict_net.weight, 0.0, self.model_config.init_std
        )
        nn.init.constant_(self.next_obs_predict_net.bias, 0.0)
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

    @staticmethod
    def _process_sequence(
        model: GTRxL,
        observation: TensorDict,
        mems: Optional[List[th.FloatTensor]] = None,
    ) -> GTransfoXLModelOutput:
        obs, prev_obs, act, goal = (
            observation["position"],
            observation["prev_position"],
            observation["prev_action"],
            observation["goal"],
        )
        obs, prev_obs, act, goal = map(lambda x: x.long(), (obs, prev_obs, act, goal))
        if len(act.shape) == 1:
            act = act[:, None]
            obs = obs[:, None]
            goal = goal[:, None]
            prev_obs = prev_obs[:, None]
        return model(
            obs,
            prev_obs,
            act,
            goal,
            mems 
        )
    
    def forward(
        self,
        observations: TensorDict,
        mems: List[Union[th.FloatTensor, np.ndarray]],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, List[Union[th.FloatTensor, np.ndarray]]]:
        outputs = self._process_sequence(self.model, observations, mems)
        mems = outputs.mems
        latent_pi = outputs.last_hidden_state
        if self.latent_detach:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = latent_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions).squeeze(-1)
        return actions, values, log_prob, mems
    

    def get_distribution(
        self,
        observations: TensorDict,
        mems: List[Union[th.FloatTensor, np.ndarray]],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, List[Union[th.FloatTensor, np.ndarray]]]:
        """
        Get the current policy distribution given the observations.
        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        outputs = self._process_sequence(self.model, observations, mems)
        latent_pi = outputs.last_hidden_state
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        mems = outputs.mems
        dist = self._get_action_dist_from_latent(latent_pi)
        return dist, mems

    def predict_values(
        self,
        observations: TensorDict,
        mems: List[Union[th.FloatTensor, np.ndarray]]
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        outputs = self._process_sequence(self.model, observations, mems)
        latent_pi = outputs.last_hidden_state
        if self.latent_detach:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = latent_pi
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        observations: TensorDict,
        act: th.Tensor,
        mems: List[Union[th.FloatTensor, np.ndarray]],
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        outputs = self._process_sequence(self.model, observations, mems)
        latent_pi = outputs.last_hidden_state
        if self.latent_detach:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = latent_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(act)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def _predict(
        self,
        observation: TensorDict,
        episode_starts: th.Tensor,
        mems: List[Union[th.FloatTensor, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, List[Union[th.FloatTensor, np.ndarray]]]:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, mems = self.get_distribution(
            observation,
            mems,
            episode_starts,
        )
        return distribution.get_actions(deterministic=deterministic), mems

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        mems: List[Union[th.FloatTensor, np.ndarray]],
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[Union[th.FloatTensor, np.ndarray]]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            episode_starts = th.tensor(episode_start).float().to(self.device)
            actions, mems = self._predict(
                observation,
                episode_starts,
                mems,
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

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, mems
