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
from ppo.trxl_rnn.model import TransfoXLModelOutput, xlTEM
from ppo.trxl_rnn.type_aliases import TensorDict, XLState


class XLActorCriticPolicy(ActorCriticPolicy):
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
        self.model = xlTEM(model_config)

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
        xlTEM: xlTEM,
        act: th.Tensor,
        obs: th.Tensor,
        xl_state: XLState,
        goal: th.Tensor,
        rew: th.Tensor,
        mask_last_obs: bool = False,
    ) -> TransfoXLModelOutput:
        if goal is not None:
            act, obs, goal = map(lambda x: x.long(), (act, obs, goal))
            if len(act.shape) == 1:
                act = act[:, None]
                obs = obs[:, None]
                goal = goal[:, None]
                rew = rew[:, None]
        return xlTEM(
            act,
            obs,
            xl_state.rnn_hidden,
            xl_state.mem,
            mask_last_obs,
            goal,
            rew,
        )

    def extract_features(
        self, obs: TensorDict
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        cue = obs["position"]
        goal = obs["goal"]
        act = obs["prev_action"]
        rew = obs["reward"]
        return cue, goal, act, rew

    def forward(
        self,
        obs: TensorDict,
        xl_state: XLState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, XLState]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation. Observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        cue, goal, act, rew = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        outputs = self._process_sequence(self.model, act, cue, xl_state, goal, rew)
        state = XLState(outputs.mems, outputs.rnn_hidden)
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
        return actions, values, log_prob, state

    def predict_next_token(
        self,
        obs: th.Tensor,
        act: th.Tensor,
        xl_state: XLState,
    ) -> Tuple[th.Tensor, XLState]:
        # cue, goal, act, rew = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        outputs = self._process_sequence(
            self.model, act, obs, xl_state, None, None, True
        )
        state = XLState(outputs.mems, outputs.rnn_hidden)
        latent = outputs.last_hidden_state
        # Evaluate the values for the given observations
        logits = self.next_obs_predict_net(latent[:, -1])
        return logits, state

    def output_hiddens(
        self,
        obs: th.Tensor,
        act: th.Tensor,
        xl_state: XLState,
    ) -> Tuple[th.Tensor, XLState]:
        # cue, goal, act, rew = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        outputs = self._process_sequence(
            self.model, act, obs, xl_state, None, None, True
        )
        state = XLState(outputs.mems, outputs.rnn_hidden)
        return outputs, state

    def get_distribution(
        self,
        obs: th.Tensor,
        xl_state: XLState,
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, XLState]:
        """
        Get the current policy distribution given the observations.
        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        cue, goal, act, rew = self.extract_features(obs)
        outputs = self._process_sequence(self.model, act, cue, xl_state, goal, rew)
        latent_pi = outputs.last_hidden_state
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        rnn_hidden = outputs.rnn_hidden
        mems = outputs.mems
        xl_state = XLState(mem=mems, rnn_hidden=rnn_hidden)
        dist = self._get_action_dist_from_latent(latent_pi)
        return dist, xl_state

    def predict_values(
        self,
        obs: th.Tensor,
        xl_state: XLState,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        cue, goal, act, rew = self.extract_features(obs)
        outputs = self._process_sequence(self.model, act, cue, xl_state, goal, rew)
        latent_pi = outputs.last_hidden_state
        if self.latent_detach:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = latent_pi
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        act: th.Tensor,
        xl_state: XLState,
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
        cue, goal, pre_act, rew = self.extract_features(obs)
        outputs = self._process_sequence(self.model, pre_act, cue, xl_state, goal, rew)
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
        observation: th.Tensor,
        episode_starts: th.Tensor,
        xl_state: XLState,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, XLState]:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, xl_state = self.get_distribution(
            observation,
            xl_state,
            episode_starts,
        )
        return distribution.get_actions(deterministic=deterministic), xl_state

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        xl_state: XLState,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[XLState]]:
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
            actions, xl_state = self._predict(
                observation,
                episode_starts,
                xl_state,
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

        return actions, xl_state
