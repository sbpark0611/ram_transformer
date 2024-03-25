import math
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization,
)
from torch.distributions import Categorical

from utils.misc import eval_memory_correct, levy_walk
from ppo.trxl_rnn.type_aliases import XLState


def evaluate_pretrain(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_rollout_steps: int = 2048,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    random_policy="levy",
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    buffer = next_token_prediction_rollout(model, env, n_rollout_steps, random_policy)

    visited_tot = 0
    visited_correct = 0
    unvisited_tot = 0
    unvisited_correct = 0
    _tot = 0
    _correct = 0

    d_rnn = model.model_config.d_rnn
    device = model.device
    mem_len = model.model_config.mem_len
    tgt_len = model.model_config.tgt_len

    _last_state = XLState(
        None,
        th.randn((env.num_envs, d_rnn)).to(device) / math.sqrt(d_rnn),
    )
    losses = []
    _criterion = th.nn.CrossEntropyLoss()
    chunks = [
        [i, min(i + tgt_len, n_rollout_steps)]
        for i in range(0, n_rollout_steps, tgt_len)
    ]

    for start, stop in chunks:
        obs = th.as_tensor(buffer["observations"][:, start:stop]).long().to(device)
        next_obs = th.as_tensor(buffer["observations"][:, stop]).long().to(device)
        act = th.as_tensor(buffer["actions"][:, start:stop]).long().to(device)
        logits, xl_state = model.predict_next_token(obs, act, _last_state)

        loss = _criterion(logits, next_obs)
        losses.append(loss.item())
        _last_state = xl_state
        ret = eval_memory_correct(
            logits,
            next_obs,
            buffer["pos"],
            start,
            stop,
            mem_len,
        )
        visited_correct += ret["visited_correct"]
        unvisited_correct += ret["unvisited_correct"]
        _correct += ret["correct"]
        visited_tot += ret["visited_tot"].item()
        unvisited_tot += ret["unvisited_tot"].item()
        _tot += env.num_envs
    test_tot_error = 1 - _correct / _tot
    test_working_memory_error = 1 - visited_correct / visited_tot
    test_reference_memory_error = 1 - unvisited_correct / unvisited_tot

    ret = {}
    ret["working_memory_error"] = test_working_memory_error
    ret["reference_memory_error"] = test_reference_memory_error
    ret["tot_error"] = test_tot_error
    ret["loss"] = np.mean(losses)

    return ret


class EvalPretrainCallback(EventCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_rollout_steps: int,
        prefix: str = "test",
        rate_map_freq: int = 10,
        random_policy: str = "levy",  # ["levy", "random", "model"]
        callback_after_eval: Optional[BaseCallback] = None,
        log_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.rate_map_freq = rate_map_freq
        self.n_rollout_steps = n_rollout_steps
        self.random_policy = random_policy
        self.n_epoch = 0
        self.prefix = prefix

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, self.prefix, "evaluations")
        self.log_path = log_path

    def _init_callback(self) -> None:
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self):
        return True

    def _on_rollout_end(self) -> bool:
        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        ret = evaluate_pretrain(
            self.model.policy,
            self.eval_env,
            n_rollout_steps=self.n_rollout_steps,
            deterministic=self.deterministic,
            render=self.render,
            callback=self.callback,
            random_policy=self.random_policy,
            warn=self.warn,
        )
        self.n_epoch += 1

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        for k, v in ret.items():
            self.logger.record("eval/%s_%s" % (self.prefix, k), v)

        if self.n_epoch % self.rate_map_freq == 0:
            _config = self.model.policy.model_config
            L = self.eval_env.envs[0].maze_size
            ret = rate_map(
                self.model.policy,
                self.eval_env,
                self.n_rollout_steps,
                random_policy=self.random_policy,
            )
            m = th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            for key, val in ret.items():
                if "softmax" in key:
                    fig = plt.figure(figsize=(8, 8))
                    for i in range(64):
                        head = i // _config.n_head
                        tmp = val[head]
                        plt.subplot(8, 8, i + 1)
                        img = m(th.from_numpy(tmp[i % 8, :, :].reshape(1, 1, L, L)))
                        plt.imshow(img[0, 0])
                        plt.axis("off")
                    figure = Figure(figure=fig, close=True)
                    self.logger.record(
                        "eval/%s" % key,
                        figure,
                        exclude=("stdout", "log", "json", "csv"),
                    )
                else:
                    fig = plt.figure(figsize=(8, 8))
                    for i in range(64):
                        plt.subplot(8, 8, i + 1)
                        img = m(th.from_numpy(val[i, :, :].reshape(1, 1, L, L)))
                        plt.imshow(img[0, 0])
                        plt.axis("off")
                    figure = Figure(figure=fig, close=True)
                    self.logger.record(
                        "eval/%s" % key,
                        figure,
                        exclude=("stdout", "log", "json", "csv"),
                    )
        self.logger.dump(self.num_timesteps)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


@th.no_grad()
def next_token_prediction_rollout(model, env, n_rollout_steps, random_policy="levy"):
    _config = model.model_config
    n_envs = env.num_envs
    model.set_training_mode(False)
    device = model.device
    buffer = {}

    n_steps = 0

    observations = np.zeros([n_envs, n_rollout_steps + 1], dtype=np.int32)
    if random_policy in ["random", "bound"]:
        actions = np.zeros([n_envs, n_rollout_steps], dtype=np.int32)
        probs = action_probs(env.envs[0].maze_size)
    elif random_policy == "levy":
        actions = levy_walk(n_rollout_steps, n_envs)
    prev_positions = np.zeros([n_envs, n_rollout_steps], dtype=np.float32)
    episode_starts = np.zeros([n_envs, n_rollout_steps], dtype=np.float32)

    _last_obs = env.reset()
    _last_state = XLState(
        None,
        th.randn((n_envs, _config.d_rnn)).to(device) / math.sqrt(_config.d_rnn),
    )
    _last_episode_starts = np.ones((n_envs,), dtype=bool)

    while n_steps < n_rollout_steps:
        observations[:, n_steps] = _last_obs["position"]
        if random_policy == "random":
            act = np.random.randint(env.action_space.n, size=(env.num_envs))
        elif random_policy == "bound":
            m = Categorical(probs[env.env_method("pos_idx")])
            act = m.sample().numpy()
        elif random_policy == "levy":
            act = actions[:, n_steps]
        elif random_policy == "model":
            obs_tensor = obs_as_tensor(_last_obs, device)
            episode_starts = th.tensor(_last_episode_starts).float().to(device)
            actions, _, _, _last_state = model.forward(
                obs_tensor,
                _last_state,
                episode_starts,
                False,
            )
            act = actions.cpu().numpy().flatten()
        _last_obs, rewards, dones, infos = env.step(act)
        if dones.any() and infos[0].get("terminal_observation") is not None:
            terminal_pos = np.array([info["pos"] for info in infos])
            terminal_obs = np.array(
                [info["terminal_observation"]["position"] for info in infos]
            )

        actions[:, n_steps] = act
        prev_positions[:, n_steps] = np.array([info["prev_pos"] for info in infos])
        episode_starts[:, n_steps] = _last_episode_starts
        _last_episode_starts = dones
        n_steps += 1

    observations[:, n_steps] = terminal_obs
    pos = np.concatenate([prev_positions, terminal_pos[..., None]], axis=1)
    pos = th.as_tensor(pos).long().to(device)
    buffer["episode_starts"] = episode_starts
    buffer["observations"] = observations
    buffer["actions"] = actions
    buffer["pos"] = pos

    return buffer


def action_probs(maze_size):
    pos_ind = np.arange(maze_size**2)
    probs = np.ones((maze_size**2, 5), dtype=float)

    probs[pos_ind % maze_size == 0, 3] = 0
    probs[pos_ind % maze_size == maze_size - 1, 2] = 0
    probs[pos_ind // maze_size == 0, 4] = 0
    probs[pos_ind // maze_size == maze_size - 1, 1] = 0
    probs = probs / probs.sum(-1, keepdims=True)
    return th.as_tensor(probs)


@th.no_grad()
def rate_map(model, env, n_rollout_steps, random_policy="levy"):
    ret = next_token_prediction_rollout(model, env, n_rollout_steps * 50, random_policy)
    config = model.model_config
    device = model.device
    L = env.envs[0].maze_size

    _obs, _act, _pos = ret["observations"][:1], ret["actions"][:1], ret["pos"][:1]
    counts = np.zeros((L, L))
    pos_counts = np.zeros((L, L))
    ret = {
        "pos_emb": np.zeros((config.d_embed, L, L)),
    }
    for l in range(1, config.n_layer + 1):
        ret["layer%d/ffn_hid" % l] = np.zeros((config.d_inner, L, L))
        ret["layer%d/softmax" % l] = np.zeros(
            (config.n_head, config.mem_len + config.tgt_len + 1, L, L)
        )

    model.set_training_mode(False)
    _last_state = XLState(
        None,
        th.randn((1, config.d_rnn)).to(device) / math.sqrt(config.d_rnn),
    )
    chunks = [
        [i, min(i + config.tgt_len, n_rollout_steps * 50)]
        for i in range(0, n_rollout_steps * 50, config.tgt_len)
    ]
    # Run through all chunks that we are going to backprop for
    for j, [start, stop] in enumerate(chunks):
        src_x = th.as_tensor(_obs[:, start:stop]).long().to(device)
        a = th.as_tensor(_act[:, start:stop]).long().to(device)
        outputs, _last_state = model.output_hiddens(src_x, a, _last_state)
        if stop >= config.mem_len + config.tgt_len:
            _x, _y = _pos[0, stop] % L, _pos[0, stop] // L
            pos_idx = _pos[0, start + 1 : stop + 1]
            pos_emb = outputs.grid_activations[0, 1:].detach().cpu().numpy()
            for n, _pos_idx in enumerate(pos_idx):
                ret["pos_emb"][:, _pos_idx % L, _pos_idx // L] = (
                    ret["pos_emb"][:, _pos_idx % L, _pos_idx // L] + pos_emb[n]
                )
                pos_counts[_pos_idx % L, _pos_idx // L] = (
                    pos_counts[_pos_idx % L, _pos_idx // L] + 1
                )
            for l in range(1, config.n_layer + 1):
                ffn_hid = outputs.ffn_activations[l - 1][0, -1].detach().cpu().numpy()
                ret["layer%d/ffn_hid" % l][:, _x, _y] = (
                    ret["layer%d/ffn_hid" % l][:, _x, _y] + ffn_hid
                )
                softmax = outputs.attentions[l - 1][0, :, -1].detach().cpu().numpy()
                ret["layer%d/softmax" % l][..., _x, _y] = (
                    ret["layer%d/softmax" % l][..., _x, _y] + softmax
                )
            counts[_x, _y] = counts[_x, _y] + 1

    ret["pos_emb"] = ret["pos_emb"] / pos_counts
    for l in range(1, config.n_layer + 1):
        ret["layer%d/ffn_hid" % l] = ret["layer%d/ffn_hid" % l] / counts
        ret["layer%d/softmax" % l] = ret["layer%d/softmax" % l] / counts
    return ret


# TODO: trainin map with reversed goal for reward seeking.


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 512,
        eval_freq: int = 5000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = False,
        render: bool = False,
        verbose: int = 1,
        prefix: str = "eval",
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.prefix = prefix

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_steps2goal = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, num_steps2goal = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_steps2goal.append(num_steps2goal)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    steps2goal=self.evaluations_steps2goal,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                print(f"Avg. steps to goal: {self.evaluations_steps2goal}")
            # Add to current Logger
            self.logger.record(f"{self.prefix}/mean_reward", float(mean_reward))
            self.logger.record(f"{self.prefix}/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self.prefix}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )

            plt.plot(np.arange(len(num_steps2goal)) + 1, num_steps2goal)
            plt.xlabel("N-th task")
            plt.ylabel("Avg. No. of steps to goal")
            figure = plt.gcf()
            self.logger.record(
                "trajectory/steps2goal",
                Figure(figure, close=True),
                exclude=("stdout", "log", "json", "csv"),
            )
            plt.close()
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    _config = model.policy.model_config
    device = model.policy.device
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    xl_state = XLState(
        None,
        th.randn((n_envs, _config.d_rnn)).to(device) / math.sqrt(_config.d_rnn),
    )
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    num_steps2goal = []
    while (episode_counts < episode_count_targets).any():
        actions, xl_state = model.predict(
            observations,  # type: ignore[arg-type]
            xl_state,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions.flatten())
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    num_steps2goal.append(info["steps"])
        if dones.any():
            xl_state = XLState(
                None,
                th.randn((n_envs, _config.d_rnn)).to(device) / math.sqrt(_config.d_rnn),
            )
        else:
            _init_hidden = th.randn((n_envs, _config.d_rnn)).to(device) / math.sqrt(
                _config.d_rnn
            )
            rew = th.tensor(rewards).float().to(device)[:, None]
            prev_hidden = (1.0 - rew) * xl_state.rnn_hidden + rew * _init_hidden
            xl_state = XLState(xl_state.mem, prev_hidden)

        observations = new_observations

        if render:
            env.render()

    df = pd.DataFrame(num_steps2goal)
    average_values = df.mean(axis=0, skipna=True).values
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths, average_values
    return mean_reward, std_reward
