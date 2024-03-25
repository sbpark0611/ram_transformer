import gym
from envs.numpad import NumPadEnv
import numpy as np
import pytest
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import FakeImageEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from ppo.trxl_rnn import TransfoXLPPO
from configuration_transfo_xl import TransfoXLConfig


class NumPad2x2(NumPadEnv):
    def __init__(self, size=2, cues=25, n_maps=1, steps_per_episode=1024):
        super().__init__(
            size=size,
            cues=range(cues),
            n_maps=n_maps,
            steps_per_episode=steps_per_episode,
        )


class ToDictWrapper(gym.Wrapper):
    """
    Simple wrapper to test MultInputPolicy on Dict obs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"obs": self.env.observation_space})

    def reset(self):
        return {"obs": self.env.reset()}

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return {"obs": obs}, reward, done, infos


class CartPoleNoVelEnv(CartPoleEnv):
    """Variant of CartPoleEnv with velocity information removed. This task requires memory to solve."""

    def __init__(self):
        super().__init__()
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ]
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    @staticmethod
    def _pos_obs(full_obs):
        xpos, _xvel, thetapos, _thetavel = full_obs
        return xpos, thetapos

    def reset(self):
        full_obs = super().reset()
        return CartPoleNoVelEnv._pos_obs(full_obs)

    def step(self, action):
        full_obs, rew, done, info = super().step(action)
        return CartPoleNoVelEnv._pos_obs(full_obs), rew, done, info


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        # {},
        dict(
            shared=True,
            model_config=TransfoXLConfig(mem_len=32),
        ),
    ],
)
def test_policy_kwargs(policy_kwargs):
    n_envs = 5
    env = make_vec_env(NumPad2x2, n_envs=n_envs)
    policy_kwargs[
        "model_config"
    ].num_labels = 25  # len(env.get_attr("params")[0]["cues"])
    model = TransfoXLPPO(
        "TransfoXLPolicy",
        env,
        batch_size=64,
        n_steps=32,
        seed=0,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=128 * 10)
