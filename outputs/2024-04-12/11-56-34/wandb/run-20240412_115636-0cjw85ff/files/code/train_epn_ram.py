import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import os.path as osp
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from ppo.epn.callbacks import EvalCallback

from envs.radial_arm_maze import RadialArmMaze
from utils.misc import linear_schedule
from ppo.epn import EPNPPO
from stable_baselines3.common.save_util import load_from_zip_file


@hydra.main(config_path="./ppo/epn", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env_kwargs = dict(cfg.env)
    if cfg.env.num_maze > 0:
        env_kwargs["maps"] = RadialArmMaze.generate_worlds(**env_kwargs)

    env = make_vec_env(RadialArmMaze, n_envs=cfg.env.n_envs, env_kwargs=env_kwargs)
    env = VecNormalize(env, norm_obs=cfg.env.norm_obs, norm_reward=cfg.env.norm_reward)

    eval_env = make_vec_env(
        RadialArmMaze, n_envs=cfg.env.n_envs, env_kwargs=env_kwargs
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=cfg.env.norm_obs, norm_reward=cfg.env.norm_reward
    )
    for env_idx in range(cfg.env.n_envs):
        eval_env.envs[env_idx].test_mode()
    callback_list = [EvalCallback(eval_env, prefix="eval/novel")]

    r_envs = {}
    for i in range(3):
        r_envs[i] = VecNormalize(
            make_vec_env(
                RadialArmMaze, n_envs=cfg.env.n_envs, env_kwargs=env_kwargs
            ), norm_obs=cfg.env.norm_obs, norm_reward=cfg.env.norm_reward
        )
        for env_idx in range(cfg.env.n_envs):
            r_envs[i].envs[env_idx].reverse_mode(case=i+1)
        callback_list.append(EvalCallback(r_envs[i], prefix="eval/reverse%d" %(i+1)))
    callback = CallbackList(callback_list)

    run = wandb.init(
        dir=cfg.logger.run_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.logger.proj_name,
        group=cfg.logger.group_name,
        sync_tensorboard=True,
        save_code=True,
    )
    if cfg.optim.linear_schedule:
        lr = linear_schedule(cfg.optim.learning_rate)
        cr = linear_schedule(cfg.optim.clip_range)
    else:
        lr = cfg.optim.learning_rate
        cr = cfg.optim.clip_range
    set_random_seed(cfg.seed)
    policy_kwargs = dict(net_arch=dict(pi=[], vf=[]), epn_kwargs=dict(cfg.epn))
    ppo_kwargs = dict(cfg.ppo)
    ppo_kwargs["policy_kwargs"] = policy_kwargs
    ppo_kwargs["tensorboard_log"] = osp.join(run.dir, run.id)
    ppo_kwargs["learning_rate"] = lr
    ppo_kwargs["clip_range"] = cr
    model = EPNPPO("ActorCriticEpnPolicy", env, **ppo_kwargs)
    print("tot num of params:", sum(p.numel() for p in model.policy.parameters()))

    model.learn(total_timesteps=cfg.optim.total_timesteps, callback=callback)

    local_path = osp.join("/".join(run.dir.split("/")[:-3]), "local_saves_wandb")
    Path(osp.join(local_path, run.id)).mkdir(parents=True, exist_ok=True)
    model.save(osp.join(local_path, run.id, "model"))
    env.save(osp.join(local_path, run.id, "venv_norm.pkl"))
    if cfg.logger.upload_to_wandb:
        wandb.save(osp.join(local_path, run.id, "model.zip"))
        wandb.save(osp.join(local_path, run.id, "venv_norm.pkl"))
    run.finish()


if __name__ == "__main__":
    main()
