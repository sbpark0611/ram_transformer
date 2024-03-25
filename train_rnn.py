import os.path as osp
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from envs.memory_planning_game import MemoryPlanningGame
from ppo.rnn import RecurrentPPO
from ppo.rnn.callbacks import EvalCallback
from utils.misc import linear_schedule


@hydra.main(config_path="./ppo/rnn", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env_kwargs = dict(cfg.env)
    if cfg.env.num_maze > 0:
        env_kwargs["maps"] = MemoryPlanningGame.generate_worlds(**env_kwargs)

    env = make_vec_env(MemoryPlanningGame, n_envs=cfg.env.n_envs, env_kwargs=env_kwargs)
    env = VecNormalize(env, norm_obs=cfg.env.norm_obs, norm_reward=cfg.env.norm_reward)

    eval_env = make_vec_env(
        MemoryPlanningGame, n_envs=cfg.env.n_envs, env_kwargs=env_kwargs
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=cfg.env.norm_obs, norm_reward=cfg.env.norm_reward
    )
    for env_idx in range(cfg.env.n_envs):
        eval_env.envs[env_idx].test_mode()

    eval_env2 = make_vec_env(
        MemoryPlanningGame, n_envs=cfg.env.n_envs, env_kwargs=env_kwargs
    )
    eval_env2 = VecNormalize(
        eval_env, norm_obs=cfg.env.norm_obs, norm_reward=cfg.env.norm_reward
    )
    for env_idx in range(cfg.env.n_envs):
        eval_env2.envs[env_idx].reverse_mode()

    callback = CallbackList(
        [
            EvalCallback(eval_env, prefix="eval/novel"),
            EvalCallback(eval_env2, prefix="eval/reverse"),
        ]
    )
    policy_kwargs = dict(cfg.model)
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

    ppo_kwargs = dict(cfg.ppo)
    ppo_kwargs["policy_kwargs"] = policy_kwargs
    ppo_kwargs["tensorboard_log"] = osp.join(run.dir, run.id)
    ppo_kwargs["learning_rate"] = lr
    ppo_kwargs["clip_range"] = cr

    model = RecurrentPPO("MultiInputLstmPolicy", env, **ppo_kwargs)
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
