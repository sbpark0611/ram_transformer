import argparse
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
from ppo.trxl_rnn import TransfoXLPPO
from ppo.trxl_rnn.callbacks import EvalPretrainCallback
from utils.misc import linear_schedule


@hydra.main(config_path="./ppo/trxl", config_name="pre_config")
def main(cfg: DictConfig) -> None:
    env_kwargs = dict(cfg.env)
    if cfg.env.num_maze > 0:
        env_kwargs["maps"] = MemoryPlanningGame.generate_worlds(**env_kwargs)
    else:
        env_kwargs["maps"] = None
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
    eval_env2 = VecNormalize(eval_env2, norm_obs=False, norm_reward=True)

    callback = CallbackList(
        [
            EvalPretrainCallback(
                eval_env,
                cfg.env.max_episode_steps,
                "test",
                rate_map_freq=cfg.env.rate_map_freq,
                random_policy=cfg.env.random_policy,
            ),
            # EvalPretrainCallback(eval_env2, config.max_episode_steps, "train"),
        ]
    )
    run = wandb.init(
        dir=cfg.logger.run_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.logger.proj_name,
        group=cfg.logger.group_name,
        sync_tensorboard=True,
        save_code=True,
    )
    policy_kwargs = dict(
        ortho_init=False,
        shared=cfg.model.shared_vf,
        net_arch=dict(pi=[], vf=[]),
        model_config=cfg.model,
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

    model = TransfoXLPPO("TransfoXLPolicy", env, **ppo_kwargs)
    print("tot num of params:", sum(p.numel() for p in model.policy.parameters()))

    model.learn_predict_next_token(
        total_timesteps=cfg.optim.epochs * cfg.env.n_envs * cfg.env.max_episode_steps,
        callback=callback,
        random_policy=cfg.env.random_policy,
    )

    local_path = osp.join("/".join(run.dir.split("/")[:-3]), "local_saves_wandb")
    Path(osp.join(local_path, run.id)).mkdir(parents=True, exist_ok=True)
    model.save(osp.join(local_path, run.id, "model"))
    env.save(osp.join(local_path, run.id, "venv_norm.pkl"))
    if cfg.logger.upload_to_wandb:
        wandb.save(osp.join(local_path, run.id, "model.zip"))
        wandb.save(osp.join(local_path, run.id, "venv_norm.pkl"))
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./")
    parser.add_argument("--proj_name", type=str, default="MemoryPlanningPretrain")
    parser.add_argument("--group_name", type=str, default="XLPPO")
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--maze_size", type=int, default=6)
    parser.add_argument("--num_maze", type=int, default=32)
    parser.add_argument("--max_episode_steps", type=int, default=2048)
    parser.add_argument("--n_envs", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--tgt_len", "-K", type=int, default=32)
    parser.add_argument("--mem_len", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=0.25)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--rate_map_freq", type=int, default=10)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_episode", type=int, default=512)
    parser.add_argument("--attn_type", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_a", type=int, default=5)
    parser.add_argument(
        "--ffn_act_ftn",
        type=str,
        default="nmda",
        choices=["relu", "swish", "gelu", "nmda", "linear"],
    )
    parser.add_argument(
        "--rnn_act_ftn",
        type=str,
        default="tanh",
        choices=["relu", "sigmoid", "tanh", "linear"],
    )
    parser.add_argument(
        "--random_policy",
        type=str,
        default="random",
        choices=["random", "levy", "model", "bound"],
    )
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log_image_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--from_scratch", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--shared_vf", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--linear_schedule", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--label_duplication", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--upload_to_wandb", "-w", default=False, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    main(config=vars(args))
