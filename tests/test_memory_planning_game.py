import gymnasium as gym
from envs.memory_planning_game import MemoryPlanningGame
import numpy as np

game = MemoryPlanningGame(
    maze_size=2,
    num_maze=1,
    max_episode_steps=100,
    target_reward=1.0,
    per_step_reward=0.0,
    num_labels=10,
    render_mode="human",
)
obs = game.reset()
for _ in range(20):
    obs, reward, done, _, info = game.step(np.random.randint(5))
    print(obs)
