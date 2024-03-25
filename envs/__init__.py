from gymnasium.envs.registration import register

register(
    id="MemoryPlanningGame-v0",
    entry_point="envs:MemoryPlanningGame",
    max_episode_steps=300,
)
