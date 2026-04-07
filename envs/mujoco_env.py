import gymnasium as gym
import numpy as np

def make_env(env_name="HalfCheetah-v5", render=False, max_episode_steps=1000):
    if render:
        env = gym.make(env_name, render_mode="human", max_episode_steps=max_episode_steps)
    else:
        env = gym.make(env_name, max_episode_steps=max_episode_steps)
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.asarray(obs, dtype=np.float32),
        env.observation_space,
    )  # 将观察转换为float32类型
    return env
