import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.mujoco_env import make_env


env = make_env()

obs, _ = env.reset()

print("obs shape:", obs.shape)
print("action shape:", env.action_space.shape)

for _ in range(1000):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()