import os
import re
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def find_latest_model(checkpoint_dir: str, env_name: str):
    pattern = re.compile(rf"^ppo_{re.escape(env_name)}_(\d{{8}}_\d{{6}})\.zip$")
    candidates = []
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            full_path = os.path.join(checkpoint_dir, fname)
            candidates.append((os.path.getmtime(full_path), full_path, match.group(1)))
    if not candidates:
        raise FileNotFoundError(f"未找到模型文件: {checkpoint_dir}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


# 创建环境并启用渲染
base_env = gym.make("HalfCheetah-v5", render_mode="human", max_episode_steps=1000)
env = DummyVecEnv([lambda: base_env])

# 加载SB3 PPO模型（自动选择最近的checkpoint）
ckpt_dir = os.path.join("checkpoints", "sb3_ppo")
model_path, run_id = find_latest_model(ckpt_dir, "HalfCheetah-v5")
vecnorm_path = os.path.join(ckpt_dir, f"vecnorm_HalfCheetah-v5_{run_id}.pkl")

if os.path.exists(vecnorm_path):
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

model = PPO.load(model_path, env=env)

# 评估模型
obs = env.reset()
done = False
episode_reward = 0.0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step(action)
    done = bool(dones[0])
    episode_reward += float(rewards[0])
    time.sleep(0.05)

print(f"Total reward: {episode_reward}")
env.close()
