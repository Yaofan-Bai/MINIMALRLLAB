import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import gymnasium as gym
from algorithms.actor_critic import ActorCritic
from algorithms.obs_norm import ObsNormalizer
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--best", action="store_true", help="优先加载best模型与归一化参数")
group.add_argument("--latest", action="store_true", help="优先加载最新模型与归一化参数")
args = parser.parse_args()

# 创建环境并启用渲染（用于直接观看）
env = gym.make("HalfCheetah-v5", render_mode="human")

# 获取环境的状态和动作维度
obs_dim = env.observation_space.shape[0]
obs_norm = ObsNormalizer(obs_dim)
obs_norm_path = "obs_norm_stats.npz"
best_norm_path = "obs_norm_best.npz"
checkpoint_path = "ppo_checkpoint.pth"
best_checkpoint_path = "ppo_checkpoint_best.pth"
policy_path = "ppo_policy.pth"
best_policy_path = "ppo_policy_best.pth"

prefer_best = args.best or not args.latest

if prefer_best and os.path.exists(best_norm_path):
    obs_norm.load(best_norm_path)
elif (not prefer_best) and os.path.exists(obs_norm_path):
    obs_norm.load(obs_norm_path)
elif os.path.exists(best_norm_path):
    obs_norm.load(best_norm_path)
elif os.path.exists(obs_norm_path):
    obs_norm.load(obs_norm_path)
elif os.path.exists(best_checkpoint_path) or os.path.exists(checkpoint_path):
    # PyTorch 2.6+ defaults weights_only=True; set False if checkpoint is trusted
    if prefer_best and os.path.exists(best_checkpoint_path):
        ckpt_path = best_checkpoint_path
    elif (not prefer_best) and os.path.exists(checkpoint_path):
        ckpt_path = checkpoint_path
    else:
        ckpt_path = best_checkpoint_path if os.path.exists(best_checkpoint_path) else checkpoint_path
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "obs_norm_state_dict" in checkpoint:
        obs_norm.load_state_dict(checkpoint["obs_norm_state_dict"])
    else:
        print("警告：checkpoint中没有obs_norm_state_dict，将使用默认归一化参数。")
else:
    print("警告：未找到obs_norm_stats.npz或checkpoint，将使用默认归一化参数。")
act_dim = env.action_space.shape[0]

# 创建策略网络并加载权重
policy = ActorCritic(obs_dim, act_dim).to(device)
if prefer_best and os.path.exists(best_policy_path):
    load_policy_path = best_policy_path
elif (not prefer_best) and os.path.exists(policy_path):
    load_policy_path = policy_path
else:
    load_policy_path = best_policy_path if os.path.exists(best_policy_path) else policy_path
policy.load_state_dict(torch.load(load_policy_path, map_location=device))
print(f"加载策略权重：{load_policy_path}")
policy.eval()  # 设置为评估模式

# 评估模型
done = False
episode_reward = 0.0
for episode in range(5):  # 评估5个episode
    obs, _ = env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        obs = obs_norm.normalize(obs, update=False)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, _ = policy.forward(obs_tensor)
        action_np = mean.detach().cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        episode_reward += reward
        time.sleep(0.05)  # 添加适当的延迟以便观察
    print(f"Episode {episode + 1}: Total reward: {episode_reward}")
env.close()
