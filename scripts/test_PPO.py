import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import torch
from algorithms.actor_critic import ActorCritic
from algorithms.PPO import PPO
from algorithms.buffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境初始化
env = gym.make("Pendulum-v1")  # 使用 Pendulum-v1 环境
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]  # 连续动作空间，使用 shape[0] 来获取维度

# 初始化网络
policy = ActorCritic(obs_dim, act_dim).to(device)
ppo = PPO(policy)

# 初始化RolloutBuffer
buffer = RolloutBuffer(2048, obs_dim, act_dim, device)

# 初始化优化器
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

# 测试 1 轮数据收集
state, _ = env.reset()  # 获取初始状态
done = False
sampled_timesteps = 0  # 记录采样的时间步数
while not done and sampled_timesteps < buffer.rollout_size:
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)  # 将状态转换为 Tensor
    action, log_prob, value, entropy = policy.act(state_tensor)  # 获取动作、log_prob、entropy、value

    action_np = action.detach().cpu().numpy()  # 从action获取动作
    next_state, reward, done, truncated, _ = env.step(action_np)  # 与环境交互
    buffer.add(state, action_np, reward, done, value.detach().squeeze(-1), log_prob.detach())  # 存储到 buffer 中
    state = next_state  # 更新状态
    sampled_timesteps += 1  # 更新采样的时间步数

# 计算优势和回报，Pendulum 没有终止则使用bootstrap
with torch.no_grad():
    if done:
        last_value = torch.tensor(0.0, device=device)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        _, last_value = policy.forward(state_tensor)
        last_value = last_value.squeeze(-1)
buffer.compute_returns_and_advantages(last_value)

# 测试 PPO 更新函数
ppo.update(buffer)
print("PPO update finished!")
