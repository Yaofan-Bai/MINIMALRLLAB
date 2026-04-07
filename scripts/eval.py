import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import torch
from envs.mujoco_env import make_env
from algorithms.actor_critic import ActorCritic
from algorithms.PPO import PPO
from trainer.trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(model_path,policy):
    policy.load_state_dict(torch.load(model_path))
    policy.to(device)
    return policy
def evaluate(env,policy,max_steps=1000):
    state, _ = env.reset()  # 获取初始状态
    done = False
    total_reward = 0.0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)  # 将状态转换为 Tensor
        action, log_prob, value, entropy = policy.act(state_tensor)  # 获取动作、log_prob、entropy、value
        action = action.detach().cpu().numpy()  # 从action获取动作
        next_state, reward, terminated, truncated, info = env.step(action)  # 与环境交互
        done = terminated or truncated  # 判断是否结束
        total_reward += reward  # 累积奖励
        state = next_state  # 更新状态
        if done:
            print(f"Episode finished with total reward: {total_reward}")
            break  # 如果环境结束，退出循环
    return total_reward

if __name__ == "__main__":
    env = make_env()
    policy = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    policy.to(device)
    model_path = "ppo_policy.pth"  # 替换为你的模型路径
    policy = load_model(model_path, policy)
    # 评估模型
    reward = evaluate(env, policy)
    print(f"Evaluation reward: {reward}")
