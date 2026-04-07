import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.mujoco_env import make_env
import torch
from algorithms.buffer_optimized import RolloutBuffer
from algorithms.actor_critic import ActorCritic
from algorithms.PPO_optimized import PPO
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    writer = SummaryWriter("ppo_logs_optimized")
    
    # 初始化网络和算法
    policy = ActorCritic(obs_dim, act_dim).to(device)
    ppo = PPO(policy, n_epochs=4, batch_size=128)  # 多 epoch 和 mini-batch 更新
    
    # 训练参数
    max_episodes = 5000
    max_steps_per_episode = 2048
    steps_per_rollout = 4096  # 改为固定每次收集4096步数据后更新
    
    # 初始化 buffer
    buffer = RolloutBuffer(obs_dim, act_dim, device)
    
    total_steps = 0
    update_count = 0
    steps_in_rollout = 0
    
    print("开始训练（优化版本）...")
    print(f"设备: {device}")
    print(f"Buffer 类型: 动态列表（减少内存浪费）")
    print(f"PPO 参数: n_epochs=4, batch_size=128")
    print()
    
    for episode in range(max_episodes):
        start_time = time.time()
        state, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0
        episode_length = 0
        
        # 采集数据并添加到 buffer
        while not done and steps < max_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            action, log_prob, value, entropy = policy.act(state_tensor)
            action_np = action.detach().cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            # 直接添加 numpy 数据到 buffer（不立即转移到 GPU）
            # value需要detach并转为标量，避免计算图过长
            buffer.add(state, action_np, reward, done, value.detach().item(), log_prob.sum().item())
            
            state = next_state
            steps += 1
            episode_reward += reward
            episode_length += 1
            total_steps += 1
        
        # 计算 bootstrap value
        with torch.no_grad():
            if done:
                last_value = 0.0
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                _, last_value = policy.forward(state_tensor)
                last_value = last_value.squeeze(-1).item()
        
        buffer.compute_returns_and_advantages(last_value)
        steps_in_rollout += episode_length  # 累积这个rollout的步数
        
        # 保存buffer大小用于日志输出
        buffer_size = len(buffer)
        
        # 每累积固定步数进行一次更新
        if steps_in_rollout >= steps_per_rollout and buffer_size > 0:
            update_start = time.time()
            ppo.update(buffer)
            update_time = time.time() - update_start
            
            writer.add_scalar("training/policy_loss", ppo.policy_loss, update_count)
            writer.add_scalar("training/value_loss", ppo.value_loss, update_count)
            writer.add_scalar("training/entropy", ppo.entropy, update_count)
            writer.add_scalar("timing/update_time", update_time, update_count)
            
            buffer.clear()
            steps_in_rollout = 0  # 重置步数计数器
            update_count += 1
        
        elapsed_time = time.time() - start_time
        
        writer.add_scalar("episode/reward", episode_reward, episode)
        writer.add_scalar("episode/length", episode_length, episode)
        writer.add_scalar("episode/step_time", elapsed_time / episode_length, episode)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:5d}/{max_episodes} | "
                  f"Reward: {episode_reward:8.2f} | Length: {episode_length:4d} | "
                  f"Buffer: {buffer_size:5d} | Steps_Rollout: {steps_in_rollout:5d} | Time: {elapsed_time:.3f}s")
    
    writer.close()
    torch.save(policy.state_dict(), "ppo_policy_optimized.pth")
    env.close()
    print("\n训练完成！")
    print(f"总步数: {total_steps}")
    print(f"更新次数: {update_count}")

if __name__ == "__main__":
    train()
