import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.mujoco_env import make_env
import torch
from utils.buffer import RolloutBuffer
from models.actor_critic import ActorCritic
from algorithms.PPO import PPO
from algorithms.obs_norm import ObsNormalizer
from torch.utils.tensorboard import SummaryWriter
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #设置设备为GPU（如果可用）或CPU
def train(resume: bool = True, checkpoint_path: str = "ppo_checkpoint.pth", rollout_size: int = 2048, max_steps: int = 2048):
    env = make_env() #创建环境
    obs_dim = env.observation_space.shape[0] #获取观察空间的维度，即状态的维度
    obs_norm = ObsNormalizer(obs_dim)
    act_dim = env.action_space.shape[0] #获取动作空间的维度，即动作的维度
    log_dir = os.path.join("logs", "ppo_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    #rollout_size = 2048 #每轮训练收集的时间步数
    #初始化RolloutBuffer
    buffer = RolloutBuffer(rollout_size, obs_dim, act_dim, device) #创建一个RolloutBuffer对象，参数为buffer的大小、状态维度、动作维度和设备
    #初始化网络
    policy = ActorCritic(obs_dim, act_dim).to(device)
    ppo = PPO(policy)
    #尝试从checkpoint继续训练（可通过resume控制）
    start_step = 0
    obs_norm_path = "obs_norm_stats.npz"
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "obs_norm_state_dict" in checkpoint:
            obs_norm.load_state_dict(checkpoint["obs_norm_state_dict"])
        elif os.path.exists(obs_norm_path):
            obs_norm.load(obs_norm_path)
            print("checkpoint里没有obs_norm_state_dict，已从obs_norm_stats.npz加载归一化参数")
        else:
            print("警告：checkpoint里没有obs_norm_state_dict，也未找到obs_norm_stats.npz")
        start_step = checkpoint.get("total_steps", checkpoint.get("iteration", 0)) + 1
        print(f"从checkpoint继续训练：{checkpoint_path}，起始Step={start_step}")
    elif resume and os.path.exists("ppo_policy.pth"):
        #兼容仅有模型参数的旧文件
        policy.load_state_dict(torch.load("ppo_policy.pth", map_location=device))
        if os.path.exists(obs_norm_path):
            obs_norm.load(obs_norm_path)
            print("已从obs_norm_stats.npz加载归一化参数")
        else:
            print("警告：未找到obs_norm_stats.npz，使用默认归一化参数")
        print("仅加载ppo_policy.pth继续训练（无优化器状态）")
    elif not resume:
        print("resume=False：忽略现有checkpoint/模型文件，从头开始训练")
    #训练参数
    #max_steps = 2048000 #最大训练步数

    #开始训练
    total_time = 0.0 #记录总训练时间
    print("开始训练...")
    print(f"设备: {device}")
    iteration = 0 #记录训练迭代次数
    total_steps = start_step #记录环境真实交互步数
    state, _ = env.reset() #重置环境，获取初始状态
    iteration_reward = 0.0 #记录当前轮的总奖励
    iteration_ep_count = 0 #当前rollout内完成的episode数量
    done = False
    best_score = -float("inf")
    best_path = "ppo_policy_best.pth"
    best_norm_path = "obs_norm_best.npz"
    best_ckpt_path = "ppo_checkpoint_best.pth"
    time_start = time.time() #记录训练开始时间
    while total_steps < max_steps:
        if buffer.ptr < rollout_size:
            state = obs_norm.normalize(state, update=True)
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device) #将状态转换为Tensor
            action, log_prob, value, entropy = policy.act(state_tensor) #通过策略网络获取动作、log_prob、entropy、value
            action_np = action.detach().cpu().numpy() #从action获取动作并转换为numpy数组
            value_np = value.detach().cpu().numpy() #从value获取状态值并转换为numpy数组
            log_prob_np = log_prob.detach().cpu().numpy() #从log_prob获取动作概率并转换为numpy数组
            next_state, reward, terminated, truncated, _ = env.step(action_np) #与环境交互，获取下一状态、奖励、是否结束等信息
            iteration_reward += reward #累积奖励
            done = terminated or truncated  # 判断是否结束
            buffer.add(state, action_np, reward, done, value_np, log_prob_np) #将当前时间步的信息存储到buffer中
            if done:
                next_state, _ = env.reset()  # 如果结束了，下一状态设为None
                iteration_ep_count += 1
            state = next_state #更新状态
            total_steps += 1 #记录真实环境交互步数
        else:
            iteration += 1 #训练迭代次数加1
            with torch.no_grad():
                if done:
                    last_value_np = 0.0
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                    _, last_value = policy.forward(state_tensor)
                    last_value_np = last_value.detach().cpu().numpy() #将最后一步的状态值转换为numpy数组
            buffer.compute_returns_and_advantages(last_value_np) #计算回报和优势函数
            ppo.update(buffer)
            grad_norm_sq = 0.0
            for param in policy.parameters():
                if param.grad is not None:
                    grad_norm_sq += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm_sq ** 0.5
            writer.add_scalar('gradient_norm', grad_norm, iteration)
            writer.add_scalar("reward_rollout_sum", iteration_reward, iteration)
            if iteration_ep_count > 0:
                writer.add_scalar("reward_rollout_mean_ep", iteration_reward / iteration_ep_count, iteration)
            writer.add_scalar("policy_loss", ppo.policy_loss, iteration) 
            writer.add_scalar("value_loss", ppo.value_loss, iteration)
            writer.add_scalar("entropy", ppo.entropy, iteration)
            writer.add_scalar("approx_kl", ppo.approx_kl, iteration)  

            # 记录当前rollout得分，并保存最佳模型
            if iteration_ep_count > 0:
                current_score = iteration_reward / iteration_ep_count
            else:
                current_score = iteration_reward
            if current_score > best_score:
                best_score = current_score
                if iteration % 5 == 0:
                    torch.save(policy.state_dict(), best_path)
                    obs_norm.save(best_norm_path)
                    torch.save({
                        "iteration": iteration,
                        "total_steps": total_steps,
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": ppo.optimizer.state_dict(),
                        "obs_norm_state_dict": obs_norm.state_dict(),
                    }, best_ckpt_path)
                    print(f"保存best policy: score={best_score:.4f}, iteration={iteration}")

            buffer.clear()  #清空buffer，为下一轮训练做准备
            elapsed_time = time.time() - time_start #计算本轮训练耗时
            total_time += elapsed_time
            time_start = time.time()
            if iteration % 10 == 0:
                print(f"Iteration {iteration} finished, Reward(sum): {iteration_reward}, Episodes: {iteration_ep_count}, Time Elapsed: {elapsed_time:.2f} seconds")
                torch.save({
                    "iteration": iteration,
                    "total_steps": total_steps,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict()
                }, checkpoint_path)
            iteration_reward = 0.0 #重置当前轮的总奖励
            iteration_ep_count = 0
    writer.close()
    torch.save(policy.state_dict(), "ppo_policy.pth") #保存模型参数
    obs_norm.save("obs_norm_stats.npz") #保存观察归一化的统计信息
    torch.save({
        "iteration": iteration,
        "total_steps": total_steps,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict()
    }, checkpoint_path)
    print("训练耗时：{:.2f} seconds".format(total_time))
    env.close() #关闭环境
