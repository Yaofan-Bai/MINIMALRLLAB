import torch
import numpy as np

class RolloutBuffer:
    """优化的 Rollout Buffer - 使用列表存储，只在训练时转移到 GPU"""
    def __init__(self, obs_dim, act_dim, device):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        
        # 使用列表存储数据，避免预分配浪费
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        
    def add(self, state, action, reward, done, value, log_prob):
        """添加一条数据，保持 CPU 上的存储"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.ptr += 1
        
    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """计算回报和优势函数"""
        gae = 0
        # 统一转换：确保last_value和values都是标量
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.detach().cpu().item()
        # 直接转为numpy数组（values已经都是标量了）
        values_np = np.array(self.values, dtype=np.float32)
        
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_value = last_value
            else:
                next_value = values_np[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * (1 - self.dones[step]) - values_np[step]
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            self.advantages.append(gae)
            self.returns.append(gae + values_np[step])
        
        # 反转列表回到原始顺序
        self.advantages.reverse()
        self.returns.reverse()
    
    def get_batch(self, batch_size=None):
        """获取批次数据，返回 GPU 张量"""
        if batch_size is None:
            batch_size = self.ptr
        
        # 转为 numpy 以便进行标准化（仅一次）
        advantages_np = np.array(self.advantages)
        adv_mean = advantages_np.mean()
        adv_std = advantages_np.std()
        advantages_np = (advantages_np - adv_mean) / (adv_std + 1e-8)
        
        # 转换为 GPU 张量（仅转换一次，所有epoch共享）
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array(self.returns), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages_np, dtype=torch.float32).to(self.device)
        
        # 生成索引并随机打乱（仅一次，所有epoch共享，保证数据一致性）
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        
        # 缓存indices以供多个epoch使用
        self.shuffled_indices = indices
        
        # 分批生成数据
        for start_idx in range(0, self.ptr, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices]
            )
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return self.ptr
