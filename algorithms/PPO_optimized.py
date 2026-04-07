import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class PPO:
    """优化的 PPO 算法 - 支持多 epoch 更新"""
    def __init__(self, policy, lr=3e-4, clip=0.2, n_epochs=4, batch_size=128):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.clip_epsilon = clip
        self.n_epochs = n_epochs  # 每轮数据的更新轮次
        self.batch_size = batch_size
        
        # 损失权重
        self.value_coef = 1.0  # 值函数损失系数（提高从0.5到1.0）
        self.entropy_coef = 0.005  # 熵损失系数（鼓励探索）
        
        # 损失记录
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy = 0.0
        
    def update(self, buffer):
        """多 epoch 更新"""
        # 计算总batch数
        num_batches = (len(buffer) + self.batch_size - 1) // self.batch_size
        
        for epoch in range(self.n_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            
            # Mini-batch SGD
            for states, actions, old_log_probs, returns, advantages in buffer.get_batch(self.batch_size):
                # 前向传播
                new_log_probs, values, entropy = self.policy.evaluate(states, actions)
                values = values.squeeze(-1)
                
                # PPO 损失
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数损失
                critic_loss = (returns - values).pow(2).mean()
                
                # 熵
                entropy_loss = entropy.mean()
                
                # 总损失（调整权重）
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # 放宽梯度裁剪
                self.optimizer.step()
                
                # 记录损失
                epoch_policy_loss += actor_loss.item()
                epoch_value_loss += critic_loss.item()
                epoch_entropy += entropy_loss.item()
            
            # 平均损失
            self.policy_loss = epoch_policy_loss / num_batches
            self.value_loss = epoch_value_loss / num_batches
            self.entropy = epoch_entropy / num_batches
