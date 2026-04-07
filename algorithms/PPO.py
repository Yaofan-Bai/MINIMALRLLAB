import torch
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class PPO:
    def __init__(self, policy, lr = 2.0633e-05, clip = 0.1, batch_size =64, n_epochs = 20, target_kl = 0.01):
        self.policy = policy                                    #PPO算法的核心类，包含了策略网络、优化器和剪切参数等信息
        self.optimizer = optim.Adam(policy.parameters(), lr=lr) #使用Adam优化器来更新策略网络的参数，学习率为lr，该学习率为网络梯度下降的步长，较小的学习率可以使训练更稳定，但可能需要更多的训练步骤来收敛
        self.clip_epsilon = clip                                #剪切参数，用于限制策略更新的幅度，防止过大更新导致性能下降
        self.batch_size = batch_size                            #批量大小，用于PPO算法中的mini-batch更新，较大的批量大小可以提高训练效率，但可能需要更多的内存资源
        self.n_epochs = n_epochs                                #每轮数据的更新轮次，增加更新轮次可以提高样本利用率，但可能导致过拟合
        self.target_kl = target_kl                              #KL早停阈值，超过则提前结束更新轮次
        #损失权重
        self.value_coef = 0.58096  #值函数损失系数（提高从0.5到1.0）
        self.entropy_coef = 0.004017  #熵损失系数（鼓励探索）
        #损失记录
        self.policy_loss = 0.0 #记录策略损失的数值，方便后续分析和可视化
        self.value_loss = 0.0 #记录值损失的数值，方便后续分析和可视化
        self.entropy = 0.0 #记录熵的数值，方便后续分析和可视化
        self.approx_kl = 0.0 #记录近似KL散度的数值，方便后续分析和可视化

    def update(self, buffer):
        num_batches = (len(buffer) + self.batch_size - 1) // self.batch_size #计算总batch数，确保所有数据都被使用
        processed_batches = 0
        stop_early = False
        for epoch in range(self.n_epochs):
            epoch_policy_loss = 0.0  #记录每个epoch的策略损失
            epoch_value_loss = 0.0 #记录每个epoch的值损失
            epoch_entropy = 0.0  #记录每个epoch的熵
            epochapprox_kl = 0.0 #记录每个epoch的近似KL散度

            #Mini-batch SGD
            for states, actions, old_log_probs, returns, advantages in buffer.get_batch(self.batch_size):
                #前向传播
                new_log_probs, values, entropy = self.policy.evaluate(states, actions)   #通过策略网络评估当前状态和动作的对数概率、状态值和熵
                values = values.squeeze(-1) #将值网络的输出从形状[batch_size, 1]转换为[batch_size]，方便后续计算
                #PPO 损失
                ratios = torch.exp((new_log_probs - old_log_probs).sum(-1)) #计算新的动作概率与旧的动作概率的比值，用于PPO算法中的剪切操作
                approx_kl = (old_log_probs - new_log_probs).mean() #计算近似的KL散度，用于监控策略更新的幅度，过大的KL散度可能表明策略更新过大，需要调整学习率或剪切参数
                surr1 = ratios * advantages #计算PPO算法中的第一个目标函数，即比值乘以优势函数
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages #计算PPO算法中的第二个目标函数，即剪切后的比值乘以优势函数
                actor_loss = -torch.min(surr1, surr2).mean() #计算策略网络的损失函数，取两个目标函数的最小值并取负号，最后求平均
                critic_loss = (returns - values).pow(2).mean() #计算值网络的损失函数，即回报与状态值的均方误差
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean() #计算总的损失函数，包含策略损失、值损失和熵损失，熵损失用于鼓励策略的探索行为
                #更新策略网络的参数
                self.optimizer.zero_grad() #在更新参数之前先将优化器的梯度清零，避免累积梯度
                loss.backward() #反向传播计算损失函数的梯度
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # 放宽梯度裁剪
                self.optimizer.step() #更新策略网络的参数，根据计算得到的梯度进行优化器的参数更新
                epoch_policy_loss += actor_loss.item() #累积策略损失的数值，方便后续分析和可视化
                epoch_value_loss += critic_loss.item() #累积值损失的数值，方便后续分析和可视化
                epoch_entropy += entropy.mean().item() #累积熵的数值，方便后续分析和可视化
                epochapprox_kl += approx_kl.item() #累积近似KL散度的数值，方便后续分析和可视化
                processed_batches += 1
                if approx_kl.mean().item() > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        denom = max(1, processed_batches)
        self.policy_loss = epoch_policy_loss / denom #计算平均策略损失，方便后续分析和可视化
        self.value_loss = epoch_value_loss / denom #计算平均值损失，方便后续分析和可视化
        self.entropy = epoch_entropy / denom #计算平均熵，方便后续分析和可视化    
        self.approx_kl = epochapprox_kl / denom #计算平均近似KL散度，方便后续分析和可视化
