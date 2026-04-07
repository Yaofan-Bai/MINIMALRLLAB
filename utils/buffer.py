import torch
import numpy as np

class RolloutBuffer:
    #用来存储强化学习中每一步的状态、动作、奖励、是否结束、状态值和动作概率等信息的缓冲区类
    def __init__(self, rollout_size, obs_dim,act_dim,device):
        self.rollout_size = rollout_size    #缓冲区的大小，即可以存储多少步的信息
        self.device = device
        self.ptr = 0                        #指针，表示当前存储的位置
        self.states = []                    #用来存储状态的列表
        self.actions = []                   #用来存储动作的列表
        self.rewards = []                   #用来存储奖励的列表
        self.dones = []                     #用来存储是否结束的列表
        self.values = []                    #用来存储状态值的列表
        self.log_probs = []                 #用来存储动作概率的列表
        self.advantages = []                #用来存储优势函数的列表
        self.returns = []                   #用来存储回报的列表
    def add(self, state, action, reward, done, value, log_prob):
        #将每一步的信息添加到缓冲区中，本质是赋值操作
        assert self.ptr < self.rollout_size#确保指针没有超过缓冲区的大小
        self.states.append(state)           #将 state 存入状态列表
        self.actions.append(action)         #将 action 存入动作列表
        self.rewards.append(reward)         #将 reward 存入奖励列表
        self.dones.append(done)             #将 done 存入是否结束列表
        self.values.append(value)           #将 value 存入状态值列表
        self.log_probs.append(log_prob)     #存储动作概率
        self.ptr += 1                       #指针加1，准备存储下一步的信息
    def compute_returns_and_advantages(self, last_value, gamma=0.98, lam=0.92):
        #计算回报和优势函数的函数，输入是最后一步的状态值、折扣因子和GAE参数
        gae = 0.0                                                               #初始化gae为0
        values_np = np.array(self.values, dtype=np.float32)                     #将状态值列表转换为numpy数组，方便计算
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)
        advantages = np.zeros(self.ptr, dtype=np.float32)
        returns = np.zeros(self.ptr, dtype=np.float32)
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:                                            #如果是最后一步
                next_value = last_value                                         #下一步的状态值就是输入的last_value
            elif dones_np[step]:                                                   #如果当前步是结束状态
                next_value = 0.0                                                   #下一步的状态值为0
            else:                                                                      #否则
                next_value = values_np[step + 1]                                #下一步的状态值就是缓冲区中下一步的值
            delta = rewards_np[step] + gamma * next_value * (1.0 - dones_np[step]) - values_np[step] #计算TD误差
            gae = delta + gamma * lam * (1.0 - dones_np[step]) * gae             #更新gae
            advantages[step] = gae                                              #对齐时间步写入
            returns[step] = gae + values_np[step]                               #计算回报并存储在回报列表中
        adv_mean = np.mean(advantages) #计算优势函数的均值
        adv_std = np.std(advantages)   #计算优势函数的标准差
        self.advantages = (advantages - adv_mean) / (adv_std + 1e-8) #标准化优势函数，提升训练稳定性
        self.returns = returns
    def clear(self):
        #清空缓冲区，重置指针和所有列表
        self.ptr = 0
        self.states.clear()    #将状态列表清零
        self.actions.clear()    #将动作列表清零
        self.rewards.clear()    #将奖励列表清零
        self.dones.clear()      #将是否结束列表清零
        self.values.clear()     #将状态值列表清零
        self.log_probs.clear()  #将动作概率列表清零
        self.advantages = []    #将优势函数列表清零
        self.returns = []       #将回报列表清零
    
    def get_batch(self, batch_size):
        #获取批次数据的函数，输入是批次大小，输出是状态、动作、动作概率、回报和优势函数的GPU张量
        advantages = np.array(self.advantages, dtype=np.float32)
        indices = np.arange(self.ptr) #生成一个从0到ptr-1的索引数组
        np.random.shuffle(indices)    #打乱索引数组，确保每次训练的数据顺序不同
        for start in range(0, self.ptr, batch_size): #按照批次大小分割数据
            end = start + batch_size #计算当前批次的结束索引
            batch_indices = indices[start:end] #获取当前批次的索引
            states = torch.tensor(np.array(self.states, dtype=np.float32))[batch_indices].to(self.device)
            actions = torch.tensor(np.array(self.actions, dtype=np.float32))[batch_indices].to(self.device)
            log_probs = torch.tensor(np.array(self.log_probs, dtype=np.float32))[batch_indices].to(self.device)
            returns = torch.tensor(np.array(self.returns, dtype=np.float32))[batch_indices].to(self.device)
            adv = torch.tensor(advantages)[batch_indices].to(self.device)
            yield (states, actions, log_probs, returns, adv)#将标准化后的优势函数转换为GPU张量并返回
            
    def __len__(self):
        #返回缓冲区中存储的数据量，即ptr的值
        return self.ptr
