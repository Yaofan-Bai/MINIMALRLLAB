import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__() #调用父类的构造函数，初始化神经网络的参数
        #共享网络结构
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), #输入层到隐藏层的全连接层，输入维度为obs_dim，输出维度为128
            #nn.LayerNorm(256), #层归一化，输入维度为128
            nn.ReLU(), #使用relu激活函数
            nn.Linear(256, 256), #隐藏层到隐藏层的全连接层，输入维度为128，输出维度为128
            #nn.LayerNorm(256), #层归一化，输入维度为128
            nn.ReLU() #使用relu激活函数
        )
        #actor 网络结构
        self.mean = nn.Sequential(
            nn.Linear(256, act_dim),
            #nn.Tanh()
        ) #从共享网络到动作均值的全连接层，输入维度为128，输出维度为act_dim
        #critic 网络结构
        self.value = nn.Linear(256, 1) #从共享网络到状态值的全连接层，输入维度为128，输出维度为1
        #log_std参数
        self.log_std = nn.Parameter(torch.ones(act_dim)*0.2) #定义一个可训练的参数log_std，初始值为0，维度为act_dim
    def forward(self, state):
        #前向传播函数，输入是状态，输出是动作分布和状态值
        x = self.net(state) #通过共享网络
        mean = self.mean(x) #通过动作网络得到动作的均值
        value = self.value(x) #通过值网络得到状态值
        return mean, value
    def act(self, state):
        #根据状态选择动作的函数，输入是状态，输出是动作和动作的对数概率
        mean, value = self.forward(state) #通过前向传播得到动作的均值
        std = torch.exp(self.log_std) #计算动作的标准差，取log_std的指数
        dist = Normal(mean, std) #创建一个正态分布对象，参数为均值和标准差
        action = dist.sample() #从分布中采样一个动作
        log_prob = dist.log_prob(action) #计算动作的对数概率是因为可能有多个动作维度
        entropy = dist.entropy() #计算动作分布的熵，sum(-1)是因为可能有多个动作维度
        return action, log_prob, value, entropy
    def evaluate(self, state, action):
        #评估状态和动作的函数，输入是状态和动作，输出是动作的对数概率、状态值和动作分布的熵
        mean, value = self.forward(state) #通过前向传播得到动作的均值和状态值
        std = torch.exp(self.log_std) #计算动作的标准差，取log_std的指数
        dist = Normal(mean, std) #创建一个正态分布对象，参数为均值和标准差
        log_prob = dist.log_prob(action) #计算动作的对数概率，sum(-1)是因为可能有多个动作维度
        entropy = dist.entropy() #计算动作分布的熵，sum(-1)是因为可能有多个动作维度
        return log_prob, value, entropy
