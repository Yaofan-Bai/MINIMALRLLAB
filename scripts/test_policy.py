import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.actor_critic import ActorCritic

policy = ActorCritic(24,6)
state = torch.randn(1,24)
action, log_prob, value, entropy = policy.act(state)

print(action)
print(log_prob)
print(value)
