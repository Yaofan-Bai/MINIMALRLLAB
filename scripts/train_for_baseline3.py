import os
import sys
from datetime import datetime

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_env(env_name: str = "HalfCheetah-v5", max_episode_steps: int = 1000):
    def _init():
        env = gym.make(env_name, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        return env

    return _init


def train():
    log_root = os.path.join("logs", "sb3_ppo")
    os.makedirs(log_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log = os.path.join(log_root, run_id)

    env_name = "HalfCheetah-v5"
    max_episode_steps = 1000
    total_timesteps = 2_000_000

    env = DummyVecEnv([make_env(env_name, max_episode_steps)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
        log_std_init=-2,
        ortho_init=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=64,
        n_epochs=20,
        gamma=0.98,
        gae_lambda=0.92,
        clip_range=0.1,
        ent_coef=0.000401762,
        learning_rate=2.0633e-5,
        vf_coef=0.58096,
        max_grad_norm=0.8,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log,
        device="cpu",
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    save_dir = os.path.join("checkpoints", "sb3_ppo")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"ppo_{env_name}_{run_id}")
    vecnorm_path = os.path.join(save_dir, f"vecnorm_{env_name}_{run_id}.pkl")
    model.save(model_path)
    env.save(vecnorm_path)
    env.close()

    print(f"Model saved to: {model_path}")
    print(f"VecNormalize saved to: {vecnorm_path}")


if __name__ == "__main__":
    train()
