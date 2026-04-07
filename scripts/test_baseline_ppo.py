import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 创建环境
env = gym.make("HalfCheetah-v5")
print(f"状态空间: {env.observation_space.shape[0]}, 动作空间: {env.action_space.shape[0]}")

# 创建并训练PPO模型
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.1)
print("开始训练...")
model.learn(total_timesteps=2000000)
model.save("ppo_halfcheetah")

# 评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"训练完成! 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

# 展示效果
print("\n展示训练效果:")
env = gym.make("HalfCheetah-v5", render_mode="human")
obs, _ = env.reset()
for _ in range(200):  # 运行200步
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()