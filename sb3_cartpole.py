import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
"""
👉 SB3 internally creates:
Actor
Critic
Advantage calculation
PPO clipping
Optimizer
Training loop
You don’t need to write any of it now.
"""
model = PPO(  #Proximal Policy Optimization
    "MlpPolicy", #use a neural network (MLP)
    env,         #use a neural network (MLP)
    verbose=1,  #where the agent learns
)

"""
SB3 handles:
Data collection
PPO updates
Policy improvement
"""

model.learn(total_timesteps=100_000)
model.save("sb3_ppo_cartpole")

# Test
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    if done:
        obs, _ = env.reset()
