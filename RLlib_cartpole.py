import ray
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym

# ===============================
# Initialize Ray
# ===============================
'''
RLlib runs on Ray, which handles:
Parallel environments
Distributed training
Scaling (even across machines)
Think of Ray as the engine, RLlib as the driver.
'''
ray.init(ignore_reinit_error=True)

# ===============================
# Create PPO Config
# ===============================
config = (
    PPOConfig() #use PPO algo
    .environment(env="CartPole-v1")
    .framework("torch")
    .rollouts(
        num_rollout_workers=1,   # parallel env workers
        rollout_fragment_length=200 #How long the dog plays before writing in the diary ,One diary page = 200 steps
    )
    .training(
        gamma=0.99, #How much the dog cares about future treats
        lr=3e-4, #How strongly the coach corrects the dog
        train_batch_size=4000 #Coach waits for 4000 steps, Then updates the dog’s brain, 4000 / 200 = 20 diary pages
    )

)

# ===============================
# Build PPO Algorithm
# ===============================
algo = config.build()

# ===============================
# Training Loop
# ===============================
for i in range(30): # 30 times 4000 steps
    result = algo.train()
    print(
        f"Iter {i} | "
        f"Reward Mean: {result['episode_reward_mean']:.2f}"
    )

# ===============================
# Save Model
# ===============================
checkpoint = algo.save()
print("✅ Model saved at:", checkpoint)

# ===============================
# Test the Trained Agent
# ===============================
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
done = False

while not done:
    action = algo.compute_single_action(obs)
    obs, reward, done, truncated, _ = env.step(action)

env.close()
ray.shutdown()
