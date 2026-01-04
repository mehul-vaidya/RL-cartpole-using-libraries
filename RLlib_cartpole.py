import ray
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
import torch

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
    .env_runners(     # NEW API (replaces .rollouts)
        num_env_runners=1, # parallel env workers
        rollout_fragment_length=200  #How long the dog plays before writing in the diary ,One diary page = 200 steps
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
for i in range(30):
    result = algo.train()
    reward_mean = result["env_runners"]["episode_return_mean"]
    print(f"Iter {i} | Reward Mean: {reward_mean:.2f}")
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

# Get the trained module (NEW API)
module = algo.get_module()

done = False
while not done:
    # RLlib expects a dict input
    input_dict = {
        "obs": torch.tensor(obs).unsqueeze(0)
    }

    # Forward pass (inference)
    output = module.forward_inference(input_dict)

    # Extract action
    logits = output["action_dist_inputs"]
    action = torch.argmax(logits, dim=-1).item()

    obs, reward, done, truncated, _ = env.step(action)

env.close()