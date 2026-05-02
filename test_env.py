import gl_gym
import gymnasium as gym
import numpy as np

print("=" * 40)
print("Testing GreenLight Farm Environment...")
print("=" * 40)

# Create the environment
env = gym.make("gl_gym/GreenLightTomato-v0")
print("✓ Environment created successfully")

# Reset it
obs, info = env.reset()
print("✓ Environment reset successfully")

# Handle dict or array observation
print("Observation type:", type(obs))
if isinstance(obs, dict):
    print("Observation keys:", list(obs.keys()))
    for key, val in obs.items():
        print(f"  {key}: shape={np.array(val).shape}, sample={np.array(val).flatten()[:3]}")
else:
    print("Observation shape:", obs.shape)
    print("Sample observation (first 5 values):", obs[:5])

print("Action space:", env.action_space)

# Step through 5 timesteps with random actions
print("\nRunning 5 random steps...")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step {i+1} | reward: {reward:.4f} | done: {done}")

env.close()
print("=" * 40)
print("Environment works perfectly!")
print("=" * 40)