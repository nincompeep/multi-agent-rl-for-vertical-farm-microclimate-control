import gl_gym
import gymnasium as gym
import numpy as np

env = gym.make("gl_gym/GreenLightTomato-v0")
obs, info = env.reset()

print("=" * 50)
print("FULL OBSERVATION BREAKDOWN")
print("=" * 50)

for key, val in obs.items():
    arr = np.array(val).flatten()
    print(f"\n{key}:")
    print(f"  Shape : {arr.shape}")
    print(f"  Values: {arr}")

print("\n" + "=" * 50)
print("INFO DICTIONARY")
print("=" * 50)
for key, val in info.items():
    print(f"  {key}: {val}")

# Run 3 steps and see how values change
print("\n" + "=" * 50)
print("HOW VALUES CHANGE OVER 3 STEPS")
print("=" * 50)
for step in range(3):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"\nStep {step+1}:")
    for key, val in obs.items():
        arr = np.array(val).flatten()
        print(f"  {key}: {arr}")
    print(f"  Default reward: {reward:.4f}")

env.close()