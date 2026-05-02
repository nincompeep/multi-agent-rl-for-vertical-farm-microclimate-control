import gl_gym
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces

# ══════════════════════════════════════════════
# STEP 1 — Wrapper to flatten dict observations
# ══════════════════════════════════════════════
class FlattenObsWrapper(gym.ObservationWrapper):
    """
    GreenLight returns observations as a dictionary.
    PPO needs a flat array — this wrapper converts it automatically.
    """
    def __init__(self, env):
        super().__init__(env)
        # Calculate total size of all observation values combined
        total_size = 0
        for key, space in env.observation_space.spaces.items():
            total_size += int(np.prod(space.shape))
        # Define new flat observation space
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (total_size,),
            dtype = np.float32
        )

    def observation(self, obs):
        # Flatten all dict values into one single array
        parts = []
        for key, val in obs.items():
            parts.append(np.array(val, dtype=np.float32).flatten())
        return np.concatenate(parts)


# ══════════════════════════════════════════════
# STEP 2 — Reward logger callback
# ══════════════════════════════════════════════
class RewardLogger(BaseCallback):
    """
    Tracks reward after every episode so we can plot it later.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward  = 0.0
        self.episode_count   = 0

    def _on_step(self):
        # Accumulate reward each step
        self.current_reward += self.locals["rewards"][0]

        # When episode ends, save the total
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_count += 1
            print(f"  Episode {self.episode_count} finished | "
                  f"Total reward: {self.current_reward:.2f}")
            self.current_reward = 0.0
        return True


# ══════════════════════════════════════════════
# STEP 3 — Create and wrap the environment
# ══════════════════════════════════════════════
print("=" * 50)
print("Setting up GreenLight Farm Environment...")
print("=" * 50)

def make_env():
    env = gym.make("gl_gym/GreenLightTomato-v0")
    env = FlattenObsWrapper(env)
    return env

env = make_vec_env(make_env, n_envs=1)
print("✓ Environment ready")

# ══════════════════════════════════════════════
# STEP 4 — Define the PPO agent
# ══════════════════════════════════════════════
print("\nBuilding PPO agent...")

model = PPO(
    policy         = "MlpPolicy",  # standard neural network
    env            = env,
    learning_rate  = 3e-4,         # how fast it learns
    n_steps        = 1024,         # steps before each update
    batch_size     = 64,           # mini batch size
    n_epochs       = 10,           # passes per update
    gamma          = 0.99,         # future reward discount
    gae_lambda     = 0.95,         # advantage smoothing
    clip_range     = 0.2,          # PPO stability clip
    verbose        = 0,            # we handle printing ourselves
    tensorboard_log= None
)

print("✓ PPO agent ready")
print("\nAgent policy network:")
print(model.policy)

# ══════════════════════════════════════════════
# STEP 5 — Train the agent
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("Starting training (this will take a few minutes)...")
print("=" * 50)

logger = RewardLogger()

model.learn(
    total_timesteps = 50_000,   # start small to test — increase later
    callback        = logger
)

print("\n✓ Training complete!")

# ══════════════════════════════════════════════
# STEP 6 — Save the model
# ══════════════════════════════════════════════
os.makedirs("models", exist_ok=True)
model.save("models/ppo_baseline")
print("✓ Model saved to models/ppo_baseline")

# ══════════════════════════════════════════════
# STEP 7 — Plot the reward curve
# ══════════════════════════════════════════════
os.makedirs("results", exist_ok=True)

rewards = logger.episode_rewards

if len(rewards) > 0:
    plt.figure(figsize=(12, 5))

    # Raw rewards
    plt.plot(rewards, alpha=0.4, color='steelblue', label='Episode reward')

    # Smoothed average line
    window = min(10, len(rewards))
    if len(rewards) >= window:
        smoothed = np.convolve(
            rewards,
            np.ones(window) / window,
            mode='valid'
        )
        plt.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color='red',
            linewidth=2,
            label=f'Smoothed (window={window})'
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Baseline — Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/baseline_reward.png", dpi=150)
    plt.show()
    print("✓ Reward plot saved to results/baseline_reward.png")
else:
    print("No episodes completed yet — increase total_timesteps")

print("\n" + "=" * 50)
print("Phase 2 Complete!")
print("=" * 50)