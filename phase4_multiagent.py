import gl_gym
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces

# ══════════════════════════════════════════════
# STEP 1 — Per-Tier Environment Wrapper
# Each tier gets its own independent environment
# ══════════════════════════════════════════════
class TierEnv(gym.Wrapper):
    """
    Wraps GreenLight to simulate one shelf tier.
    Each tier has:
    - Its own microclimate (slight temp/light variation)
    - Its own PPO agent
    - Its own phenotyping reward
    """

    OPTIMAL_TEMP_MIN  = 18.0
    OPTIMAL_TEMP_MAX  = 26.0
    OPTIMAL_CO2_MIN   = 600.0
    OPTIMAL_CO2_MAX   = 1000.0
    OPTIMAL_HUM_MIN   = 70.0
    OPTIMAL_HUM_MAX   = 90.0

    # Each tier has slightly different conditions
    # Top tier = hotter + more light
    # Bottom tier = cooler + less light
    TIER_TEMP_OFFSET  = {0: -2.0, 1: 0.0, 2: +2.0}  # bottom, mid, top
    TIER_LIGHT_OFFSET = {0: -0.1, 1: 0.0, 2: +0.1}

    def __init__(self, tier_id: int):
        base_env = gym.make("gl_gym/GreenLightTomato-v0")
        super().__init__(base_env)
        self.tier_id      = tier_id
        self.tier_name    = ["Bottom", "Middle", "Top"][tier_id]
        self.previous_lai = 0.0
        self.step_count   = 0

        # Flatten observation space
        total_size = 0
        for key, space in base_env.observation_space.spaces.items():
            total_size += int(np.prod(space.shape))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_size,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_lai = 0.0
        self.step_count   = 0
        return self._flatten(obs), info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        self.step_count += 1

        # ── Extract observations ───────────────────────────────
        crop    = np.array(obs["BasicCropObservations"]).flatten()
        climate = np.array(obs["IndoorClimateObservations"]).flatten()
        control = np.array(obs["ControlObservations"]).flatten()

        lai_proxy = crop[2]
        co2       = climate[0]
        temp      = climate[1] + self.TIER_TEMP_OFFSET[self.tier_id]
        humidity  = climate[2]

        # ── LAI reward ────────────────────────────────────────
        current_lai = lai_proxy / 5000.0
        lai_gain    = current_lai - self.previous_lai
        lai_reward  = lai_gain * 20.0

        # ── Stress penalty ────────────────────────────────────
        temp_stress = max(0, self.OPTIMAL_TEMP_MIN - temp) / 10.0 \
                    + max(0, temp - self.OPTIMAL_TEMP_MAX) / 10.0

        co2_stress  = max(0, self.OPTIMAL_CO2_MIN - co2) / 500.0 \
                    + max(0, co2 - self.OPTIMAL_CO2_MAX) / 500.0

        hum_stress  = max(0, self.OPTIMAL_HUM_MIN - humidity) / 20.0 \
                    + max(0, humidity - self.OPTIMAL_HUM_MAX) / 20.0

        stress_penalty = temp_stress + co2_stress + hum_stress

        # ── Energy penalty ────────────────────────────────────
        energy_penalty = np.mean(np.abs(control)) * 0.05

        # ── Final phenotyping reward ──────────────────────────
        reward = lai_reward - stress_penalty - energy_penalty

        # Update state
        self.previous_lai = current_lai

        # Add tier info
        info["tier_id"]       = self.tier_id
        info["tier_name"]     = self.tier_name
        info["lai"]           = current_lai
        info["temp"]          = temp
        info["stress"]        = stress_penalty
        info["reward"]        = reward

        return self._flatten(obs), reward, done, truncated, info

    def _flatten(self, obs):
        parts = []
        for key, val in obs.items():
            parts.append(np.array(val, dtype=np.float32).flatten())
        return np.concatenate(parts)


# ══════════════════════════════════════════════
# STEP 2 — Reward Logger per agent
# ══════════════════════════════════════════════
class TierLogger(BaseCallback):
    def __init__(self, tier_name):
        super().__init__()
        self.tier_name       = tier_name
        self.episode_rewards = []
        self.current_reward  = 0.0
        self.episode_count   = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_count += 1
            print(f"  [{self.tier_name}] Episode {self.episode_count:3d} | "
                  f"Reward: {self.current_reward:.3f}")
            self.current_reward = 0.0
        return True


# ══════════════════════════════════════════════
# STEP 3 — Create one PPO agent per tier
# ══════════════════════════════════════════════
print("=" * 55)
print("Phase 4 — Multi-Agent Per-Tier PPO Training")
print("=" * 55)

NUM_TIERS = 3
agents    = []
envs      = []
loggers   = []
tier_names = ["Bottom", "Middle", "Top"]

for tier_id in range(NUM_TIERS):
    print(f"\nSetting up Tier {tier_id+1} ({tier_names[tier_id]})...")

    # Each tier gets its own environment instance
    env = TierEnv(tier_id=tier_id)
    envs.append(env)

    # Each tier gets its own PPO agent
    agent = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = 3e-4,
        n_steps       = 1024,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        verbose       = 0,
        tensorboard_log = None
    )
    agents.append(agent)

    # Each tier gets its own logger
    logger = TierLogger(tier_names[tier_id])
    loggers.append(logger)

    print(f"  ✓ Agent {tier_id+1} ready for {tier_names[tier_id]} tier")

# ══════════════════════════════════════════════
# STEP 4 — Train all agents
# We train them sequentially (one after another)
# In a real system they'd run in parallel
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print("Training all 3 agents...")
print("=" * 55)

TIMESTEPS = 50_000  # increase to 200_000 for better results

for tier_id in range(NUM_TIERS):
    print(f"\n{'─'*55}")
    print(f"Training Agent {tier_id+1} — {tier_names[tier_id]} Tier")
    print(f"{'─'*55}")

    agents[tier_id].learn(
        total_timesteps = TIMESTEPS,
        callback        = loggers[tier_id]
    )
    print(f"✓ {tier_names[tier_id]} tier agent training complete!")

# ══════════════════════════════════════════════
# STEP 5 — Save all agents
# ══════════════════════════════════════════════
os.makedirs("models", exist_ok=True)
for tier_id in range(NUM_TIERS):
    path = f"models/ppo_tier_{tier_names[tier_id].lower()}"
    agents[tier_id].save(path)
    print(f"✓ Saved: {path}")

# ══════════════════════════════════════════════
# STEP 6 — Plot all 3 agents together
# ══════════════════════════════════════════════
os.makedirs("results", exist_ok=True)

colors = ['#2196F3', '#4CAF50', '#FF5722']  # blue, green, orange

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1 — All 3 reward curves on same graph
for tier_id in range(NUM_TIERS):
    rewards = loggers[tier_id].episode_rewards
    if len(rewards) == 0:
        continue

    axes[0].plot(
        rewards, alpha=0.25,
        color=colors[tier_id]
    )
    # Smoothed line
    window = min(10, len(rewards))
    if len(rewards) >= window:
        smoothed = np.convolve(
            rewards, np.ones(window)/window, mode='valid'
        )
        axes[0].plot(
            range(window-1, len(rewards)),
            smoothed,
            color=colors[tier_id],
            linewidth=2.5,
            label=f'{tier_names[tier_id]} tier'
        )

axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].set_title("Multi-Agent Training — All 3 Tiers")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2 — Final average reward per tier (bar chart)
final_rewards = []
for tier_id in range(NUM_TIERS):
    rewards = loggers[tier_id].episode_rewards
    if len(rewards) >= 10:
        # Average of last 10 episodes
        final_rewards.append(np.mean(rewards[-10:]))
    else:
        final_rewards.append(np.mean(rewards) if rewards else 0)

bars = axes[1].bar(
    tier_names, final_rewards,
    color=colors, alpha=0.8,
    edgecolor='black', linewidth=0.5
)
axes[1].set_xlabel("Shelf Tier")
axes[1].set_ylabel("Average Reward (last 10 episodes)")
axes[1].set_title("Final Performance Per Tier")
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, final_rewards):
    axes[1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{val:.3f}',
        ha='center', va='bottom', fontsize=11
    )

plt.suptitle("Phase 4 — Multi-Agent Per-Tier Results",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/multiagent_results.png", dpi=150)
plt.show()
print("\n✓ Plot saved to results/multiagent_results.png")

# ══════════════════════════════════════════════
# STEP 7 — Print summary
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print("MULTI-AGENT TRAINING SUMMARY")
print("=" * 55)
for tier_id in range(NUM_TIERS):
    rewards = loggers[tier_id].episode_rewards
    if rewards:
        print(f"{tier_names[tier_id]:8} tier | "
              f"Episodes: {len(rewards):4d} | "
              f"Best reward: {max(rewards):.3f} | "
              f"Final avg: {np.mean(rewards[-10:]):.3f}")

print("\n✓ Phase 4 Complete — 3 independent agents trained!")
print("=" * 55)