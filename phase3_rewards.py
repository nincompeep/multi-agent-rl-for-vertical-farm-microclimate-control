import gl_gym
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import spaces

# ══════════════════════════════════════════════
# STEP 1 — Flatten wrapper (same as before)
# ══════════════════════════════════════════════
class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        total_size = 0
        for key, space in env.observation_space.spaces.items():
            total_size += int(np.prod(space.shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_size,), dtype=np.float32
        )

    def observation(self, obs):
        parts = []
        for key, val in obs.items():
            parts.append(np.array(val, dtype=np.float32).flatten())
        return np.concatenate(parts)


# ══════════════════════════════════════════════
# STEP 2 — Phenotyping Reward Wrapper
# This is the CORE INNOVATION of your project
# ══════════════════════════════════════════════
class PhenotypingRewardWrapper(gym.Wrapper):
    """
    Replaces the default reward with a biologically meaningful one.

    Instead of waiting for dry biomass at the end (sparse reward),
    we compute reward every step using:

    1. LAI GAIN     — reward the agent for growing more leaf area
    2. STRESS PENALTY — penalise the agent when temp/CO2/humidity
                        is outside the optimal range for the plant
    3. ENERGY PENALTY — penalise the agent for wasting energy
                        (running LEDs at full power unnecessarily)
    """

    # Optimal climate ranges for tomato plants
    OPTIMAL_TEMP_MIN  = 18.0   # °C
    OPTIMAL_TEMP_MAX  = 26.0   # °C
    OPTIMAL_CO2_MIN   = 600.0  # ppm
    OPTIMAL_CO2_MAX   = 1000.0 # ppm
    OPTIMAL_HUM_MIN   = 70.0   # %
    OPTIMAL_HUM_MAX   = 90.0   # %

    def __init__(self, env):
        super().__init__(env)
        self.previous_lai    = 0.0
        self.previous_biomass = 0.0
        self.step_count      = 0

        # Track metrics for analysis
        self.lai_history     = []
        self.stress_history  = []
        self.reward_history  = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_lai     = 0.0
        self.previous_biomass = 0.0
        self.step_count       = 0
        return obs, info

    def step(self, action):
        obs, default_reward, done, truncated, info = self.env.step(action)
        self.step_count += 1

        # ── Extract values from observation dict ──────────────────
        crop    = np.array(obs["BasicCropObservations"]).flatten()
        climate = np.array(obs["IndoorClimateObservations"]).flatten()
        control = np.array(obs["ControlObservations"]).flatten()

        # BasicCropObservations: [dev_stage, dry_weight, lai_proxy]
        dev_stage  = crop[0]
        dry_weight = crop[1]
        lai_proxy  = crop[2]   # proxy for Leaf Area Index

        # IndoorClimateObservations: [co2, temp, humidity, light]
        co2      = climate[0]
        temp     = climate[1]
        humidity = climate[2]
        light    = climate[3]

        # ── 1. LAI REWARD ─────────────────────────────────────────
        # Reward the agent for increasing leaf area each step
        # Normalise by dividing by a typical max value
        current_lai  = lai_proxy / 5000.0   # normalise to ~0-1 range
        lai_gain     = current_lai - self.previous_lai
        lai_reward   = lai_gain * 20.0      # scale up so it's meaningful

        # ── 2. CHLOROPHYLL STRESS PENALTY ─────────────────────────
        # Penalise when temperature is outside optimal range
        if temp < self.OPTIMAL_TEMP_MIN:
            temp_stress = (self.OPTIMAL_TEMP_MIN - temp) / 10.0
        elif temp > self.OPTIMAL_TEMP_MAX:
            temp_stress = (temp - self.OPTIMAL_TEMP_MAX) / 10.0
        else:
            temp_stress = 0.0

        # Penalise when CO2 is outside optimal range
        if co2 < self.OPTIMAL_CO2_MIN:
            co2_stress = (self.OPTIMAL_CO2_MIN - co2) / 500.0
        elif co2 > self.OPTIMAL_CO2_MAX:
            co2_stress = (co2 - self.OPTIMAL_CO2_MAX) / 500.0
        else:
            co2_stress = 0.0

        # Penalise when humidity is outside optimal range
        if humidity < self.OPTIMAL_HUM_MIN:
            hum_stress = (self.OPTIMAL_HUM_MIN - humidity) / 20.0
        elif humidity > self.OPTIMAL_HUM_MAX:
            hum_stress = (humidity - self.OPTIMAL_HUM_MAX) / 20.0
        else:
            hum_stress = 0.0

        # Combined stress score (0 = no stress, higher = more stress)
        stress_penalty = (temp_stress + co2_stress + hum_stress)

        # ── 3. ENERGY EFFICIENCY PENALTY ──────────────────────────
        # Mildly penalise high energy use (average of all controls)
        energy_use     = np.mean(np.abs(control))
        energy_penalty = energy_use * 0.05

        # ── 4. COMBINE INTO FINAL REWARD ──────────────────────────
        phenotype_reward = (
              lai_reward          # grow more leaves = good
            - stress_penalty      # plant stress = bad
            - energy_penalty      # wasting energy = bad
        )

        # ── Store for analysis ────────────────────────────────────
        self.previous_lai      = current_lai
        self.previous_biomass  = dry_weight
        self.lai_history.append(current_lai)
        self.stress_history.append(stress_penalty)
        self.reward_history.append(phenotype_reward)

        # Add breakdown to info for debugging
        info["phenotype_reward"]  = phenotype_reward
        info["lai_reward"]        = lai_reward
        info["stress_penalty"]    = stress_penalty
        info["energy_penalty"]    = energy_penalty
        info["current_lai"]       = current_lai
        info["temp_stress"]       = temp_stress
        info["co2_stress"]        = co2_stress

        return obs, phenotype_reward, done, truncated, info


# ══════════════════════════════════════════════
# STEP 3 — Reward logger callback
# ══════════════════════════════════════════════
class PhenoRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward  = 0.0
        self.episode_count   = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_count += 1
            print(f"  Episode {self.episode_count:3d} | "
                  f"Phenotype reward: {self.current_reward:.3f}")
            self.current_reward = 0.0
        return True


# ══════════════════════════════════════════════
# STEP 4 — Build environment with new reward
# ══════════════════════════════════════════════
print("=" * 50)
print("Phase 3 — Phenotyping Reward Training")
print("=" * 50)

def make_pheno_env():
    env = gym.make("gl_gym/GreenLightTomato-v0")
    env = PhenotypingRewardWrapper(env)   # apply phenotyping reward
    env = FlattenObsWrapper(env)          # flatten for PPO
    return env

env = make_vec_env(make_pheno_env, n_envs=1)
print("✓ Environment with phenotyping reward ready")

# ══════════════════════════════════════════════
# STEP 5 — Train PPO with phenotyping reward
# ══════════════════════════════════════════════
model = PPO(
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

print("✓ PPO agent ready")
print("\nStarting training with phenotyping reward...")
print("=" * 50)

logger = PhenoRewardLogger()
model.learn(total_timesteps=50_000, callback=logger)

print("\n✓ Training complete!")

# ══════════════════════════════════════════════
# STEP 6 — Save model
# ══════════════════════════════════════════════
os.makedirs("models", exist_ok=True)
model.save("models/ppo_phenotyping")
print("✓ Model saved to models/ppo_phenotyping")

# ══════════════════════════════════════════════
# STEP 7 — Plot and compare results
# ══════════════════════════════════════════════
os.makedirs("results", exist_ok=True)
rewards = logger.episode_rewards

if len(rewards) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1 — Phenotyping reward curve
    axes[0].plot(rewards, alpha=0.4, color='green', label='Episode reward')
    window = min(10, len(rewards))
    if len(rewards) >= window:
        smoothed = np.convolve(
            rewards, np.ones(window)/window, mode='valid'
        )
        axes[0].plot(
            range(window-1, len(rewards)), smoothed,
            color='darkgreen', linewidth=2,
            label=f'Smoothed (window={window})'
        )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Phenotyping Reward")
    axes[0].set_title("Phase 3 — Phenotyping Reward Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2 — Reward breakdown explanation
    labels  = ['LAI gain\n(leaf growth)', 
               'Stress penalty\n(temp/CO2/humidity)', 
               'Energy penalty\n(control usage)']
    weights = [20.0, -1.0, -0.05]
    colors  = ['green', 'red', 'orange']
    axes[1].bar(labels, [20.0, 1.0, 0.05], color=colors, alpha=0.7)
    axes[1].set_title("Reward Component Weights")
    axes[1].set_ylabel("Weight magnitude")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Phase 3 — Phenotyping Reward Analysis", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/phenotyping_reward.png", dpi=150)
    plt.show()
    print("✓ Plot saved to results/phenotyping_reward.png")

print("\n" + "=" * 50)
print("Phase 3 Complete!")
print("=" * 50)