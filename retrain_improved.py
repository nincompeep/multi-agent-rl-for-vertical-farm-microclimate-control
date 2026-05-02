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
# IMPROVED TIER ENVIRONMENT
# Based on REAL simulator values:
# Temp:  6-14°C  avg 8°C
# CO2:   461-1048 avg 606
# Humid: 81-92%  avg 89%
# ══════════════════════════════════════════════
class ImprovedTierEnv(gym.Wrapper):

    # ── Real optimal ranges from simulator data ──
    OPTIMAL_TEMP_MIN  = 6.0    # real simulator min
    OPTIMAL_TEMP_MAX  = 14.0   # real simulator max
    OPTIMAL_CO2_MIN   = 450.0  # real simulator min
    OPTIMAL_CO2_MAX   = 1100.0 # real simulator max
    OPTIMAL_HUM_MIN   = 78.0   # real simulator min
    OPTIMAL_HUM_MAX   = 94.0   # real simulator max

    # ── Tier microclimate offsets ────────────────
    TIER_TEMP_OFFSET  = {0: -1.0, 1: 0.0, 2: +1.0}

    def __init__(self, tier_id):
        base_env = gym.make("gl_gym/GreenLightTomato-v0")
        super().__init__(base_env)
        self.tier_id       = tier_id
        self.tier_name     = ["Bottom", "Middle", "Top"][tier_id]
        self.previous_lai  = 0.0
        self.step_count    = 0
        self.total_energy  = 0.0
        self.episode_lai_gains   = []
        self.episode_stress_vals = []
        self.episode_energy_vals = []

        total_size = sum(
            int(np.prod(s.shape))
            for s in base_env.observation_space.spaces.values()
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_size,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_lai        = 0.0
        self.step_count          = 0
        self.total_energy        = 0.0
        self.episode_lai_gains   = []
        self.episode_stress_vals = []
        self.episode_energy_vals = []
        return self._flatten(obs), info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        self.step_count += 1

        # ── Extract real observations ──────────────
        crop    = np.array(obs["BasicCropObservations"]).flatten()
        climate = np.array(obs["IndoorClimateObservations"]).flatten()
        control = np.array(obs["ControlObservations"]).flatten()

        lai_proxy = crop[2]
        co2       = climate[0]
        temp      = climate[1] + self.TIER_TEMP_OFFSET[self.tier_id]
        humidity  = climate[2]

        # ── 1. LAI GROWTH REWARD ──────────────────
        # Normalise LAI proxy to 0-1 range
        # From data: lai_proxy around 3097-3500
        current_lai = lai_proxy / 5000.0
        lai_gain    = current_lai - self.previous_lai

        # Strong positive reward for any growth
        if lai_gain > 0:
            lai_reward = lai_gain * 50.0   # bigger scale = clearer signal
        else:
            lai_reward = lai_gain * 10.0   # small penalty for loss

        # ── 2. CLIMATE QUALITY BONUS ──────────────
        # Instead of penalising — reward for being IN optimal range
        temp_ok = 1.0 if self.OPTIMAL_TEMP_MIN <= temp <= self.OPTIMAL_TEMP_MAX \
                  else -abs(temp - (self.OPTIMAL_TEMP_MIN + self.OPTIMAL_TEMP_MAX)/2) / 10.0
        co2_ok  = 1.0 if self.OPTIMAL_CO2_MIN  <= co2  <= self.OPTIMAL_CO2_MAX \
                  else -abs(co2  - (self.OPTIMAL_CO2_MIN  + self.OPTIMAL_CO2_MAX )/2) / 500.0
        hum_ok  = 1.0 if self.OPTIMAL_HUM_MIN  <= humidity <= self.OPTIMAL_HUM_MAX \
                  else -abs(humidity - (self.OPTIMAL_HUM_MIN + self.OPTIMAL_HUM_MAX)/2) / 10.0

        climate_bonus = (temp_ok + co2_ok + hum_ok) * 0.1

        # ── 3. STRESS PENALTY (for going OUT of range) ──
        temp_stress = max(0, self.OPTIMAL_TEMP_MIN - temp) / 8.0 \
                    + max(0, temp - self.OPTIMAL_TEMP_MAX) / 8.0
        co2_stress  = max(0, self.OPTIMAL_CO2_MIN  - co2)  / 600.0 \
                    + max(0, co2  - self.OPTIMAL_CO2_MAX)   / 600.0
        hum_stress  = max(0, self.OPTIMAL_HUM_MIN  - humidity) / 15.0 \
                    + max(0, humidity - self.OPTIMAL_HUM_MAX)   / 15.0
        stress      = temp_stress + co2_stress + hum_stress

        # ── 4. ENERGY COST TRACKING ───────────────
        # Track each control separately for energy analysis
        led_energy    = abs(float(control[0])) * 0.04
        heat_energy   = abs(float(control[1])) * 0.03
        vent_energy   = abs(float(control[2])) * 0.02
        co2_energy    = abs(float(control[3])) * 0.02
        other_energy  = np.mean(np.abs(control[4:])) * 0.01
        total_energy  = led_energy + heat_energy + vent_energy \
                      + co2_energy + other_energy
        self.total_energy += total_energy

        # ── 5. SURVIVAL BONUS ────────────────────
        # Small reward just for keeping plant alive each step
        survival_bonus = 0.05

        # ── 6. FINAL REWARD ───────────────────────
        reward = (
              survival_bonus    # always positive base
            + lai_reward        # growth signal
            + climate_bonus     # bonus for good climate
            - stress * 0.5      # penalty for bad climate
            - total_energy      # energy efficiency
        )

        # ── Track metrics ─────────────────────────
        self.previous_lai = current_lai
        self.episode_lai_gains.append(lai_gain)
        self.episode_stress_vals.append(stress)
        self.episode_energy_vals.append(total_energy)

        # ── Info dict ─────────────────────────────
        info["lai"]          = float(current_lai)
        info["stress"]       = float(stress)
        info["temp"]         = float(temp)
        info["co2"]          = float(co2)
        info["humidity"]     = float(humidity)
        info["reward"]       = float(reward)
        info["energy"]       = float(total_energy)
        info["total_energy"] = float(self.total_energy)
        info["led_energy"]   = float(led_energy)
        info["heat_energy"]  = float(heat_energy)
        info["vent_energy"]  = float(vent_energy)
        info["lai_gain"]     = float(lai_gain)
        info["climate_bonus"]= float(climate_bonus)
        info["tier_name"]    = self.tier_name

        return self._flatten(obs), reward, done, truncated, info

    def _flatten(self, obs):
        parts = []
        for key, val in obs.items():
            parts.append(np.array(val, dtype=np.float32).flatten())
        return np.concatenate(parts)


# ══════════════════════════════════════════════
# IMPROVED LOGGER
# ══════════════════════════════════════════════
class ImprovedLogger(BaseCallback):
    def __init__(self, tier_name, verbose=True):
        super().__init__()
        self.tier_name       = tier_name
        self.verbose         = verbose
        self.episode_rewards = []
        self.episode_lai     = []
        self.episode_stress  = []
        self.episode_energy  = []
        self.current_reward  = 0.0
        self.current_lai     = []
        self.current_stress  = []
        self.current_energy  = []
        self.episode_count   = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        info = self.locals["infos"][0]
        self.current_lai.append(info.get("lai", 0))
        self.current_stress.append(info.get("stress", 0))
        self.current_energy.append(info.get("energy", 0))

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lai.append(np.mean(self.current_lai))
            self.episode_stress.append(np.mean(self.current_stress))
            self.episode_energy.append(np.sum(self.current_energy))
            self.episode_count += 1

            if self.verbose and self.episode_count % 5 == 0:
                print(f"  [{self.tier_name}] Ep {self.episode_count:4d} | "
                      f"Reward: {self.current_reward:8.3f} | "
                      f"LAI: {np.mean(self.current_lai):.4f} | "
                      f"Stress: {np.mean(self.current_stress):.4f} | "
                      f"Energy: {np.sum(self.current_energy):.3f}")

            self.current_reward = 0.0
            self.current_lai    = []
            self.current_stress = []
            self.current_energy = []
        return True


# ══════════════════════════════════════════════
# TRAINING SETUP
# ══════════════════════════════════════════════
print("=" * 60)
print("IMPROVED MULTI-AGENT TRAINING")
print("Real simulator ranges:")
print("  Temp:  6-14°C  (avg 8°C)")
print("  CO2:   461-1048 ppm (avg 606)")
print("  Humid: 81-92%  (avg 89%)")
print("=" * 60)

TIER_NAMES  = ["bottom", "middle", "top"]
TIMESTEPS   = 200_000   # 4x more than before
agents      = []
envs        = []
loggers     = []

for tier_id, name in enumerate(TIER_NAMES):
    print(f"\nSetting up {name.capitalize()} Tier agent...")
    env    = ImprovedTierEnv(tier_id=tier_id)
    envs.append(env)

    # ── Improved PPO hyperparameters ────────────
    agent = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = 1e-4,      # slower = more stable
        n_steps        = 2048,      # more steps per update
        batch_size     = 128,       # larger batch
        n_epochs       = 15,        # more passes per update
        gamma          = 0.995,     # care more about future
        gae_lambda     = 0.97,      # smoother advantage
        clip_range     = 0.15,      # tighter clipping = more stable
        ent_coef       = 0.01,      # encourage exploration
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        verbose        = 0,
        tensorboard_log= None,
        policy_kwargs  = dict(
            net_arch   = [128, 128, 64]  # deeper network
        )
    )
    agents.append(agent)

    logger = ImprovedLogger(name.capitalize())
    loggers.append(logger)
    print(f"  ✓ {name.capitalize()} agent ready (128-128-64 network)")

# ══════════════════════════════════════════════
# TRAIN ALL AGENTS
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Training 3 agents × {TIMESTEPS:,} timesteps each")
print(f"Estimated time: 30-45 minutes")
print("=" * 60)

for tier_id, name in enumerate(TIER_NAMES):
    print(f"\n{'─' * 60}")
    print(f"Training {name.capitalize()} Tier Agent ({tier_id+1}/3)")
    print(f"{'─' * 60}")
    agents[tier_id].learn(
        total_timesteps=TIMESTEPS,
        callback=loggers[tier_id]
    )
    print(f"✓ {name.capitalize()} agent training complete!")
    # Save immediately after each tier finishes
    os.makedirs("models", exist_ok=True)
    path = f"models/ppo_tier_{name}"
    agents[tier_id].save(path)
    print(f"✓ Saved to {path}")

# ══════════════════════════════════════════════
# GENERATE IMPROVED RESULT GRAPHS
# ══════════════════════════════════════════════
print("\nGenerating result graphs...")
os.makedirs("results", exist_ok=True)

colors     = ["#1e88e5", "#00a84f", "#ff7043"]
tier_labels = ["Bottom Tier", "Middle Tier", "Top Tier"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor("#ffffff")

for i, (name, label, color) in enumerate(
        zip(TIER_NAMES, tier_labels, colors)):
    lg = loggers[i]
    ep = range(1, len(lg.episode_rewards) + 1)

    # Smooth helper
    def smooth(data, w=20):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')

    # ── Plot 1: Reward ──
    axes[0,0].plot(ep, lg.episode_rewards,
                   alpha=0.2, color=color)
    if len(lg.episode_rewards) >= 20:
        axes[0,0].plot(
            range(20, len(lg.episode_rewards)+1),
            smooth(lg.episode_rewards),
            color=color, linewidth=2.5, label=label)

    # ── Plot 2: LAI ──
    axes[0,1].plot(ep, lg.episode_lai,
                   alpha=0.2, color=color)
    if len(lg.episode_lai) >= 20:
        axes[0,1].plot(
            range(20, len(lg.episode_lai)+1),
            smooth(lg.episode_lai),
            color=color, linewidth=2.5, label=label)

    # ── Plot 3: Stress ──
    axes[1,0].plot(ep, lg.episode_stress,
                   alpha=0.2, color=color)
    if len(lg.episode_stress) >= 20:
        axes[1,0].plot(
            range(20, len(lg.episode_stress)+1),
            smooth(lg.episode_stress),
            color=color, linewidth=2.5, label=label)

    # ── Plot 4: Energy ──
    axes[1,1].plot(ep, lg.episode_energy,
                   alpha=0.2, color=color)
    if len(lg.episode_energy) >= 20:
        axes[1,1].plot(
            range(20, len(lg.episode_energy)+1),
            smooth(lg.episode_energy),
            color=color, linewidth=2.5, label=label)

titles = [
    "Training Reward per Episode",
    "Average LAI per Episode",
    "Average Stress per Episode",
    "Total Energy per Episode"
]
ylabels = ["Reward", "LAI", "Stress Score", "Energy Cost"]

for ax, title, ylabel in zip(axes.flat, titles, ylabels):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")

plt.suptitle(
    "Improved Multi-Agent PPO Training Results\n"
    "(200k timesteps, real simulator ranges, 128-128-64 network)",
    fontsize=15, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig("results/improved_training.png", dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Saved results/improved_training.png")

# ══════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 60)
for i, (name, label) in enumerate(zip(TIER_NAMES, tier_labels)):
    lg = loggers[i]
    if lg.episode_rewards:
        last10 = lg.episode_rewards[-10:]
        first10 = lg.episode_rewards[:10]
        improvement = np.mean(last10) - np.mean(first10)
        print(f"\n{label}:")
        print(f"  Episodes trained : {lg.episode_count}")
        print(f"  First 10 avg reward : {np.mean(first10):.3f}")
        print(f"  Last  10 avg reward : {np.mean(last10):.3f}")
        print(f"  Improvement      : {improvement:+.3f}")
        print(f"  Best reward      : {max(lg.episode_rewards):.3f}")
        print(f"  Final avg LAI    : {np.mean(lg.episode_lai[-10:]):.4f}")
        print(f"  Final avg stress : {np.mean(lg.episode_stress[-10:]):.4f}")

print("\n✓ All 3 improved agents saved to models/")
print("✓ Run python phase5_comparison.py next")
print("✓ Then run python app.py for the dashboard")
print("=" * 60)