import gl_gym
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from gymnasium import spaces

# ══════════════════════════════════════════════
# STEP 1 — All wrappers needed
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


class TierEnv(gym.Wrapper):
    OPTIMAL_TEMP_MIN  = 18.0
    OPTIMAL_TEMP_MAX  = 26.0
    OPTIMAL_CO2_MIN   = 600.0
    OPTIMAL_CO2_MAX   = 1000.0
    OPTIMAL_HUM_MIN   = 70.0
    OPTIMAL_HUM_MAX   = 90.0
    TIER_TEMP_OFFSET  = {0: -2.0, 1: 0.0, 2: +2.0}

    def __init__(self, tier_id):
        base_env = gym.make("gl_gym/GreenLightTomato-v0")
        super().__init__(base_env)
        self.tier_id      = tier_id
        self.previous_lai = 0.0
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
        self.previous_lai = 0.0
        return self._flatten(obs), info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        crop    = np.array(obs["BasicCropObservations"]).flatten()
        climate = np.array(obs["IndoorClimateObservations"]).flatten()
        control = np.array(obs["ControlObservations"]).flatten()
        lai_proxy = crop[2]
        co2       = climate[0]
        temp      = climate[1] + self.TIER_TEMP_OFFSET[self.tier_id]
        humidity  = climate[2]
        current_lai  = lai_proxy / 5000.0
        lai_gain     = current_lai - self.previous_lai
        lai_reward   = lai_gain * 20.0
        temp_stress  = max(0, self.OPTIMAL_TEMP_MIN - temp) / 10.0 \
                     + max(0, temp - self.OPTIMAL_TEMP_MAX) / 10.0
        co2_stress   = max(0, self.OPTIMAL_CO2_MIN - co2) / 500.0 \
                     + max(0, co2 - self.OPTIMAL_CO2_MAX) / 500.0
        hum_stress   = max(0, self.OPTIMAL_HUM_MIN - humidity) / 20.0 \
                     + max(0, humidity - self.OPTIMAL_HUM_MAX) / 20.0
        stress       = temp_stress + co2_stress + hum_stress
        energy       = np.mean(np.abs(control)) * 0.05
        reward       = lai_reward - stress - energy
        self.previous_lai = current_lai
        info["lai"]    = current_lai
        info["stress"] = stress
        info["temp"]   = temp
        return self._flatten(obs), reward, done, truncated, info

    def _flatten(self, obs):
        parts = []
        for key, val in obs.items():
            parts.append(np.array(val, dtype=np.float32).flatten())
        return np.concatenate(parts)


# ══════════════════════════════════════════════
# STEP 2 — Rule-based controller
# Fixed setpoints — no learning at all
# ══════════════════════════════════════════════
def run_rule_based(n_episodes=20):
    """
    Simple rule-based controller:
    Always sets controls to fixed optimal values.
    No learning — just hardcoded rules.
    """
    print("\nRunning Rule-Based Controller...")
    env = gym.make("gl_gym/GreenLightTomato-v0")

    all_rewards  = []
    all_lai      = []
    all_stress   = []

    OPTIMAL_TEMP = 18.0
    OPTIMAL_CO2  = 600.0
    OPTIMAL_HUM  = 70.0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_lai    = []
        ep_stress = []
        done      = False
        previous_lai = 0.0

        while not done:
            # Fixed action — always set to midpoint (0.0)
            # representing "maintain optimal setpoint"
            action = np.zeros(env.action_space.shape)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            # Compute phenotyping metrics manually
            crop    = np.array(obs["BasicCropObservations"]).flatten()
            climate = np.array(obs["IndoorClimateObservations"]).flatten()
            lai_proxy = crop[2]
            temp      = climate[1]
            co2       = climate[0]
            humidity  = climate[2]

            current_lai = lai_proxy / 5000.0
            lai_gain    = current_lai - previous_lai
            lai_reward  = lai_gain * 20.0

            temp_stress = max(0, OPTIMAL_TEMP - temp) / 10.0 \
                        + max(0, temp - 26.0) / 10.0
            co2_stress  = max(0, OPTIMAL_CO2 - co2) / 500.0 \
                        + max(0, co2 - 1000.0) / 500.0
            hum_stress  = max(0, OPTIMAL_HUM - humidity) / 20.0 \
                        + max(0, humidity - 90.0) / 20.0
            stress      = temp_stress + co2_stress + hum_stress

            pheno_reward = lai_reward - stress
            ep_reward   += pheno_reward
            ep_lai.append(current_lai)
            ep_stress.append(stress)
            previous_lai = current_lai

        all_rewards.append(ep_reward)
        all_lai.append(np.mean(ep_lai))
        all_stress.append(np.mean(ep_stress))
        print(f"  Rule-based Episode {ep+1:3d} | "
              f"Reward: {ep_reward:.3f} | "
              f"Avg LAI: {np.mean(ep_lai):.4f}")

    env.close()
    return all_rewards, all_lai, all_stress


# ══════════════════════════════════════════════
# STEP 3 — Evaluate single PPO agent
# ══════════════════════════════════════════════
def run_single_agent(n_episodes=20):
    print("\nRunning Single-Agent PPO (phenotyping reward)...")
    env   = TierEnv(tier_id=1)  # middle tier
    model = PPO.load("models/ppo_phenotyping", env=env)

    all_rewards = []
    all_lai     = []
    all_stress  = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_lai    = []
        ep_stress = []
        done      = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            ep_reward += reward
            ep_lai.append(info.get("lai", 0))
            ep_stress.append(info.get("stress", 0))

        all_rewards.append(ep_reward)
        all_lai.append(np.mean(ep_lai))
        all_stress.append(np.mean(ep_stress))
        print(f"  Single-Agent Episode {ep+1:3d} | "
              f"Reward: {ep_reward:.3f} | "
              f"Avg LAI: {np.mean(ep_lai):.4f}")

    env.close()
    return all_rewards, all_lai, all_stress


# ══════════════════════════════════════════════
# STEP 4 — Evaluate multi-agent system
# ══════════════════════════════════════════════
def run_multi_agent(n_episodes=20):
    print("\nRunning Multi-Agent PPO (3 tier agents)...")

    tier_names = ["bottom", "middle", "top"]
    agents     = []
    envs       = []

    for tier_id in range(3):
        env   = TierEnv(tier_id=tier_id)
        model = PPO.load(
            f"models/ppo_tier_{tier_names[tier_id]}",
            env=env
        )
        agents.append(model)
        envs.append(env)

    all_rewards = []
    all_lai     = []
    all_stress  = []

    for ep in range(n_episodes):
        # Reset all tier environments
        observations = []
        for env in envs:
            obs, _ = env.reset()
            observations.append(obs)

        ep_reward = 0
        ep_lai    = []
        ep_stress = []
        done      = False
        step      = 0

        while not done and step < 1000:
            step += 1
            tier_rewards  = []
            tier_lai      = []
            tier_stress   = []

            # Each agent acts independently in its tier
            for tier_id in range(3):
                action, _ = agents[tier_id].predict(
                    observations[tier_id], deterministic=True
                )
                obs, reward, d, t, info = envs[tier_id].step(action)
                observations[tier_id] = obs
                tier_rewards.append(reward)
                tier_lai.append(info.get("lai", 0))
                tier_stress.append(info.get("stress", 0))
                if d or t:
                    done = True

            # Average across all tiers
            ep_reward += np.mean(tier_rewards)
            ep_lai.append(np.mean(tier_lai))
            ep_stress.append(np.mean(tier_stress))

        all_rewards.append(ep_reward)
        all_lai.append(np.mean(ep_lai))
        all_stress.append(np.mean(ep_stress))
        print(f"  Multi-Agent  Episode {ep+1:3d} | "
              f"Reward: {ep_reward:.3f} | "
              f"Avg LAI: {np.mean(ep_lai):.4f}")

    for env in envs:
        env.close()
    return all_rewards, all_lai, all_stress


# ══════════════════════════════════════════════
# STEP 5 — Run all 3 evaluations
# ══════════════════════════════════════════════
print("=" * 55)
print("Phase 5 — Final Comparison of All Approaches")
print("=" * 55)

N_EPISODES = 20

rb_rewards,  rb_lai,  rb_stress  = run_rule_based(N_EPISODES)
sa_rewards,  sa_lai,  sa_stress  = run_single_agent(N_EPISODES)
ma_rewards,  ma_lai,  ma_stress  = run_multi_agent(N_EPISODES)

# ══════════════════════════════════════════════
# STEP 6 — Print summary table
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print("FINAL RESULTS SUMMARY")
print("=" * 55)
print(f"{'Method':<25} {'Avg Reward':>12} {'Avg LAI':>10} {'Avg Stress':>12}")
print("-" * 55)
print(f"{'Rule-Based':<25} "
      f"{np.mean(rb_rewards):>12.3f} "
      f"{np.mean(rb_lai):>10.4f} "
      f"{np.mean(rb_stress):>12.4f}")
print(f"{'Single-Agent PPO':<25} "
      f"{np.mean(sa_rewards):>12.3f} "
      f"{np.mean(sa_lai):>10.4f} "
      f"{np.mean(sa_stress):>12.4f}")
print(f"{'Multi-Agent PPO (ours)':<25} "
      f"{np.mean(ma_rewards):>12.3f} "
      f"{np.mean(ma_lai):>10.4f} "
      f"{np.mean(ma_stress):>12.4f}")
print("=" * 55)

# ══════════════════════════════════════════════
# STEP 7 — Plot all comparison graphs
# ══════════════════════════════════════════════
os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors  = ['#F44336', '#2196F3', '#4CAF50']
labels  = ['Rule-Based', 'Single-Agent PPO', 'Multi-Agent PPO (ours)']
episodes = range(1, N_EPISODES + 1)

# Graph 1 — Reward per episode
axes[0,0].plot(episodes, rb_rewards,
               color=colors[0], marker='o', markersize=4,
               linewidth=2, label=labels[0])
axes[0,0].plot(episodes, sa_rewards,
               color=colors[1], marker='s', markersize=4,
               linewidth=2, label=labels[1])
axes[0,0].plot(episodes, ma_rewards,
               color=colors[2], marker='^', markersize=4,
               linewidth=2, label=labels[2])
axes[0,0].set_title("Total Reward per Episode")
axes[0,0].set_xlabel("Episode")
axes[0,0].set_ylabel("Total Reward")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Graph 2 — Average LAI per episode
axes[0,1].plot(episodes, rb_lai,
               color=colors[0], marker='o', markersize=4,
               linewidth=2, label=labels[0])
axes[0,1].plot(episodes, sa_lai,
               color=colors[1], marker='s', markersize=4,
               linewidth=2, label=labels[1])
axes[0,1].plot(episodes, ma_lai,
               color=colors[2], marker='^', markersize=4,
               linewidth=2, label=labels[2])
axes[0,1].set_title("Average LAI (Leaf Area Index) per Episode")
axes[0,1].set_xlabel("Episode")
axes[0,1].set_ylabel("Average LAI")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Graph 3 — Stress per episode
axes[1,0].plot(episodes, rb_stress,
               color=colors[0], marker='o', markersize=4,
               linewidth=2, label=labels[0])
axes[1,0].plot(episodes, sa_stress,
               color=colors[1], marker='s', markersize=4,
               linewidth=2, label=labels[1])
axes[1,0].plot(episodes, ma_stress,
               color=colors[2], marker='^', markersize=4,
               linewidth=2, label=labels[2])
axes[1,0].set_title("Average Plant Stress per Episode")
axes[1,0].set_xlabel("Episode")
axes[1,0].set_ylabel("Stress Score (lower = better)")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Graph 4 — Final bar chart summary
x       = np.arange(3)
metrics = [
    [np.mean(rb_rewards),  np.mean(sa_rewards),  np.mean(ma_rewards)],
    [np.mean(rb_lai)*100,  np.mean(sa_lai)*100,  np.mean(ma_lai)*100],
    [np.mean(rb_stress),   np.mean(sa_stress),   np.mean(ma_stress)],
]
metric_names = ["Avg Reward", "Avg LAI (×100)", "Avg Stress"]
bar_width    = 0.25

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    bars = axes[1,1].bar(
        x + i * bar_width, metric,
        bar_width, label=name,
        alpha=0.8
    )

axes[1,1].set_title("Overall Comparison — All Metrics")
axes[1,1].set_xticks(x + bar_width)
axes[1,1].set_xticklabels(labels, fontsize=9)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.suptitle(
    "Phase 5 — Final Comparison: Rule-Based vs Single-Agent vs Multi-Agent",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig("results/final_comparison.png", dpi=150)
plt.show()
print("\n✓ Final comparison plot saved to results/final_comparison.png")

print("\n" + "=" * 55)
print("Phase 5 Complete!")
print("Your project now has full experimental results.")
print("=" * 55)