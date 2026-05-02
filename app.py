from flask import Flask, jsonify, render_template
import numpy as np
import threading
import time
import gl_gym
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import spaces

app = Flask(__name__)

# ══════════════════════════════════════════════
# TIER ENVIRONMENT
# ══════════════════════════════════════════════
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
        info["lai"]      = float(current_lai)
        info["stress"]   = float(stress)
        info["temp"]     = float(temp)
        info["co2"]      = float(co2)
        info["humidity"] = float(humidity)
        info["reward"]   = float(reward)
        return self._flatten(obs), reward, done, truncated, info

    def _flatten(self, obs):
        parts = []
        for key, val in obs.items():
            parts.append(np.array(val, dtype=np.float32).flatten())
        return np.concatenate(parts)


# ══════════════════════════════════════════════
# SIMULATION STATE
# ══════════════════════════════════════════════
sim = {
    "running":    False,
    "step":       0,
    "agents":     None,
    "envs":       None,
    "obs":        None,
    "tiers": {
        "bottom": {"lai":[], "temp":[], "co2":[], "humidity":[], "reward":[], "stress":[]},
        "middle": {"lai":[], "temp":[], "co2":[], "humidity":[], "reward":[], "stress":[]},
        "top":    {"lai":[], "temp":[], "co2":[], "humidity":[], "reward":[], "stress":[]},
    },
    "steps": []
}

TIER_NAMES = ["bottom", "middle", "top"]


def load_agents():
    envs   = []
    agents = []
    obs    = []
    for tier_id, name in enumerate(TIER_NAMES):
        env   = TierEnv(tier_id=tier_id)
        model = PPO.load(f"models/ppo_tier_{name}", env=env)
        o, _  = env.reset()
        envs.append(env)
        agents.append(model)
        obs.append(o)
    sim["envs"]   = envs
    sim["agents"] = agents
    sim["obs"]    = obs
    print("✓ All 3 agents loaded")


def simulation_loop():
    while True:
        if sim["running"] and sim["agents"]:
            for tier_id, name in enumerate(TIER_NAMES):
                action, _ = sim["agents"][tier_id].predict(
                    sim["obs"][tier_id], deterministic=True
                )
                o, reward, done, truncated, info = \
                    sim["envs"][tier_id].step(action)
                sim["obs"][tier_id] = o

                t = sim["tiers"][name]
                t["lai"].append(info.get("lai", 0))
                t["temp"].append(info.get("temp", 20))
                t["co2"].append(info.get("co2", 800))
                t["humidity"].append(info.get("humidity", 80))
                t["reward"].append(info.get("reward", 0))
                t["stress"].append(info.get("stress", 0))

                # Keep last 200 steps only
                for key in t:
                    if len(t[key]) > 200:
                        t[key] = t[key][-200:]

                if done or truncated:
                    o, _ = sim["envs"][tier_id].reset()
                    sim["obs"][tier_id] = o

            sim["step"] += 1
            sim["steps"].append(sim["step"])
            if len(sim["steps"]) > 200:
                sim["steps"] = sim["steps"][-200:]

        time.sleep(0.3)


# ══════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start")
def start():
    if not sim["agents"]:
        load_agents()
    sim["running"] = True
    return jsonify({"status": "started"})


@app.route("/api/stop")
def stop():
    sim["running"] = False
    return jsonify({"status": "stopped"})


@app.route("/api/reset")
def reset():
    sim["running"] = False
    sim["step"]    = 0
    sim["steps"]   = []
    for name in TIER_NAMES:
        for key in sim["tiers"][name]:
            sim["tiers"][name][key] = []
    if sim["envs"]:
        for tier_id in range(3):
            o, _ = sim["envs"][tier_id].reset()
            sim["obs"][tier_id] = o
    return jsonify({"status": "reset"})


@app.route("/api/state")
def state():
    result = {"step": sim["step"], "running": sim["running"], "tiers": {}}
    for name in TIER_NAMES:
        t = sim["tiers"][name]
        result["tiers"][name] = {
            "lai":      t["lai"][-1]      if t["lai"]      else 0,
            "temp":     t["temp"][-1]     if t["temp"]     else 20,
            "co2":      t["co2"][-1]      if t["co2"]      else 800,
            "humidity": t["humidity"][-1] if t["humidity"] else 80,
            "reward":   t["reward"][-1]   if t["reward"]   else 0,
            "stress":   t["stress"][-1]   if t["stress"]   else 0,
            "lai_history":    t["lai"][-50:]     if t["lai"]     else [],
            "temp_history":   t["temp"][-50:]    if t["temp"]    else [],
            "reward_history": t["reward"][-50:]  if t["reward"]  else [],
            "stress_history": t["stress"][-50:]  if t["stress"]  else [],
        }
    result["steps"] = sim["steps"][-50:]
    return jsonify(result)


thread = threading.Thread(target=simulation_loop, daemon=True)
thread.start()

if __name__ == "__main__":
    print("Starting Vertical Farm RL Dashboard...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)