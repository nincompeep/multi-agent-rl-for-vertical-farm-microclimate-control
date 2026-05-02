import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
import os
import gl_gym
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import spaces

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Vertical Farm RL",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }
    .tier-card {
        background: linear-gradient(135deg, #1a2332, #0d1b2a);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        text-align: center;
    }
    .tier-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .tier-top    { border-left: 4px solid #ff7043; }
    .tier-middle { border-left: 4px solid #66bb6a; }
    .tier-bottom { border-left: 4px solid #42a5f5; }
    .metric-box {
        background: #1e2a3a;
        border-radius: 8px;
        padding: 10px;
        margin: 4px;
        text-align: center;
        display: inline-block;
        min-width: 80px;
    }
    .metric-val {
        font-size: 20px;
        font-weight: bold;
        color: #00e676;
    }
    .metric-lbl {
        font-size: 11px;
        color: #90a4ae;
        margin-top: 2px;
    }
    .big-header {
        text-align: center;
        background: linear-gradient(90deg, #1b5e20, #0d47a1);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        height: 42px;
    }
    div[data-testid="metric-container"] {
        background: #1e2a3a;
        border: 1px solid #2d4a6e;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ENVIRONMENT WRAPPER
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
        info["lai"]      = current_lai
        info["stress"]   = stress
        info["temp"]     = temp
        info["co2"]      = co2
        info["humidity"] = humidity
        info["reward"]   = reward
        return self._flatten(obs), reward, done, truncated, info

    def _flatten(self, obs):
        parts = []
        for key, val in obs.items():
            parts.append(np.array(val, dtype=np.float32).flatten())
        return np.concatenate(parts)


# ══════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════
def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized   = True
        st.session_state.running       = False
        st.session_state.step_count    = 0
        st.session_state.models_loaded = False
        for t in ["bottom", "middle", "top"]:
            st.session_state[f"{t}_lai"]      = [0.0]
            st.session_state[f"{t}_temp"]     = [20.0]
            st.session_state[f"{t}_co2"]      = [800.0]
            st.session_state[f"{t}_humidity"] = [80.0]
            st.session_state[f"{t}_reward"]   = [0.0]
            st.session_state[f"{t}_stress"]   = [0.0]
        st.session_state.steps   = [0]
        st.session_state.envs    = None
        st.session_state.agents  = None
        st.session_state.obs     = None

init_state()

TIER_NAMES  = ["bottom", "middle", "top"]
TIER_LABELS = ["Bottom Tier 🔵", "Middle Tier 🟢", "Top Tier 🔴"]
TIER_COLORS = ["#42a5f5", "#66bb6a", "#ff7043"]
TIER_CSS    = ["tier-bottom", "tier-middle", "tier-top"]


# ══════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════
def load_models():
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
    return envs, agents, obs


# ══════════════════════════════════════════════
# SIMULATION STEP
# ══════════════════════════════════════════════
def simulation_step():
    for tier_id, name in enumerate(TIER_NAMES):
        action, _ = st.session_state.agents[tier_id].predict(
            st.session_state.obs[tier_id], deterministic=True
        )
        o, reward, done, truncated, info = \
            st.session_state.envs[tier_id].step(action)
        st.session_state.obs[tier_id] = o
        st.session_state[f"{name}_lai"].append(
            float(info.get("lai", 0)))
        st.session_state[f"{name}_temp"].append(
            float(info.get("temp", 20)))
        st.session_state[f"{name}_co2"].append(
            float(info.get("co2", 800)))
        st.session_state[f"{name}_humidity"].append(
            float(info.get("humidity", 80)))
        st.session_state[f"{name}_reward"].append(float(reward))
        st.session_state[f"{name}_stress"].append(
            float(info.get("stress", 0)))
        if done or truncated:
            o, _ = st.session_state.envs[tier_id].reset()
            st.session_state.obs[tier_id] = o
    st.session_state.step_count += 1
    st.session_state.steps.append(st.session_state.step_count)


# ══════════════════════════════════════════════
# DRAW FARM VISUAL
# ══════════════════════════════════════════════
def draw_farm_visual(key_suffix=""):
    fig = go.Figure()
    for tier_id, (name, label, color) in enumerate(
            zip(TIER_NAMES, TIER_LABELS, TIER_COLORS)):
        y_base = tier_id * 3.5
        lai    = st.session_state[f"{name}_lai"][-1]
        stress = st.session_state[f"{name}_stress"][-1]
        temp   = st.session_state[f"{name}_temp"][-1]
        fig.add_shape(type="rect",
            x0=0, x1=10, y0=y_base, y1=y_base+3,
            fillcolor="#1a2332",
            line=dict(color=color, width=2))
        plant_color = "#ef5350" if stress > 0.5 else \
                      "#ffa726" if stress > 0.1 else "#66bb6a"
        plant_size  = max(0.3, min(1.2, lai * 2))
        for i in range(5):
            x_pos = 1 + i * 2
            fig.add_shape(type="line",
                x0=x_pos, x1=x_pos,
                y0=y_base+0.3, y1=y_base+0.3+plant_size,
                line=dict(color="#795548", width=3))
            fig.add_shape(type="circle",
                x0=x_pos-plant_size*0.6,
                x1=x_pos+plant_size*0.6,
                y0=y_base+0.3+plant_size*0.5,
                y1=y_base+0.3+plant_size*1.5,
                fillcolor=plant_color,
                line=dict(color=plant_color, width=1),
                opacity=0.85)
        led_color = "#fff176"
        fig.add_shape(type="rect",
            x0=0.2, x1=9.8,
            y0=y_base+2.75, y1=y_base+2.95,
            fillcolor=led_color,
            line=dict(color=led_color, width=1),
            opacity=0.9)
        fig.add_annotation(
            x=5, y=y_base+0.15,
            text=f"<b>{label}</b> | "
                 f"LAI: {lai:.3f} | "
                 f"Temp: {temp:.1f}°C | "
                 f"Stress: {'🔴 HIGH' if stress>0.5 else '🟡 MID' if stress>0.1 else '🟢 OK'}",
            showarrow=False,
            font=dict(color=color, size=12),
            xanchor="center")
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(
            text="🌱 Live Vertical Farm — 3 Tier View",
            font=dict(color="white", size=16),
            x=0.5),
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[-0.3, 11.5]),
        showlegend=False)
    return fig


# ══════════════════════════════════════════════
# DRAW LIVE CHARTS
# ══════════════════════════════════════════════
def draw_live_charts(key_suffix=""):
    steps = st.session_state.steps
    fig   = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "LAI (Leaf Area Index)",
            "Temperature (°C)",
            "Reward per Step",
            "Plant Stress Score"),
        vertical_spacing=0.18,
        horizontal_spacing=0.1)
    for name, label, color in zip(TIER_NAMES, TIER_LABELS, TIER_COLORS):
        fig.add_trace(go.Scatter(
            x=steps, y=st.session_state[f"{name}_lai"],
            name=label, line=dict(color=color, width=2),
            showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=steps, y=st.session_state[f"{name}_temp"],
            name=label, line=dict(color=color, width=2),
            showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=steps, y=st.session_state[f"{name}_reward"],
            name=label, line=dict(color=color, width=2),
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=steps, y=st.session_state[f"{name}_stress"],
            name=label, line=dict(color=color, width=2),
            showlegend=False), row=2, col=2)
    fig.add_hline(y=18, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)", row=1, col=2)
    fig.add_hline(y=26, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)", row=1, col=2)
    fig.update_layout(
        height=480,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white", size=11),
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(
            orientation="h", y=1.08, x=0.5,
            xanchor="center", font=dict(size=11)))
    fig.update_xaxes(
        gridcolor="#1e2a3a", zerolinecolor="#2d4a6e",
        title_text="Simulation Step")
    fig.update_yaxes(
        gridcolor="#1e2a3a", zerolinecolor="#2d4a6e")
    return fig


# ══════════════════════════════════════════════
# DRAW GAUGE
# ══════════════════════════════════════════════
def draw_gauge(value, title, min_val, max_val,
               opt_min, opt_max, unit):
    color = "#00e676" if opt_min <= value <= opt_max \
            else "#ffa726" if abs(
                value - (opt_min + opt_max) / 2) < (opt_max - opt_min) \
            else "#ef5350"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(
            text=f"{title}<br>"
                 f"<span style='font-size:11px'>{unit}</span>",
            font=dict(color="white", size=13)),
        number=dict(font=dict(color=color, size=20)),
        gauge=dict(
            axis=dict(
                range=[min_val, max_val],
                tickcolor="white",
                tickfont=dict(color="white", size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#1e2a3a",
            bordercolor="#2d4a6e",
            steps=[
                dict(range=[min_val, opt_min],
                     color="rgba(239,83,80,0.2)"),
                dict(range=[opt_min, opt_max],
                     color="rgba(0,230,118,0.15)"),
                dict(range=[opt_max, max_val],
                     color="rgba(239,83,80,0.2)")],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.75, value=value))))
    fig.update_layout(
        height=180,
        margin=dict(l=15, r=15, t=40, b=10),
        paper_bgcolor="#0e1117",
        font=dict(color="white"))
    return fig


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌱 Farm Control Panel")
    st.markdown("---")
    if not st.session_state.models_loaded:
        if st.button("⚡ Load Trained Agents", type="primary"):
            with st.spinner("Loading 3 PPO agents..."):
                try:
                    envs, agents, obs = load_models()
                    st.session_state.envs          = envs
                    st.session_state.agents        = agents
                    st.session_state.obs           = obs
                    st.session_state.models_loaded = True
                    st.success("✅ All 3 agents loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading models: {e}")
    else:
        st.success("✅ Agents loaded and ready")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "▶ Start",
            disabled=not st.session_state.models_loaded,
            type="primary")
    with col2:
        stop_btn = st.button("⏹ Stop")
    reset_btn = st.button("🔄 Reset Simulation")
    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    if reset_btn:
        st.session_state.running    = False
        st.session_state.step_count = 0
        st.session_state.steps      = [0]
        for t in TIER_NAMES:
            st.session_state[f"{t}_lai"]      = [0.0]
            st.session_state[f"{t}_temp"]     = [20.0]
            st.session_state[f"{t}_co2"]      = [800.0]
            st.session_state[f"{t}_humidity"] = [80.0]
            st.session_state[f"{t}_reward"]   = [0.0]
            st.session_state[f"{t}_stress"]   = [0.0]
        if st.session_state.models_loaded:
            for tier_id in range(3):
                o, _ = st.session_state.envs[tier_id].reset()
                st.session_state.obs[tier_id] = o
        st.rerun()
    st.markdown("---")
    speed = st.slider("Simulation Speed", 1, 10, 3)
    st.markdown("---")
    st.markdown("### 📊 Status")
    status = "🟢 RUNNING" if st.session_state.running \
             else "🔴 STOPPED" if st.session_state.models_loaded \
             else "🟡 WAITING"
    st.markdown(f"**Simulation:** {status}")
    st.markdown(f"**Total Steps:** {st.session_state.step_count}")
    st.markdown(
        f"**Agents Active:** "
        f"{'3' if st.session_state.models_loaded else '0'}")
    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **Algorithm:** PPO
    - **Tiers:** 3
    - **Reward:** Phenotyping
    - **Simulator:** GreenLight-Gym
    """)


# ══════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class='big-header'>
    <h1 style='color:white; margin:0; font-size:28px;'>
        🌱 Vertical Farm Multi-Agent RL Dashboard
    </h1>
    <p style='color:#90caf9; margin:4px 0 0;'>
        Per-Tier Microclimate Control using PPO
        with Phenotyping Rewards
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏭 Live Simulation",
    "📊 Live Charts",
    "🌡️ Tier Gauges",
    "🏆 Comparison",
    "📋 About"
])

# ── TAB 1 ──────────────────────────────────────
with tab1:
    if not st.session_state.models_loaded:
        st.info("👈 Click **Load Trained Agents** in the sidebar to begin")

    farm_ph = st.empty()
    farm_ph.plotly_chart(
        draw_farm_visual(),
        use_container_width=True,
        key=f"farm_init_{st.session_state.step_count}")

    st.markdown("---")
    st.markdown("### 📡 Real-Time Tier Metrics")
    metric_cols = st.columns(3)
    metric_phs  = []
    for col in metric_cols:
        with col:
            metric_phs.append(st.empty())

    def render_metrics():
        for i, (name, label, color, css) in enumerate(
                zip(TIER_NAMES, TIER_LABELS,
                    TIER_COLORS, TIER_CSS)):
            lai    = st.session_state[f"{name}_lai"][-1]
            temp   = st.session_state[f"{name}_temp"][-1]
            co2    = st.session_state[f"{name}_co2"][-1]
            hum    = st.session_state[f"{name}_humidity"][-1]
            stress = st.session_state[f"{name}_stress"][-1]
            reward = st.session_state[f"{name}_reward"][-1]
            health = "🟢 Healthy"    if stress < 0.1 \
                     else "🟡 Mild"  if stress < 0.5 \
                     else "🔴 Stress"
            r_color = "#00e676" if reward > 0 else "#ef5350"
            metric_phs[i].markdown(f"""
            <div class='tier-card {css}'>
                <div class='tier-title' style='color:{color}'>
                    {label}
                </div>
                <div>
                    <div class='metric-box'>
                        <div class='metric-val'>{lai:.4f}</div>
                        <div class='metric-lbl'>LAI</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'>{temp:.1f}°C</div>
                        <div class='metric-lbl'>Temp</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'>{co2:.0f}</div>
                        <div class='metric-lbl'>CO₂ ppm</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'>{hum:.1f}%</div>
                        <div class='metric-lbl'>Humidity</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'
                             style='color:{r_color}'>
                            {reward:.3f}
                        </div>
                        <div class='metric-lbl'>Reward</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'
                             style='font-size:13px'>
                            {health}
                        </div>
                        <div class='metric-lbl'>Health</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    render_metrics()

# ── TAB 2 ──────────────────────────────────────
with tab2:
    st.markdown("### 📈 Live Performance Charts")
    charts_ph = st.empty()
    charts_ph.plotly_chart(
        draw_live_charts(),
        use_container_width=True,
        key=f"charts_init_{st.session_state.step_count}")

    if st.session_state.step_count > 5:
        st.markdown("### 📊 Running Statistics")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Avg LAI",
                np.mean([st.session_state[f"{n}_lai"][-1]
                         for n in TIER_NAMES]))
        with sc2:
            st.metric("Avg Temp (°C)",
                round(np.mean([st.session_state[f"{n}_temp"][-1]
                               for n in TIER_NAMES]), 1))
        with sc3:
            st.metric("Avg Stress",
                round(np.mean([st.session_state[f"{n}_stress"][-1]
                               for n in TIER_NAMES]), 4))
        with sc4:
            st.metric("Avg Reward",
                round(np.mean([st.session_state[f"{n}_reward"][-1]
                               for n in TIER_NAMES]), 4))

# ── TAB 3 ──────────────────────────────────────
with tab3:
    st.markdown("### 🌡️ Per-Tier Climate Gauges")
    gauge_phs = []
    for i, (name, label, color) in enumerate(
            zip(TIER_NAMES, TIER_LABELS, TIER_COLORS)):
        st.markdown(
            f"<h4 style='color:{color}'>{label}</h4>",
            unsafe_allow_html=True)
        gcols   = st.columns(4)
        row_phs = []
        for col in gcols:
            with col:
                row_phs.append(st.empty())
        gauge_phs.append(row_phs)

    GAUGE_CONFIGS = [
        ("temp",     "Temperature", 0,   50,   18,   26,   "°C"),
        ("co2",      "CO₂",         200, 1500, 600,  1000, "ppm"),
        ("humidity", "Humidity",    20,  100,  70,   90,   "%"),
        ("lai",      "LAI",         0,   1,    0.3,  0.8,  "index"),
    ]

    def render_gauges(suffix=""):
        for i, name in enumerate(TIER_NAMES):
            for j, (key, title, mn, mx,
                    omn, omx, unit) in enumerate(GAUGE_CONFIGS):
                val = st.session_state[f"{name}_{key}"][-1]
                gauge_phs[i][j].plotly_chart(
                    draw_gauge(val, title, mn, mx, omn, omx, unit),
                    use_container_width=True,
                    key=f"gauge_{name}_{key}_{suffix}")

    render_gauges(suffix=f"init_{st.session_state.step_count}")

# ── TAB 4 ──────────────────────────────────────
with tab4:
    st.markdown("### 🏆 Approach Comparison")
    df = pd.DataFrame({
        "Method": [
            "🔴 Rule-Based",
            "🔵 Single-Agent PPO",
            "🟢 Multi-Agent PPO (ours)"
        ],
        "Agents":           ["1 fixed", "1 learning", "3 learning"],
        "Reward Signal":    ["None", "Phenotyping", "Phenotyping per tier"],
        "Per-Tier Control": ["❌", "❌", "✅"],
        "Phenotyping":      ["❌", "✅", "✅"],
        "Multi-Agent":      ["❌", "❌", "✅"],
        "Best For": [
            "Simple fixed environments",
            "Single zone farms",
            "Multi-tier vertical farms ✅"
        ]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("**🔴 Rule-Based**\n\n"
                 "Fixed setpoints. No learning. "
                 "Ignores tier differences.")
    with c2:
        st.info("**🔵 Single-Agent PPO**\n\n"
                "Learns from experience. "
                "Phenotyping reward. "
                "One policy for whole farm.")
    with c3:
        st.success("**🟢 Multi-Agent PPO ✅**\n\n"
                   "3 specialised agents. "
                   "Per-tier control. "
                   "Lowest stress, highest LAI.")
    st.markdown("---")
    img_path = "results/final_comparison.png"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning("Run phase5_comparison.py to generate results.")

# ── TAB 5 ──────────────────────────────────────
with tab5:
    st.markdown("### 📋 Project Details")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### Problem
        Traditional vertical farms use static rule-based
        controllers that ignore microclimate variation
        across tiers and use dry biomass as reward —
        which is sparse, delayed and destructive.

        #### Solution
        **Multi-Agent RL with Phenotyping Reward:**
        - Independent PPO agent per shelf tier
        - Continuous reward from LAI + stress indicator
        - Non-destructive plant monitoring
        """)
        st.latex(r"""
        R_t = \underbrace{(LAI_t - LAI_{t-1}) \times 20}_\text{growth}
            - \underbrace{S_{temp} + S_{CO_2} + S_{hum}}_\text{stress}
            - \underbrace{\bar{|u_t|} \times 0.05}_\text{energy}
        """)
    with c2:
        st.markdown("""
        #### Tech Stack
        | Component | Tool |
        |---|---|
        | Simulation | GreenLight-Gym |
        | RL Algorithm | PPO |
        | RL Library | Stable-Baselines3 |
        | Language | Python 3.12 |
        | Dashboard | Streamlit + Plotly |

        #### Optimal Climate Ranges
        | Variable | Min | Max |
        |---|---|---|
        | Temperature | 18°C | 26°C |
        | CO₂ | 600 ppm | 1000 ppm |
        | Humidity | 70% | 90% |

        #### Tier Microclimate Offsets
        | Tier | Temp Offset |
        |---|---|
        | Bottom | -2°C |
        | Middle | 0°C |
        | Top | +2°C |
        """)

# ══════════════════════════════════════════════
# SIMULATION LOOP
# ══════════════════════════════════════════════
if st.session_state.running and st.session_state.models_loaded:
    simulation_step()
    time.sleep(1.0 / speed)

    farm_ph.plotly_chart(
        draw_farm_visual(),
        use_container_width=True,
        key=f"farm_{st.session_state.step_count}")

    charts_ph.plotly_chart(
        draw_live_charts(),
        use_container_width=True,
        key=f"charts_{st.session_state.step_count}")

    render_metrics()
    render_gauges(suffix=str(st.session_state.step_count))

    st.rerun()  