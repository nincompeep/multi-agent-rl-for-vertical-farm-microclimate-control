# 🌱 FarmMind — Multi-Agent RL for Vertical Farm Microclimate Control

A multi-agent reinforcement learning framework for per-tier 
microclimate control in vertical farms, using PPO with 
phenotyping-based reward shaping.

## Project Overview

Traditional vertical farms use static rule-based controllers 
that treat the entire farm as one uniform zone, ignoring 
microclimate variations across shelf tiers. This project 
proposes a multi-agent RL approach where each shelf tier 
is managed by an independent PPO agent.

## Key Innovation

- **Phenotyping Reward** — Uses Leaf Area Index (LAI) and 
  chlorophyll stress as continuous non-destructive reward signals
- **Per-Tier Control** — 3 independent PPO agents, one per shelf
- **Real-Time Dashboard** — Flask + Chart.js live visualisation

## Tech Stack

| Component | Technology |
|---|---|
| Simulation | GreenLight-Gym |
| RL Algorithm | PPO |
| RL Library | Stable-Baselines3 |
| Language | Python 3.12 |
| Backend | Flask |
| Frontend | HTML / CSS / JS |
| Charts | Chart.js |

## Project Structure
mini project/
├── GreenLight-Gym2/          # Simulator
├── models/                   # Trained PPO agents
├── results/                  # Training graphs
├── static/
│   ├── css/style.css         # Frontend styles
│   └── js/main.js            # Frontend logic
├── templates/
│   └── index.html            # Main webpage
├── app.py                    # Flask backend
├── train_baseline.py         # Phase 2 training
├── phase3_rewards.py         # Phase 3 training
├── phase4_multiagent.py      # Phase 4 training
├── phase5_comparison.py      # Phase 5 evaluation
├── retrain_improved.py       # Improved retraining
├── requirements.txt          # Dependencies
└── README.md                 # This file

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/farmmind.git
cd farmmind

# 2. Create virtual environment
python -m venv farm_env
farm_env\Scripts\activate  # Windows
source farm_env/bin/activate  # Mac/Linux

# 3. Install GreenLight-Gym
git clone https://github.com/BartvLaatum/GreenLight-Gym2.git
cd GreenLight-Gym2
pip install -e .
cd ..

# 4. Install dependencies
pip install -r requirements.txt
```

## Running the Project

### Train agents
```bash
python retrain_improved.py
```

### Run comparison
```bash
python phase5_comparison.py
```

### Launch dashboard
```bash
python app.py
```

Then open http://localhost:5000 in your browser.

## Results

| Method | Avg LAI | Avg Stress | Avg Reward |
|---|---|---|---|
| Rule-Based | Baseline | High | Lowest |
| Single-Agent PPO | +12% | Medium | Better |
| Multi-Agent PPO (ours) | +23% | Low | Best |

## License
MIT License