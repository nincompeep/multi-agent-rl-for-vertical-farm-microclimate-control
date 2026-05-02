// ══════════════════════════════════════════════
// CHART SETUP
// ══════════════════════════════════════════════
const COLORS = {
  bottom: "#42a5f5",
  middle: "#00c853",
  top:    "#ff7043"
};
const LABELS = {
  bottom: "Bottom Tier",
  middle: "Middle Tier",
  top:    "Top Tier"
};

function makeChart(id, label, yLabel) {
  const ctx = document.getElementById(id).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: ["bottom","middle","top"].map(t => ({
        label: LABELS[t],
        data: [],
        borderColor: COLORS[t],
        backgroundColor: COLORS[t] + "18",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: false
      }))
    },
    options: {
      responsive: true,
      animation: false,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: {
          labels: { color: "#7cb87c", font: { size: 12 } }
        }
      },
      scales: {
        x: {
          ticks: { color: "#7cb87c", maxTicksLimit: 8 },
          grid:  { color: "#1a3a1a" }
        },
        y: {
          title: { display: true, text: yLabel, color: "#7cb87c" },
          ticks: { color: "#7cb87c" },
          grid:  { color: "#1a3a1a" }
        }
      }
    }
  });
}

const charts = {
  lai:    makeChart("laiChart",    "LAI",         "LAI"),
  temp:   makeChart("tempChart",   "Temperature", "°C"),
  reward: makeChart("rewardChart", "Reward",      "Reward"),
  stress: makeChart("stressChart", "Stress",      "Stress")
};


// ══════════════════════════════════════════════
// HERO FARM CANVAS
// ══════════════════════════════════════════════
function drawHeroFarm() {
  const canvas = document.getElementById("heroFarm");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const tiers = [
    { y: 260, color: "#ff7043", label: "Top Tier",    dot: "#ff7043" },
    { y: 160, color: "#00c853", label: "Middle Tier", dot: "#00c853" },
    { y: 60,  color: "#42a5f5", label: "Bottom Tier", dot: "#42a5f5" }
  ];

  tiers.forEach(({ y, color, label, dot }) => {
    // Shelf background
    ctx.fillStyle = "#0d1a0d";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    roundRect(ctx, 10, y, 400, 80, 8);
    ctx.fill(); ctx.stroke();

    // LED strip
    const grad = ctx.createLinearGradient(20, y+8, 390, y+8);
    grad.addColorStop(0, "rgba(255,245,100,0.2)");
    grad.addColorStop(0.5, "rgba(255,245,100,0.9)");
    grad.addColorStop(1, "rgba(255,245,100,0.2)");
    ctx.fillStyle = grad;
    ctx.fillRect(20, y+6, 370, 5);

    // Plants
    const plantCount = 8;
    const lai = laiValues[label.split(" ")[0].toLowerCase()] || 0.5;
    for (let i = 0; i < plantCount; i++) {
      const px = 30 + i * 48;
      const stemH = 20 + lai * 30;
      const headR = 8 + lai * 10;
      const stress = stressValues[label.split(" ")[0].toLowerCase()] || 0;
      const plantColor = stress > 0.5 ? "#ef5350"
                       : stress > 0.1 ? "#ffa726"
                       : "#00c853";

      // Stem
      ctx.fillStyle = "#5d4037";
      ctx.fillRect(px - 1.5, y + 70 - stemH, 3, stemH);

      // Head
      ctx.fillStyle = plantColor;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.arc(px, y + 70 - stemH - headR * 0.5, headR, 0, Math.PI*2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Tier label
    ctx.fillStyle = color;
    ctx.font = "bold 12px Segoe UI";
    ctx.fillText(label, 20, y + 96);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

let laiValues    = { bottom: 0.5, middle: 0.5, top: 0.5 };
let stressValues = { bottom: 0.0, middle: 0.0, top: 0.0 };
drawHeroFarm();


// ══════════════════════════════════════════════
// PLANT VISUAL IN TIERS
// ══════════════════════════════════════════════
function renderPlants(tierId, lai, stress) {
  const container = document.getElementById(`plants-${tierId}`);
  if (!container) return;
  container.innerHTML = "";
  const count  = 8;
  const stemH  = Math.max(16, Math.min(52, lai * 80));
  const headSz = Math.max(14, Math.min(36, lai * 50));
  const color  = stress > 0.5 ? "#ef5350"
               : stress > 0.1 ? "#ffa726"
               : "#00c853";
  for (let i = 0; i < count; i++) {
    const plant = document.createElement("div");
    plant.className = "plant";
    plant.innerHTML = `
      <div class="plant-head"
           style="width:${headSz}px;height:${headSz}px;
                  background:${color};opacity:0.85;
                  border-radius:50%"></div>
      <div class="plant-stem"
           style="height:${stemH}px"></div>
    `;
    container.appendChild(plant);
  }
}

["top","mid","bot"].forEach(t => renderPlants(t, 0.3, 0));


// ══════════════════════════════════════════════
// CONTROLS
// ══════════════════════════════════════════════
function startSim() {
  fetch("/api/start")
    .then(r => r.json())
    .then(() => {
      document.getElementById("simStatus").className = "status-running";
      document.getElementById("simStatus").textContent = "● RUNNING";
    });
}

function stopSim() {
  fetch("/api/stop")
    .then(r => r.json())
    .then(() => {
      document.getElementById("simStatus").className = "status-stopped";
      document.getElementById("simStatus").textContent = "● STOPPED";
    });
}

function resetSim() {
  fetch("/api/reset").then(r => r.json()).then(() => {
    document.getElementById("simStatus").className = "status-waiting";
    document.getElementById("simStatus").textContent = "● WAITING";
    document.getElementById("stepCount").textContent = "0";
    Object.values(charts).forEach(c => {
      c.data.labels = [];
      c.data.datasets.forEach(d => d.data = []);
      c.update();
    });
  });
}


// ══════════════════════════════════════════════
// UPDATE LOOP
// ══════════════════════════════════════════════
const TIER_MAP = {
  bottom: { prefix:"bot", idx:2 },
  middle: { prefix:"mid", idx:1 },
  top:    { prefix:"top", idx:0 }
};

const CHART_KEYS = ["lai","temp","reward","stress"];

function updateUI(data) {
  document.getElementById("stepCount").textContent = data.step;

  const steps = data.steps;

  CHART_KEYS.forEach(key => {
    charts[key].data.labels = steps;
  });

  Object.entries(data.tiers).forEach(([name, t], i) => {
    const { prefix } = TIER_MAP[name];
    const shortName  = name;

    // Update metrics
    document.getElementById(`${prefix}-lai`).textContent =
      t.lai.toFixed(4);
    document.getElementById(`${prefix}-temp`).textContent =
      t.temp.toFixed(1);
    document.getElementById(`${prefix}-co2`).textContent =
      Math.round(t.co2);
    document.getElementById(`${prefix}-hum`).textContent =
      t.humidity.toFixed(1);

    const rewardEl = document.getElementById(`${prefix}-reward`);
    rewardEl.textContent = t.reward.toFixed(3);
    rewardEl.style.color = t.reward > 0 ? "#00c853" : "#ef5350";

    // Health badge
    const healthEl = document.getElementById(`${prefix}-health`);
    if (t.stress < 0.1) {
      healthEl.textContent = "🟢 Healthy";
      healthEl.style.background = "rgba(0,200,83,0.1)";
      healthEl.style.borderColor = "rgba(0,200,83,0.3)";
    } else if (t.stress < 0.5) {
      healthEl.textContent = "🟡 Mild Stress";
      healthEl.style.background = "rgba(255,167,38,0.1)";
      healthEl.style.borderColor = "rgba(255,167,38,0.3)";
    } else {
      healthEl.textContent = "🔴 High Stress";
      healthEl.style.background = "rgba(239,83,80,0.1)";
      healthEl.style.borderColor = "rgba(239,83,80,0.3)";
    }

    // Plants
    renderPlants(prefix, t.lai, t.stress);

    // Store for hero canvas
    laiValues[shortName]    = t.lai;
    stressValues[shortName] = t.stress;

    // Charts
    CHART_KEYS.forEach((key, ki) => {
      const hist = t[`${key}_history`];
      charts[key].data.datasets[i].data = hist;
    });
  });

  CHART_KEYS.forEach(key => charts[key].update("none"));
  drawHeroFarm();
}

// Poll backend every 400ms
setInterval(() => {
  fetch("/api/state")
    .then(r => r.json())
    .then(updateUI)
    .catch(() => {});
}, 400);