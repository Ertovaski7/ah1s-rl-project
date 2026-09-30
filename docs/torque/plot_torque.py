"""Tork sınırı şekli: kalkışta tork (eski / yeni model) + güç zarfındaki senaryolarda 56 psi üstü süre.
python docs/torque/plot_torque.py   →   docs/torque/fig_torque.png"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import evaluate_takeoff as ET  # noqa: E402

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRIT, WARN = "#d03b3b", "#fab219"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2.0})

RUNS = [("eski model, stok fizik", REPO / "models_takeoff/takeoff_final.zip", None, ORANGE),
        ("yeni model (tork gözlemi + ceza), güç tavanı", REPO / "models_takeoff/takeoff_torque.zip", None, BLUE)]


def trace(model, env_ov, sid="kalkis_500"):
    env, pol, _ = ET.make_env(model, env_ov)
    sc = [s for s in ET.SCENARIOS if s[0] == sid][0]
    _, _, _, level, start, tasks, fuel = sc
    obs, info = env.reset(seed=0, options=dict(level=level, tasks=[dict(t) for t in tasks], fuel=fuel, start="ground",
                                               start_heading_deg=0.0, start_perturb=0.0))
    t, q, done = [], [], False
    while not done:
        obs, r, te, tr, info = env.step(pol(obs))
        t.append(info["t"])
        q.append(info["torque_psi"])
        done = te or tr
    return np.array(t), np.array(q)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
for lab, model, ov, col in RUNS:
    t, q = trace(model, ov)
    a1.plot(t, q, color=col, lw=1.7, label=lab)
a1.axhline(56, color=CRIT, lw=1.2, ls="--")
a1.axhline(50, color=WARN, lw=1.2, ls="--")
a1.text(10, 57.3, "56 psi = %100 tork (30 dk)", color=CRIT, fontsize=8.5)
a1.text(10, 46.5, "50 psi sürekli", color="#9a6a00", fontsize=8.5)
a1.set_xlim(0, 30)
a1.set_ylim(0, 105)
a1.set_xlabel("zaman (s)")
a1.set_ylabel("ana rotor torku (psi)")
a1.set_title("Yerden 500 ft kalkış (8500 lbs): kalkış anındaki tork", loc="left")
a1.legend(loc="upper right")

labels, over, peak = [], [], []
for lab, f in (("eski model, stok fizik", "eval_takeoff_final_stok.json"),
               ("eski model, yeni fizik (kalibre GE + tavan)", "eval_takeoff_final_yenifizik.json"),
               ("yeni model, yeni fizik", "eval_takeoff_torque.json")):
    rs = [r for r in json.loads((HERE / f).read_text())["results"] if not r["id"].startswith("agir")]
    labels.append(lab)
    over.append(sum(r["t_over56_s"] for r in rs))
    peak.append(max(r["torque_max"] for r in rs))
y = np.arange(len(labels))[::-1]
bars = a2.barh(y, over, color=[ORANGE, AQUA, BLUE], height=0.55)
for yi, o, p in zip(y, over, peak):
    a2.text(o + 0.4, yi, f"{o:.0f} s  (en yüksek {p:.0f} psi)", va="center", fontsize=9, color=INK2)
a2.set_yticks(y)
a2.set_yticklabels(labels)
a2.set_xlim(0, max(over) * 1.9)
a2.set_xlabel("56 psi üstünde toplam süre (s), 25 senaryo ≤ 9700 lbs")
a2.set_title("Güç zarfındaki senaryolarda sınır aşımı", loc="left")
a2.grid(axis="y", visible=False)
fig.tight_layout()
out = HERE / "fig_torque.png"
fig.savefig(out, dpi=150)
print(out, dict(zip(labels, zip(over, peak))))
