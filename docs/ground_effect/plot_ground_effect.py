"""ge_probe.json → fig_ground_effect.png (stok vs kalibre yer etkisi, teori eğrileriyle)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_ground_effect import HUB_ABOVE_SKID_FT, KT, R_FT, kappa_cb, kappa_target  # noqa: E402

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2.0})

d = json.loads((HERE / "ge_probe.json").read_text())
runs = d["runs"]


def pick(ac, sweep, fuel=(0.0, 0.0), hs=None):
    out = {}
    for r in runs:
        if r["aircraft"] == ac and r["sweep"] == sweep and tuple(r["fuel"]) == tuple(fuel) and (hs is None or r["hs"] == hs):
            out[r["u_kt"] if sweep == "speed" else r["hs"]] = r
    return out


off = pick("off", "height")
oge = off[293.7]
f_i = oge["weight"] * oge["vi"] / (oge["torque"] * oge["rpm"] / 60 * 2 * np.pi)     # indüklenmiş güç payı
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
hh = np.linspace(0.5, 90, 300)
a1.plot(hh, f_i * (1 - kappa_target(hh)) * 100, color=INK2, ls="--", lw=1.4, label="hedef: Hayden (1976)")
a1.plot(hh, f_i * (1 - kappa_cb(hh)) * 100, color=MUTED, ls=":", lw=1.4, label="Cheeseman–Bennett (1955)")
for ac, col, lab in (("stock", ORANGE, "JSBSim stok (0.19 / 10 ft)"), ("calibrated", BLUE, "JSBSim kalibre (bu çalışma)")):
    rr = pick(ac, "height")
    hs = np.array(sorted(h for h in rr if h <= 90))
    red = [(off[h]["torque"] - rr[h]["torque"]) / off[h]["torque"] * 100 for h in hs]
    a1.plot(hs, red, color=col, marker="o", ms=4.5, label=lab)
a1.set_xlim(0, 90)
a1.set_ylim(0, 22)
a1.set_xlabel("kızak yüksekliği (ft)   [göbek = kızak + 13.1 ft, R = 22 ft]")
a1.set_ylabel("hover torkunda azalma, yer etkisi kapalıya göre (%)")
a1.set_title("Yükseklik: kalibre model hedefle örtüşüyor (8500 lbs)", loc="left")
a1.legend(loc="upper right")
a1.text(0.98, 0.45, f"teori eğrileri modelin indüklenmiş güç\npayıyla torka çevrildi (f_i = {f_i:.2f})",
        transform=a1.transAxes, ha="right", fontsize=8, color=MUTED)

offs = pick("off", "speed", hs=5.0)
v_h = offs[0]["vi"]
kk = np.linspace(0, 40, 200)
a2.plot(kk, 100 / (1 + (kk * KT / v_h) ** 2), color=INK2, ls="--", lw=1.4, label="hedef: Cheeseman–Bennett 1/(1+(V/vᵢ)²)")
for ac, col, lab in (("stock", ORANGE, "JSBSim stok: exp(−0.056·V)"), ("calibrated", BLUE, "JSBSim kalibre")):
    rr = pick(ac, "speed", hs=5.0)
    kts = np.array(sorted(rr))
    frac = np.array([1 - rr[k]["vi"] / offs[k]["vi"] for k in kts])
    a2.plot(kts, frac / frac[0] * 100, color=col, marker="o", ms=4.5, label=lab)
a2.set_xlim(0, 40)
a2.set_ylim(0, 105)
a2.set_xlabel("yatay hava hızı (kt) — rüzgârda hover da bu eksende")
a2.set_ylabel("yer etkisinin gücü, hover'a göre (%)")
a2.set_title("Hava hızıyla sönüm: kalibre model teoriyi izliyor (kızak 5 ft)", loc="left")
a2.legend(loc="upper right")
fig.tight_layout()
out = HERE / "fig_ground_effect.png"
fig.savefig(out, dpi=150)
print(out, f"f_i={f_i:.3f}")
