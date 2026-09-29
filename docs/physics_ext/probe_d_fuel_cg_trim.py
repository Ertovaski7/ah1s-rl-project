"""Probe (d): yakıt azaldıkça hover triminin değişimi — ağırlık ve CG kayması (özellikle uzunlamasına cyclic).

OGE hover (300 ft AGL, konum tutulur), toplam yakıt %100 → %0 (1780 → 0 lbs), üç çekim sırasıyla:
equal (varsayılan varsayım: iki tanktan eşit), fwd_first (önce ön tank), aft_first (önce arka tank). Tank x: ön 146,
arka 206 in; boş CG 172 in. Her noktada tutucu oturunca trim (4 kumanda), θ / φ, CG, tork. Ayrıca probe (a)'daki sürekli
hover koşusunun (chart modeli, eşit çekim) kumanda zaman serisi aynı şekle çizilir.

Çıktı: probe_d_fuel_cg_trim.csv, fig_d_fuel_cg_trim.png, probe_d_fuel_cg_trim.txt
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np

import probe_common  # noqa: E402  (outp: çıktı adı eki)
from probe_common import GROUND_MSL_FT, OUT, Holder, Rig, env_config_from_args, settle_and_measure, sfd_trim  # (repo kökünü path'e ekler)
from physics_ext import draw_from_tanks  # noqa: E402

FRACS = (1.0, 0.75, 0.5, 0.25, 0.0)
KEYS = ("c0", "c1", "c2", "c3", "theta", "phi", "psi_gauge", "rpm", "weight", "cg_x", "cg_z")


def hover_trim(fuel, env_config=None) -> dict:
    rig = Rig(fuel=fuel, env_config=env_config)
    w = 8500.0 + sum(fuel)
    rig.teleport(300.0, 0.0, ctrl=sfd_trim(0.0, GROUND_MSL_FT + 300.0, w))
    s = rig.state()
    hold = Holder(300.0, 0.0, mode="ground", n0=s["n"], e0=s["e"], weight=w)
    m = settle_and_measure(rig, hold, t_min=45.0, t_max=300.0, avg_s=12.0, tol=dict(dctrl=0.003))
    return dict(fuel_fwd=fuel[0], fuel_aft=fuel[1], converged=m["converged"], **{k: m[k] for k in KEYS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aircraft", default=None)
    ap.add_argument("--power-cap", type=float, default=0.0, help="repo uçağında güç tavanı, psi (0 → yok)")
    ap.add_argument("--plot-only", action="store_true", help="CSV'den yalnızca şekil")
    args = ap.parse_args()
    env_config = env_config_from_args(args.aircraft, args.power_cap)
    if args.plot_only:
        with open(probe_common.outp("probe_d_fuel_cg_trim.csv")) as fh:
            rows = [{k: (v if k in ("draw", "converged") else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
        plot(rows)
        return
    rows, cache = [], {}
    for mode in ("equal", "fwd_first", "aft_first"):
        for fr in FRACS:
            burn = (1.0 - fr) * 1780.0
            fuel = tuple(float(x) for x in np.round(draw_from_tanks((890.0, 890.0), burn, mode), 3))
            if fuel not in cache:
                cache[fuel] = hover_trim(fuel, env_config)
            r = dict(draw=mode, fuel_frac=fr, **cache[fuel])
            rows.append(r)
            print(f"{mode:9s} %{fr * 100:3.0f} tank {fuel[0]:5.0f}/{fuel[1]:5.0f} W {r['weight']:6.0f} "
                  f"CG x {r['cg_x']:6.2f} z {r['cg_z']:5.2f} | coll {r['c0']:.4f} lon {r['c1']:+.4f} "
                  f"lat {r['c2']:+.4f} ped {r['c3']:+.4f} θ {math.degrees(r['theta']):+.2f} "
                  f"φ {math.degrees(r['phi']):+.2f} psi {r['psi_gauge']:.1f} conv={r['converged']}", flush=True)
    with open(probe_common.outp("probe_d_fuel_cg_trim.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    # CG duyarlılığı: tüm noktalardan doğrusal regresyon lon ~ a + b·CGx + c·W
    X = np.array([[1.0, r["cg_x"] - 172.0, (r["weight"] - 8500.0) / 1000.0] for r in cache.values()])
    lines = ["Probe (d) — yakıt / CG → OGE hover trimi (300 ft AGL)", ""]
    for key, lab in (("c1", "elevator (uzunlamasına cyclic)"), ("c0", "collective"), ("c3", "pedal"),
                     ("c2", "aileron"), ("theta", "θ (rad)")):
        y = np.array([r[key] for r in cache.values()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        lines.append(f"{lab:32s} = {coef[0]:+.4f} {coef[1]:+.5f}·(CGx − 172 in) {coef[2]:+.4f}·(W − 8500)/1000 lbs"
                     f"   (artık en çok {np.abs(res).max():.4f})")
    eq = [r for r in rows if r["draw"] == "equal"]
    lines += ["", "eşit çekim yolu (%100 → %0): "
              f"elevator {eq[0]['c1']:+.4f} → {eq[-1]['c1']:+.4f}, CG x {eq[0]['cg_x']:.2f} → {eq[-1]['cg_x']:.2f} in, "
              f"collective {eq[0]['c0']:.4f} → {eq[-1]['c0']:.4f}, pedal {eq[0]['c3']:.4f} → {eq[-1]['c3']:.4f}, "
              f"θ {math.degrees(eq[0]['theta']):+.2f} → {math.degrees(eq[-1]['theta']):+.2f}°"]
    for mode in ("fwd_first", "aft_first"):
        rr = [r for r in rows if r["draw"] == mode]
        lo, hi = min(r["c1"] for r in rr), max(r["c1"] for r in rr)
        lines.append(f"{mode} yolu: elevator {lo:+.4f} … {hi:+.4f} (aralık {hi - lo:.4f}), CG x "
                     f"{min(r['cg_x'] for r in rr):.2f} … {max(r['cg_x'] for r in rr):.2f} in")
    (probe_common.outp("probe_d_fuel_cg_trim.txt")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"equal": "#2a6fdb", "fwd_first": "#d9480f", "aft_first": "#2f9e44"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = (("c1", "elevator (uzunlamasına cyclic)", 1.0), ("theta", "yunuslama θ, °", 180 / math.pi),
              ("cg_x", "CG x, in", 1.0), ("c0", "collective", 1.0), ("c3", "pedal", 1.0), ("c2", "aileron", 1.0))
    ts = None
    try:
        with open(probe_common.outp("probe_a_timeseries.csv")) as fh:
            ts = [r for r in csv.DictReader(fh) if r["cond"] == "hover" and r["fuel_model"] == "chart"]
    except FileNotFoundError:
        pass
    for ax, (key, lab, k) in zip(axes.flat, panels):
        for mode, col in cols.items():
            rr = [r for r in rows if r["draw"] == mode]
            ax.plot([r["fuel_frac"] * 1780 for r in rr], [r[key] * k for r in rr], "-o", color=col, ms=4,
                    label={"equal": "eşit çekim (varsayım)", "fwd_first": "önce ön tank",
                           "aft_first": "önce arka tank"}[mode])
        if ts and key in ("c0", "c1", "c2", "c3", "cg_x"):
            ax.plot([float(r["fuel_lbs"]) for r in ts], [float(r[key]) * k for r in ts], "-", color="0.6", lw=1,
                    label="probe (a) sürekli hover (eşit çekim)")
        if ts and key == "theta":
            ax.plot([float(r["fuel_lbs"]) for r in ts], [float(r["theta_deg"]) for r in ts], "-", color="0.6", lw=1)
        ax.set_title(lab)
        ax.set_xlabel("toplam yakıt, lbs")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Probe (d): yakıt azaldıkça OGE hover trimi (300 ft AGL) — çekim sırasına göre")
    fig.tight_layout()
    fig.savefig(probe_common.outp("fig_d_fuel_cg_trim.png"), dpi=110)


if __name__ == "__main__":
    main()
