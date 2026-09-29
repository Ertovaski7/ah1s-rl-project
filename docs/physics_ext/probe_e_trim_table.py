"""Probe (e): hava hızına dayalı trim tablosu — gövde ekseninde İLERİ HAVA hızı (yer hızı değil) × ağırlık.

JSBSim'in `steady_flight_data.xml` tabloları 8500 lbs için ve YER hızıyla (velocities/vg-fps × sign(u)) bakılıyor →
rüzgârda ve yana uçuşta yanlış (probe b). Burada trim, düz ve dengeli uçuşta (irtifa, heading, yana hava hızı 0 tutulur)
ölçülür: u_air ∈ [−20, 120] kt, ağırlık 8500 … 10,280 lbs (iki tank eşit), 300 ft AGL; ayrıca 1000 ft AGL (irtifa
etkisi) ve 8500 lbs'de düşük hız ızgarası (u, v ∈ ±20 kt; yana hava hızının etkisi, bilgi için).

Çıktı: probe_e_trim_table.csv (tüm noktalar), trim_table_airspeed.json (tablo: u_kt × ağırlık → collective, elevator,
aileron, rudder, θ, φ, psi), fig_e_trim_table.png, probe_e_trim_table.txt
Kullanım: python docs/physics_ext/probe_e_trim_table.py [--aircraft repo]
"""
from __future__ import annotations

import argparse
import csv
import json
import math

import numpy as np

from probe_common import GROUND_MSL_FT, KT_TO_FPS, OUT, Holder, Rig, settle_and_measure, sfd_trim

U_KT = (-20.0, -10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
        110.0, 120.0)
FUEL_PER_TANK = (0.0, 297.0, 593.0, 890.0)          # → 8500, 9094, 9686, 10280 lbs
KEYS = ("c0", "c1", "c2", "c3", "theta", "phi", "psi_gauge", "rpm", "p_rotor_hp", "weight", "cg_x", "u_air", "v_air",
        "rot_coll", "rot_lon", "rot_lat", "rot_tail")


def trim_at(u_kt: float, v_kt: float, fuel_tank: float, h_agl: float, env_config=None, prev=None) -> dict:
    rig = Rig(fuel=(fuel_tank, fuel_tank), env_config=env_config)
    w = 8500.0 + 2 * fuel_tank
    u, v = u_kt * KT_TO_FPS, v_kt * KT_TO_FPS
    ff = sfd_trim(u_kt, GROUND_MSL_FT + h_agl, w) if prev is None else prev
    rig.teleport(h_agl, 0.0, u_air=u, v_air=v, ctrl=ff)
    hold = Holder(h_agl, 0.0, mode="air", u=u, v=v, weight=w)
    m = settle_and_measure(rig, hold, t_min=45.0, t_max=300.0, avg_s=12.0, tol=dict(dctrl=0.003))
    return dict(h_agl=h_agl, fuel_tank=fuel_tank, u_kt=u_kt, v_kt=v_kt, converged=m["converged"],
                settle_s=m["settle_s"], ctrl_std=m["ctrl_std"], **{k: m[k] for k in KEYS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aircraft", default=None)
    args = ap.parse_args()
    env_config = None
    if args.aircraft:
        from helicopter_env_takeoff import TakeoffEnvConfig
        env_config = TakeoffEnvConfig(aircraft=args.aircraft)
    rows = []

    def run(u, v, ft, h):
        r = trim_at(u, v, ft, h, env_config)
        rows.append(r)
        print(f"h {h:5.0f} W {r['weight']:6.0f} u {u:+5.0f} v {v:+4.0f} kt conv={r['converged']!s:5s} "
              f"coll {r['c0']:.4f} lon {r['c1']:+.4f} lat {r['c2']:+.4f} ped {r['c3']:+.4f} "
              f"θ {math.degrees(r['theta']):+5.2f} φ {math.degrees(r['phi']):+5.2f} psi {r['psi_gauge']:5.1f} "
              f"rpm {r['rpm']:.1f}", flush=True)

    for ft in FUEL_PER_TANK:
        for u in U_KT:
            run(u, 0.0, ft, 300.0)
    for ft in (0.0, 890.0):
        for u in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0):
            run(u, 0.0, ft, 1000.0)
    for u in (-10.0, 0.0, 10.0, 20.0):
        for v in (-20.0, -10.0, 10.0, 20.0):
            run(u, v, 0.0, 300.0)

    with open(OUT / "probe_e_trim_table.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # --- tablo (300 ft, v = 0) ---------------------------------------------------------------------------------------
    main_rows = [r for r in rows if r["h_agl"] == 300.0 and r["v_kt"] == 0.0]
    weights = sorted({round(r["weight"]) for r in main_rows})
    tab = {"source": "docs/physics_ext/probe_e_trim_table.py (JSBSim 1.3.1 ah1s, AFCS yalnızca SAS, "
                     f"{'aircraft=' + args.aircraft if args.aircraft else 'stok uçak'}, 300 ft AGL, düz uçuş, "
                     "yana hava hızı 0, heading sabit; iki tank eşit)",
           "u_kt": list(U_KT), "weight_lbs": weights, "h_agl_ft": 300.0}
    fields = {"collective": "c0", "elevator": "c1", "aileron": "c2", "rudder": "c3", "theta_deg": "theta",
              "phi_deg": "phi", "torque_psi": "psi_gauge", "rotor_rpm": "rpm"}
    conv = []
    for name, k in fields.items():
        grid = []
        for w in weights:
            line = []
            for u in U_KT:
                r = next(x for x in main_rows if round(x["weight"]) == w and x["u_kt"] == u)
                val = r[k]
                line.append(round(math.degrees(val), 4) if name.endswith("_deg") else round(val, 5))
                if name == "collective":
                    conv.append(bool(r["converged"]))
            grid.append(line)
        tab[name] = grid
    tab["converged_all"] = all(conv)
    (OUT / "trim_table_airspeed.json").write_text(json.dumps(tab, indent=1))
    report(rows, weights)
    plot(rows, weights)


def report(rows, weights):
    lines = ["Probe (e) — hava hızına dayalı trim tablosu (300 ft AGL, düz uçuş, v_air = 0)", ""]
    nc = [r for r in rows if not r["converged"]]
    lines.append(f"nokta: {len(rows)}, oturmayan: {len(nc)} "
                 + ", ".join(f"(h {r['h_agl']:.0f}, W {r['weight']:.0f}, u {r['u_kt']:+.0f}, v {r['v_kt']:+.0f})"
                             for r in nc))
    hdr = "u kt  | " + " | ".join(f"W {w:5d}: coll   lon     lat    ped   psi " for w in weights)
    lines += ["", hdr]
    for u in U_KT:
        parts = []
        for w in weights:
            r = next(x for x in rows if x["h_agl"] == 300.0 and x["v_kt"] == 0.0 and round(x["weight"]) == w
                     and x["u_kt"] == u)
            parts.append(f"{r['c0']:.3f} {r['c1']:+.3f} {r['c2']:+.3f} {r['c3']:+.3f} {r['psi_gauge']:4.1f}")
        lines.append(f"{u:+5.0f} | " + " | ".join(f"         {p}" for p in parts))
    # SFD karşılaştırması (8500 lbs)
    lines += ["", "SFD tablosu (JSBSim, 8500 lbs, aynı hıza yer hızı gibi bakılırsa) − ölçülen trim, 8500 lbs:"]
    for u in U_KT:
        r = next(x for x in rows if x["h_agl"] == 300.0 and x["v_kt"] == 0.0 and round(x["weight"]) == 8500
                 and x["u_kt"] == u)
        sf = sfd_trim(u, GROUND_MSL_FT + 300.0, 8500.0)
        d = sf - np.array([r[f"c{i}"] for i in range(4)])
        lines.append(f"  u {u:+5.0f} kt: coll {d[0]:+.3f} lon {d[1]:+.3f} lat {d[2]:+.3f} ped {d[3]:+.3f}")
    # irtifa etkisi
    lines += ["", "irtifa etkisi (1000 ft − 300 ft AGL):"]
    for w in (8500, 10280):
        for u in (0.0, 40.0, 80.0, 120.0):
            a = next((x for x in rows if x["h_agl"] == 1000.0 and round(x["weight"]) == w and x["u_kt"] == u), None)
            b = next((x for x in rows if x["h_agl"] == 300.0 and x["v_kt"] == 0.0 and round(x["weight"]) == w
                      and x["u_kt"] == u), None)
            if a and b:
                d = [a[f"c{i}"] - b[f"c{i}"] for i in range(4)]
                lines.append(f"  W {w} u {u:+4.0f}: coll {d[0]:+.4f} lon {d[1]:+.4f} lat {d[2]:+.4f} ped {d[3]:+.4f} "
                             f"psi {a['psi_gauge'] - b['psi_gauge']:+.2f}")
    # yana hava hızı
    lines += ["", "yana hava hızı etkisi (8500 lbs, 300 ft): trim(u, v) − trim(u, 0)"]
    for r in rows:
        if r["v_kt"] != 0.0:
            b = next(x for x in rows if x["h_agl"] == 300.0 and x["v_kt"] == 0.0 and round(x["weight"]) == 8500
                     and x["u_kt"] == r["u_kt"])
            d = [r[f"c{i}"] - b[f"c{i}"] for i in range(4)]
            lines.append(f"  u {r['u_kt']:+4.0f} v {r['v_kt']:+4.0f}: coll {d[0]:+.3f} lon {d[1]:+.3f} lat {d[2]:+.3f} "
                         f"ped {d[3]:+.3f} φ {math.degrees(r['phi'] - b['phi']):+.1f}° conv={r['converged']}")
    (OUT / "probe_e_trim_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(rows, weights):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["collective", "elevator (uzunlamasına cyclic)", "aileron (yanal cyclic)", "rudder (pedal)"]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    us = np.array(U_KT)
    for i in range(4):
        ax = axes.flat[i]
        for j, w in enumerate(weights):
            rr = [next(x for x in rows if x["h_agl"] == 300.0 and x["v_kt"] == 0.0 and round(x["weight"]) == w
                       and x["u_kt"] == u) for u in U_KT]
            ax.plot(us, [r[f"c{i}"] for r in rr], "-o", ms=3, color=cmap(j / max(1, len(weights) - 1)),
                    label=f"{w} lbs")
            bad = [(u, r[f"c{i}"]) for u, r in zip(us, rr) if not r["converged"]]
            if bad:
                ax.plot(*zip(*bad), "x", color="red", ms=8)
        ax.plot(us, [sfd_trim(u, GROUND_MSL_FT + 300.0, 8500.0)[i] for u in U_KT], "--", color="0.4",
                label="JSBSim SFD (8500 lbs)")
        ax.set_title(names[i])
        ax.set_xlabel("ileri hava hızı u_air, kt")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    for k, (key, lab, conv) in enumerate((("theta", "yunuslama θ, °", math.degrees), ("psi_gauge", "tork, psi", float))):
        ax = axes.flat[4 + k]
        for j, w in enumerate(weights):
            rr = [next(x for x in rows if x["h_agl"] == 300.0 and x["v_kt"] == 0.0 and round(x["weight"]) == w
                       and x["u_kt"] == u) for u in U_KT]
            ax.plot(us, [conv(r[key]) for r in rr], "-o", ms=3, color=cmap(j / max(1, len(weights) - 1)))
        if key == "psi_gauge":
            ax.axhline(50, color="orange", lw=1)
            ax.axhline(56, color="red", lw=1)
        ax.set_title(lab)
        ax.set_xlabel("ileri hava hızı u_air, kt")
        ax.grid(alpha=0.3)
    fig.suptitle("Probe (e): hava hızı × ağırlık trim tablosu (300 ft AGL, düz uçuş, yana hava hızı 0)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_e_trim_table.png", dpi=110)


if __name__ == "__main__":
    main()
