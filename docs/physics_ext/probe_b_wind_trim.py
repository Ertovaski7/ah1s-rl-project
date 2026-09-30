"""Probe (b): rüzgârın rotor modeline yansıması — rüzgârda yerinde hover trimi ↔ rüzgârsız aynı hava hızında uçuş trimi.

Galile değişmezliği: 300 ft AGL, heading 0 (kuzey), 8500 lbs. Rüzgâr W ∈ {0, 10, 20, 30} kt, geldiği yön:
karşıdan (0°), sağdan (90°), soldan (270°), arkadan (180°). Karşılığı rüzgârsız: ileri / sağa / sola / geri uçuş,
hava hızı W (gövde ekseninde). Aynı hava hızı vektörü → aynı aerodinamik durum → trimler aynı olmalı. Fark varsa
JSBSim'de bir şey yer hızını ya da atalet hızını kullanıyor demektir (ör. SFD tablosu; rotor downwash hesabı
`ah1s.xml`'de "deneysel" diye işaretli). Ayrıca: SFD tablosunun (yer hızına göre) hover-in-wind için verdiği trim.

Çıktı: probe_b_wind_trim.csv, fig_b_wind_trim.png, probe_b_wind_trim.txt
Kullanım: python docs/physics_ext/probe_b_wind_trim.py [--aircraft repo]
"""
from __future__ import annotations

import argparse
import math

import numpy as np

import probe_common  # noqa: E402  (outp: çıktı adı eki)
from probe_common import KT_TO_FPS, OUT, Holder, Rig, env_config_from_args, sfd_trim, settle_and_measure

DIRS = {"karşı": 0.0, "sağ": 90.0, "arka": 180.0, "sol": 270.0}
FIELDS = ("c0", "c1", "c2", "c3", "theta", "phi", "psi_gauge", "p_rotor_hp", "u_air", "v_air", "u_gnd", "v_gnd", "rpm",
          "rot_coll", "rot_lon", "rot_lat", "rot_tail", "h")


def trim_point(wind_kt: float, from_deg: float, mode: str, env_config=None) -> dict:
    rig = Rig(fuel=(0.0, 0.0), env_config=env_config)
    h0, psi0 = 300.0, 0.0
    if mode == "hover_wind":
        rig.set_wind(wind_kt, from_deg)
        a = math.radians(from_deg)                  # hava kütlesi gövdeye "from" yönünden gelir
        u_air, v_air = wind_kt * KT_TO_FPS * math.cos(a), wind_kt * KT_TO_FPS * math.sin(a)
    else:
        a = math.radians(from_deg)
        u_air, v_air = wind_kt * KT_TO_FPS * math.cos(a), wind_kt * KT_TO_FPS * math.sin(a)
    ff = sfd_trim(u_air / KT_TO_FPS, 2283.5 + h0)
    rig.teleport(h0, psi0, u_air=u_air, v_air=v_air, ctrl=ff)
    s = rig.state()
    if mode == "hover_wind":
        hold = Holder(h0, psi0, mode="ground", n0=s["n"], e0=s["e"])
    else:
        hold = Holder(h0, psi0, mode="air", u=u_air, v=v_air)
    m = settle_and_measure(rig, hold, t_min=60.0, t_max=300.0, avg_s=15.0, tol=dict(dctrl=0.003))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aircraft", default=None, help="repo → ground-effect-torque uçak kopyası (varsa)")
    ap.add_argument("--power-cap", type=float, default=0.0, help="repo uçağında güç tavanı, psi (0 → yok)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    env_config = env_config_from_args(args.aircraft, args.power_cap)
    if args.plot_only:
        import csv
        with open(probe_common.outp("probe_b_wind_trim.csv")) as fh:
            rows = [{k: (v if k in ("dir", "mode", "converged") else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(fh)]
        plot(rows)
        return
    rows = []
    for name, d in DIRS.items():
        for w in (0.0, 10.0, 20.0, 30.0):
            if w == 0.0 and name != "karşı":
                continue
            for mode in ("hover_wind", "flight_nowind"):
                m = trim_point(w, d, mode, env_config)
                sfd_gnd = sfd_trim(0.0, 2583.5)                  # JSBSim'in SFD'si hover-in-wind'de: yer hızı 0
                rows.append(dict(dir=name, from_deg=d, wind_kt=w, mode=mode, converged=m["converged"],
                                 settle_s=m["settle_s"], ctrl_std=m["ctrl_std"],
                                 **{k: m[k] for k in FIELDS},
                                 sfd_gnd_c0=sfd_gnd[0], sfd_gnd_c1=sfd_gnd[1], sfd_gnd_c2=sfd_gnd[2],
                                 sfd_gnd_c3=sfd_gnd[3]))
                r = rows[-1]
                print(f"{name:5s} {w:4.0f} kt {mode:13s} conv={r['converged']!s:5s} "
                      f"coll {r['c0']:.4f} lon {r['c1']:+.4f} lat {r['c2']:+.4f} ped {r['c3']:+.4f} "
                      f"θ {math.degrees(r['theta']):+5.2f} φ {math.degrees(r['phi']):+5.2f} psi {r['psi_gauge']:5.1f} "
                      f"u_air {r['u_air']:+6.1f} v_air {r['v_air']:+6.1f} yer ({r['u_gnd']:+.2f},{r['v_gnd']:+.2f})",
                      flush=True)
    import csv
    path = probe_common.outp("probe_b_wind_trim.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    # özet: aynı hava hızında fark
    lines = ["Probe (b) — rüzgârda hover ↔ rüzgârsız aynı hava hızı (8500 lbs, 300 ft AGL, heading 0)",
             "Δ = rüzgârda hover − rüzgârsız uçuş. 'otur' = iki tutucu da oturdu mu (hover / uçuş).", "",
             "yön   W kt otur | Δcoll    Δlon     Δlat     Δped   | Δrotor coll/lon/lat/kuyruk (mrad) | Δθ°    Δφ°  "
             " | Δpsi  Δrpm | SFD(yer hızı) − gerçek trim: coll lon lat ped"]
    maxd = np.zeros(4)
    for name, d in DIRS.items():
        for w in (10.0, 20.0, 30.0):
            a = next(r for r in rows if r["dir"] == name and r["wind_kt"] == w and r["mode"] == "hover_wind")
            b = next(r for r in rows if r["dir"] == name and r["wind_kt"] == w and r["mode"] == "flight_nowind")
            dc = np.array([a[f"c{i}"] - b[f"c{i}"] for i in range(4)])
            dr = 1000.0 * np.array([a[k] - b[k] for k in ("rot_coll", "rot_lon", "rot_lat", "rot_tail")])
            both = a["converged"] and b["converged"]
            if both:
                maxd = np.maximum(maxd, np.abs(dc))
            se = np.array([a[f"sfd_gnd_c{i}"] - a[f"c{i}"] for i in range(4)])
            conv = f"{'E' if a['converged'] else 'H'}/{'E' if b['converged'] else 'H'}"
            lines.append(f"{name:5s} {w:4.0f} {conv:4s} | {dc[0]:+.4f} {dc[1]:+.4f} {dc[2]:+.4f} {dc[3]:+.4f} | "
                         f"{dr[0]:+5.2f} {dr[1]:+5.2f} {dr[2]:+5.2f} {dr[3]:+5.2f} | "
                         f"{math.degrees(a['theta'] - b['theta']):+.2f} {math.degrees(a['phi'] - b['phi']):+.2f} | "
                         f"{a['psi_gauge'] - b['psi_gauge']:+.2f} {a['rpm'] - b['rpm']:+.2f} | "
                         f"{se[0]:+.3f} {se[1]:+.3f} {se[2]:+.3f} {se[3]:+.3f}")
    lines += ["", f"iki tutucunun da oturduğu çiftlerde en büyük |Δ| (coll, lon, lat, ped): {np.round(maxd, 4).tolist()}"]
    (probe_common.outp("probe_b_wind_trim.txt")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["collective", "uzunlamasına cyclic (elevator)", "yanal cyclic (aileron)", "pedal (rudder)"]
    colors = {"karşı": "#2a6fdb", "sağ": "#d9480f", "sol": "#2f9e44", "arka": "#7048e8"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    def series(name, mode, key, conv=float):
        ws = sorted({r["wind_kt"] for r in rows})
        ok, bad = [], []
        for w in ws:
            r = next((x for x in rows if x["mode"] == mode and x["wind_kt"] == w and (x["dir"] == name or w == 0.0)),
                     None)
            if r is None:
                continue
            (ok if r["converged"] in (True, "True") else bad).append((w, conv(r[key])))
        return ok, bad

    for i in range(4):
        ax = axes.flat[i]
        for name, col in colors.items():
            fl, flb = series(name, "flight_nowind", f"c{i}")
            hv, hvb = series(name, "hover_wind", f"c{i}")
            if fl:
                ax.plot(*zip(*fl), "-", color=col, lw=1.6, label=f"rüzgârsız uçuş ({name})")
            if hv:
                ax.plot(*zip(*hv), "o", color=col, ms=6, mfc="none", label=f"rüzgârda hover ({name} rüzgâr)")
            for b in flb + hvb:
                ax.plot(*b, "x", color=col, ms=7)
        sfd0 = rows[0][f"sfd_gnd_c{i}"]
        ax.axhline(sfd0, color="0.5", ls=":", lw=1.2, label="SFD tablosu hover-in-wind'de (yer hızı 0)")
        ax.set_title(labels[i])
        ax.set_xlabel("hava hızı / rüzgâr, kt")
        ax.grid(alpha=0.3)
    for j, (key, lab, conv) in enumerate((("theta", "yunuslama θ, °", math.degrees),
                                          ("psi_gauge", "tork, psi", float))):
        ax = axes.flat[4 + j]
        for name, col in colors.items():
            fl, flb = series(name, "flight_nowind", key, conv)
            hv, hvb = series(name, "hover_wind", key, conv)
            if fl:
                ax.plot(*zip(*fl), "-", color=col, lw=1.6)
            if hv:
                ax.plot(*zip(*hv), "o", color=col, ms=6, mfc="none")
            for b in flb + hvb:
                if abs(b[1]) < 12.0 or key != "theta":
                    ax.plot(*b, "x", color=col, ms=7)
        ax.set_title(lab)
        ax.set_xlabel("hava hızı / rüzgâr, kt")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle("Probe (b): rüzgârda yerinde hover (daire) = rüzgârsız aynı hava hızında uçuş (çizgi)? "
                 "8500 lbs, 300 ft AGL, heading 0  (x: tutucu oturmadı)")
    fig.tight_layout()
    fig.savefig(probe_common.outp("fig_b_wind_trim.png"), dpi=110)


if __name__ == "__main__":
    main()
