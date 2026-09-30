"""
PROBE GROUND EFFECT — JSBSim AH-1S yer etkisi ölçümü (stok / kalibre / kapalı)
=============================================================================

Her koşuda helikopter reset'e özel PID ile (kalkış env'inin `_settle_hover`'ıyla aynı kazançlar) sabit kızak
yüksekliğinde hover'da ya da sabit yer hızında tutulur; son `avg` saniyenin ortalaması alınır: collective, ana rotor
torku (modelin psi göstergesi), indüklenmiş hız (propulsion/engine/vi-fps), güç.

Uçaklar
  stock       JSBSim paketindeki ah1s (eski modellerin eğitildiği fizik)
  calibrated  repo kopyası aircraft/ah1s, ge/model = 1 (Hayden 1976 + Cheeseman–Bennett 1955)
  stocklike   repo kopyası, ge/model = 0 (stok formülün XML'de yeniden kurulmuşu — sağlama)
  off         repo kopyası, ge/enable = 0 (yer etkisi yok; A/B referansı)

Hedef (kalibre): indüklenmiş hız oranı κ(z, V) = 1 − (1 − κ_H(z)) / (1 + (V / v_h)²),
κ_H = 1 / (0.9926 + 0.03794 (2R/z)²), z = göbek yüksekliği = kızak + 13.1 ft, R = 22 ft.

Kullanım
  python docs/ground_effect/probe_ground_effect.py                   # hepsi → docs/ground_effect/ge_probe.json
  python docs/ground_effect/probe_ground_effect.py --only height --aircraft stock calibrated off
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from helicopter_env_command import CONTROL_DT, wrap_deg  # noqa: E402
from helicopter_env_takeoff import CTRL_HI, CTRL_LO, HelicopterEnvTakeoff, TakeoffEnvConfig, torque_psi  # noqa: E402
from takeoff_curriculum import GROUND_H_FT  # noqa: E402

HUB_ABOVE_SKID_FT = (153.0 + 4.5) / 12.0
R_FT = 22.0
KT = 1.68781
AIRCRAFT = {
    "stock": dict(aircraft="stock"),
    "calibrated": dict(aircraft="repo", ground_effect="calibrated"),
    "stocklike": dict(aircraft="repo", ground_effect="stock"),
    "off": dict(aircraft="repo", ground_effect="off"),
}
READ = {"coll": "fcs/collective-cmd-norm", "torque": "propulsion/engine/torque-lbsft", "vi": "propulsion/engine/vi-fps",
        "rpm": "propulsion/engine/rotor-rpm", "hp": "propulsion/engine/power-hp", "h": "position/h-agl-ft",
        "weight": "inertia/weight-lbs", "mu": "propulsion/engine/advance-ratio",
        "ge_scale": "propulsion/engine/groundeffect-scale-norm"}


def kappa_target(hs_ft, v_fps=0.0, v_h=35.5):
    z = np.asarray(hs_ft, dtype=float) + HUB_ABOVE_SKID_FT
    k_h = np.minimum(1.0, 1.0 / (0.9926 + 0.03794 * (2 * R_FT / z) ** 2))
    return 1.0 - (1.0 - k_h) / (1.0 + (np.asarray(v_fps, dtype=float) / v_h) ** 2)


def kappa_cb(hs_ft):
    z = np.asarray(hs_ft, dtype=float) + HUB_ABOVE_SKID_FT
    return 1.0 - (R_FT / (4.0 * z)) ** 2


def fly(env: HelicopterEnvTakeoff, hs: float, fuel=(0.0, 0.0), u_kt: float = 0.0, T: float = 40.0, avg: float = 8.0,
        psi0: float = 0.0) -> dict:
    """Kızak yüksekliği hs'de, yer hızı u_kt ile (psi0 yönünde) düz uçuş / hover; son avg s ortalaması."""
    env._create_fdm()
    env._set_fuel(fuel)
    env._warmup_rotor()
    f = env.fdm
    env._lat0, env._lon0 = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
    h0, u_ref, p0 = GROUND_H_FT + hs, u_kt * KT, math.radians(psi0)
    coll0 = env.trim[0] + 0.057 * sum(fuel) / 1000.0 - 0.004 * u_kt
    env._write_controls(np.array([coll0, *env.trim[1:]]))
    env._set_ic_from_state(h=h0, psi_deg=psi0, still=True)
    if u_ref:
        env._set_ic_from_state(dn=u_ref * math.cos(p0), de=u_ref * math.sin(p0))
    env._set_sas(psi0)
    integ = np.array([0.0, -0.0015 * u_kt, 0.0, 0.0])
    rec = {k: [] for k in (*READ, "ug", "psi")}
    t, ok = 0.0, True
    while t < T - 1e-9:
        s = env._state()
        ps = math.radians(s["psi_deg"])
        dn, de = -s["n"], -s["e"]
        if u_ref:
            eu = u_ref - s["ug"]
            cross = dn * math.sin(p0) - de * math.cos(p0)                # yola göre sağa sapma (ft)
            ev = float(np.clip(-0.12 * cross, -6, 6)) - s["vg"]
        else:
            xf, yr = dn * math.cos(ps) + de * math.sin(ps), -dn * math.sin(ps) + de * math.cos(ps)
            eu = float(np.clip(0.12 * xf, -6, 6)) - s["ug"]
            ev = float(np.clip(0.12 * yr, -6, 6)) - s["vg"]
        evs = float(np.clip(0.4 * (h0 - s["h"]), -6, 6)) - s["vs"]
        eps = wrap_deg(psi0 - s["psi_deg"])
        integ[0] = np.clip(integ[0] + 0.01 * evs * CONTROL_DT, -0.3, 0.3)
        integ[1] = np.clip(integ[1] - 0.004 * eu * CONTROL_DT, -0.3, 0.3)
        integ[2] = np.clip(integ[2] + 0.004 * ev * CONTROL_DT, -0.3, 0.3)
        integ[3] = np.clip(integ[3] - 0.003 * eps * CONTROL_DT, -0.3, 0.3)
        th_ref = float(np.clip(-0.02 * eu, -0.25, 0.25)) + integ[1]
        ph_ref = float(np.clip(0.03 * ev, -0.25, 0.25)) + integ[2]
        c = np.array([coll0 + 0.03 * evs + integ[0],
                      env.trim[1] + 3.0 * (s["theta"] - th_ref) + 1.6 * s["q"],
                      env.trim[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"],
                      env.trim[3] - 0.02 * eps + 0.8 * s["r"] + integ[3]])
        env._write_controls(np.clip(c, CTRL_LO, CTRL_HI))
        if not env._run_plain():
            ok = False
            break
        t += CONTROL_DT
        if t >= T - avg:
            for k, p in READ.items():
                rec[k].append(float(f[p]))
            rec["ug"].append(s["ug"])
            rec["psi"].append(float(torque_psi(float(f["propulsion/engine/torque-lbsft"]))))
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in rec.items()}
    out.update(h_std=float(np.std(rec["h"])) if rec["h"] else float("nan"), hs=hs, u_kt=u_kt, fuel=list(fuel), ok=ok,
               z_over_R=(hs + HUB_ABOVE_SKID_FT) / R_FT)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aircraft", nargs="*", default=list(AIRCRAFT))
    ap.add_argument("--only", nargs="*", default=["height", "speed"])
    ap.add_argument("--json", default=str(Path(__file__).with_name("ge_probe.json")))
    args = ap.parse_args(argv)
    t0 = time.time()
    heights = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 66, 88, 293.7]
    speeds = [0, 5, 10, 15, 20, 30, 40]
    res = {"heights": heights, "speeds": speeds, "runs": []}
    for name in args.aircraft:
        env = HelicopterEnvTakeoff(config=TakeoffEnvConfig(**AIRCRAFT[name]))
        if "height" in args.only:
            for fuel in ((0.0, 0.0), (890.0, 890.0)):
                for hs in heights:
                    r = dict(fly(env, hs, fuel=fuel), aircraft=name, sweep="height")
                    res["runs"].append(r)
                    print(f"{name:10s} W={r['weight']:.0f} hs={hs:6.1f} coll={r['coll']:.4f} Q={r['torque']:.0f} "
                          f"({r['psi']:.1f} psi) vi={r['vi']:.2f} ge_scale={r['ge_scale']:.4f} ok={r['ok']}", flush=True)
        if "speed" in args.only:
            for hs in (5.0, 293.7):
                for u in speeds:
                    r = dict(fly(env, hs, u_kt=u, T=60.0, avg=10.0), aircraft=name, sweep="speed")
                    res["runs"].append(r)
                    print(f"{name:10s} SPEED hs={hs:.0f} u={u:3d} kt Q={r['torque']:.0f} vi={r['vi']:.2f} "
                          f"ug={r['ug']:.1f} ok={r['ok']}", flush=True)
    Path(args.json).write_text(json.dumps(res, indent=1))
    print(f"kaydedildi: {args.json} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
