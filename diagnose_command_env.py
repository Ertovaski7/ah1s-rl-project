from __future__ import annotations

"""
DIAGNOSE COMMAND ENV — HelicopterEnvCommand gerçek JSBSim'de sağlıklı mı?
=========================================================================

Eğitim YOK. Eğitime başlamadan önce bir kez koştur (Colab'da ~1 dk):
    %run diagnose_command_env.py

Ölçülenler
  A) Reset: 300 ft / 15 ft/s'ye teleport + stabilizasyon başarılı mı, kaç s sürdü,
     rotor devri korundu mu, trim değerleri. Ayrıca yerden tırmanış (yedek yol).
  B) Sıfır action (a = 0) ile 60 s: helikopter kendi kendine ne kadar kayıyor?
  C) Açık-döngü basamak: her kanala ±0.5 action 4 s → tepki (dikey hız, hız,
     yatış, yaw rate, heading değişimi). Action ölçeklerini doğrulamak için.
  D) PPO'nun ilk (rastgele) politikası gibi gürültülü action (σ = 0.37) ile 90 s
     → ilk episode'lar hemen düşüyor mu?
  E) Hız: saniyede kaç kontrol adımı (eğitim süresini tahmin etmek için).

Çıktı: tablo + results_command_diagnostic/diagnostic.json
"""

import json
import time
from pathlib import Path

import numpy as np

from helicopter_env_command import CONTROL_DT, CommandEnvConfig, HelicopterEnvCommand

OUT = Path("results_command_diagnostic")
CH = ("collective", "elevator", "aileron", "rudder")


def fly(env, actions_fn, seconds, record=True):
    rows = []
    info = {}
    for k in range(int(round(seconds / CONTROL_DT))):
        a = actions_fn(k * CONTROL_DT)
        _, r, term, trunc, info = env.step(a)
        if record:
            rows.append(dict(t=info["t"], h=info["altitude"], vs=info["vertical_speed"], u=info["speed"],
                             v=info["lateral_speed"], roll=info["roll_deg"], pitch=info["pitch_deg"],
                             r=info["yaw_rate_dps"], hdg=info["heading_unwrapped"], rew=r))
        if term:
            break
    return rows, info


def settled_env(seed=0, cfg=None, **opts):
    env = HelicopterEnvCommand(level="H1", config=cfg or CommandEnvConfig())
    # komut yok, uzun episode: yalnızca açık-döngü ölçüm
    env.reset(seed=seed, options=dict(commands=[], episode_s=600.0, **opts))
    return env


def main():
    OUT.mkdir(exist_ok=True)
    report = {}
    print("=" * 100)
    print("DIAGNOSE COMMAND ENV — gerçek JSBSim, eğitim yok")
    print("=" * 100)

    # ---------------- A) reset ----------------
    print("\n[A] Reset: teleport → 300 ft / 15 ft/s, rastgele heading")
    rows = []
    env = HelicopterEnvCommand(level="H1")
    for seed in range(5):
        t0 = time.time()
        _, info = env.reset(seed=seed)
        su = info["setup"]
        rows.append(dict(seed=seed, mode=su["mode"], fresh=su["fresh_fdm"], setup_sim_s=su["t"],
                         wall_s=time.time() - t0, rpm=su["rpm"], psi0=su["psi0"],
                         trim=[round(float(x), 4) for x in env.trim], errors=su["errors"]))
        print(f"   seed={seed} mode={su['mode']} fresh={su['fresh_fdm']} setup={su['t']:5.1f}s sim "
              f"({time.time() - t0:4.2f}s wall) rpm={su['rpm']:.1f} ψ0={su['psi0']:6.1f}° "
              f"trim={rows[-1]['trim']} {'ERR: ' + str(su['errors']) if su['errors'] else ''}")
    t0 = time.time()
    envc = HelicopterEnvCommand(level="H1", config=CommandEnvConfig(start_mode="climb"))
    _, info = envc.reset(seed=0)
    print(f"   climb (yedek yol): setup={info['setup']['t']:.1f}s sim ({time.time() - t0:.2f}s wall) "
          f"trim={np.round(envc.trim, 4).tolist()}")
    report["A_reset"] = dict(teleport=rows, climb=dict(setup_sim_s=info["setup"]["t"], trim=envc.trim.tolist()))

    # ---------------- B) zero action ----------------
    print("\n[B] Sıfır action, 60 s (a = 0)")
    env = settled_env(seed=1)
    rec, info = fly(env, lambda t: np.zeros(4), 60.0)
    r0, r1 = rec[0], rec[-1]
    b = dict(dh=r1["h"] - r0["h"], vs_end=r1["vs"], du=r1["u"] - r0["u"], dpsi=r1["hdg"] - r0["hdg"],
             roll_end=r1["roll"], max_abs_r=max(abs(x["r"]) for x in rec), failure=info.get("failure"))
    print(f"   Δh={b['dh']:+.1f} ft  vs_son={b['vs_end']:+.2f} ft/s  Δu={b['du']:+.2f} ft/s  "
          f"Δψ={b['dpsi']:+.2f}°  roll_son={b['roll_end']:+.1f}°  max|r|={b['max_abs_r']:.2f}°/s  "
          f"failure={b['failure']}")
    report["B_zero_action"] = b

    # ---------------- C) open-loop steps ----------------
    print("\n[C] Açık-döngü basamak: tek kanal ±0.5 action, 4 s; tepki 8 s içinde")
    print(f"   {'kanal':10s} {'a':>5s} {'Δvs':>7s} {'Δu':>7s} {'Δroll':>7s} {'Δpitch':>7s} "
          f"{'r_max':>7s} {'Δψ@8s':>7s}")
    steps = []
    for c in range(4):
        for amp in (+0.5, -0.5):
            env = settled_env(seed=2)
            base, _ = fly(env, lambda t: np.zeros(4), 1.0)
            b0 = base[-1]

            def act(t, c=c, amp=amp):
                a = np.zeros(4)
                if t < 4.0:
                    a[c] = amp
                return a
            rec, info = fly(env, act, 8.0)
            pk = lambda key: max((x[key] - b0[key] for x in rec), key=abs)
            row = dict(channel=CH[c], amp=amp, dvs=pk("vs"), du=pk("u"), droll=pk("roll"), dpitch=pk("pitch"),
                       r_max=pk("r"), dpsi_8s=rec[-1]["hdg"] - b0["hdg"], failure=info.get("failure"))
            steps.append(row)
            print(f"   {CH[c]:10s} {amp:+5.1f} {row['dvs']:+7.2f} {row['du']:+7.2f} {row['droll']:+7.2f} "
                  f"{row['dpitch']:+7.2f} {row['r_max']:+7.2f} {row['dpsi_8s']:+7.2f}"
                  f"{'  FAIL ' + str(row['failure']) if row['failure'] else ''}")
    report["C_steps"] = steps

    # ---------------- D) noisy initial-policy-like actions ----------------
    print("\n[D] Gürültülü action (σ=0.37, PPO başlangıcı gibi), 90 s × 5")
    rng = np.random.default_rng(0)
    d_rows = []
    for seed in range(5):
        env = settled_env(seed=10 + seed)
        rec, info = fly(env, lambda t: rng.normal(0.0, 0.37, 4), 90.0)
        d_rows.append(dict(seed=seed, survived_s=rec[-1]["t"], failure=info.get("failure"),
                           max_roll=max(abs(x["roll"]) for x in rec), max_dh=max(abs(x["h"] - rec[0]["h"]) for x in rec),
                           max_dpsi=max(abs(x["hdg"] - rec[0]["hdg"]) for x in rec)))
        d = d_rows[-1]
        print(f"   seed={seed} süre={d['survived_s']:5.1f}s failure={d['failure']} max|roll|={d['max_roll']:.1f}° "
              f"max|Δh|={d['max_dh']:.1f} ft max|Δψ|={d['max_dpsi']:.1f}°")
    report["D_noise"] = d_rows

    # ---------------- E) speed ----------------
    env = settled_env(seed=3)
    n, t0 = 0, time.time()
    while time.time() - t0 < 3.0:
        env.step(rng.normal(0.0, 0.2, 4))
        n += 1
    sps = n / (time.time() - t0)
    print(f"\n[E] Hız: {sps:.0f} kontrol adımı/s (1 çekirdek)  → 1M adım ≈ {1e6 / sps / 60:.1f} dk/çekirdek")
    report["E_steps_per_s"] = sps

    with open(OUT / "diagnostic.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nkaydedildi: {OUT / 'diagnostic.json'}")
    return report


if __name__ == "__main__":
    main()
