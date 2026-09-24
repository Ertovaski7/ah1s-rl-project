"""Kalkış ajanı için kanıt grafikleri (docs/takeoff/fig_takeoff.png).

  python docs/takeoff/fig_takeoff.py --model models_takeoff/takeoff_final.zip

Üç panel grubu:
  1) Neden dört kumanda: aynı yerden kalkış, (a) cyclic / pedal hover triminde sabit + collective rampası (açık döngü),
     (b) ajan. Heading sapması ve pad'den uzaklaşma.
  2) Ajanın dört kumandası: 300 ft'e kalkış, trimden sapma (collective, longitudinal / lateral cyclic, pedal).
  3) İniş: 50 ft hover → pad'e iniş; kızak yüksekliği, dikey hız, kızaklardaki ağırlık oranı, collective.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helicopter_env_takeoff import HelicopterEnvTakeoff, expo_inv  # noqa: E402
from takeoff_curriculum import GROUND_H_FT  # noqa: E402


def fly(env, policy, opts, seconds):
    obs, info = env.reset(seed=0, options=opts)
    psi0 = info["heading_unwrapped"]
    rows, done = [], False
    while not done and info["t"] < seconds:
        obs, r, te, tr, info = env.step(policy(obs, env))
        rows.append((info["t"], info["altitude"], info["err_xy"], info["heading_unwrapped"] - psi0,
                     info["vertical_speed"], info["weight_on_skids"], info["skid_height"], *info["controls"]))
        done = te or tr
    return np.array(rows), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(Path(__file__).with_name("fig_takeoff.png")))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO

    m = PPO.load(args.model, device="cpu")
    agent = lambda o, env: m.predict(o, deterministic=True)[0]          # noqa: E731

    def open_loop(o, env):
        t = env.steps * 0.075
        c = min(0.62, 0.62 * t / 4.0)                                      # collective rampası, 4 s
        a0 = float(expo_inv(np.clip((c - env.trim[0]) / env.rng_ctrl[0], -1, 1), env.cfg.expo))
        return np.array([a0, 0.0, 0.0, 0.0])                               # cyclic / pedal trimde sabit

    env = HelicopterEnvTakeoff(level="K9")
    base = dict(level="K3", start="ground", start_heading_deg=0.0, fuel=(0.0, 0.0))
    ol, _ = fly(env, open_loop, dict(base, tasks=[dict(kind="takeoff", h=100.0)]), 16.0)
    ag, _ = fly(env, agent, dict(base, tasks=[dict(kind="takeoff", h=100.0)]), 16.0)
    to, _ = fly(env, agent, dict(base, tasks=[dict(kind="takeoff", h=300.0)]), 45.0)
    ld, info = fly(env, agent, dict(level="K7", start="hover", start_alt_ft=50.0, start_heading_deg=0.0, start_perturb=0.0,
                                    fuel=(0.0, 0.0), tasks=[dict(kind="hold", T=4.0), dict(kind="land")]), 60.0)

    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    # 1) neden dört kumanda
    a = ax[0, 0]
    a.plot(ol[:, 0], ol[:, 3], label="açık döngü (cyclic / pedal trimde)", color="#c0392b")
    a.plot(ag[:, 0], ag[:, 3], label="ajan (4 kumanda)", color="#2471a3")
    a.set_title("Kalkış → 100 ft: heading sapması (°)"); a.set_xlabel("t (s)"); a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[0, 1]
    a.plot(ol[:, 0], ol[:, 2], color="#c0392b"); a.plot(ag[:, 0], ag[:, 2], color="#2471a3")
    a.set_title("Pad'den uzaklık (ft)"); a.set_xlabel("t (s)"); a.grid(alpha=.3)
    a = ax[0, 2]
    a.plot(ol[:, 0], ol[:, 1], color="#c0392b"); a.plot(ag[:, 0], ag[:, 1], color="#2471a3")
    a.set_title("İrtifa, CG (ft)"); a.set_xlabel("t (s)"); a.grid(alpha=.3)
    # 2) dört kumanda, 300 ft kalkış
    trim = env.trim
    names = ("collective", "long. cyclic", "lat. cyclic", "pedal")
    cols = ("#8e44ad", "#16a085", "#d35400", "#2c3e50")
    a = ax[1, 0]
    a.plot(to[:, 0], to[:, 1], color="#2471a3"); a.axhline(300, ls="--", color="gray", lw=.8)
    a.set_title("Kalkış → 300 ft: irtifa (ft)"); a.set_xlabel("t (s)"); a.grid(alpha=.3)
    a = ax[1, 1]
    for k in range(4):
        a.plot(to[:, 0], to[:, 7 + k] - trim[k], label=names[k], color=cols[k], lw=1)
    a.set_title("Dört kumanda: trimden sapma"); a.set_xlabel("t (s)"); a.legend(fontsize=8, ncol=2); a.grid(alpha=.3)
    a = ax[1, 2]
    a.plot(to[:, 0], to[:, 2], color="#2471a3", label="konum hatası (ft)")
    a.plot(to[:, 0], np.abs(to[:, 3]), color="#c0392b", label="|heading sapması| (°)")
    a.set_title("Kalkışta pad üstünde kalma"); a.set_xlabel("t (s)"); a.legend(fontsize=8); a.grid(alpha=.3)
    # 3) iniş
    a = ax[2, 0]
    a.plot(ld[:, 0], ld[:, 6], color="#2471a3"); a.axhline(0, color="k", lw=.8)
    a.set_title("İniş: kızak yüksekliği (ft)"); a.set_xlabel("t (s)"); a.grid(alpha=.3)
    a = ax[2, 1]
    a.plot(ld[:, 0], ld[:, 4], color="#2471a3"); a.axhline(-4, ls="--", color="#c0392b", lw=.8, label="temas sınırı −4 ft/s")
    a.set_title("Dikey hız (ft/s)"); a.set_xlabel("t (s)"); a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[2, 2]
    a.plot(ld[:, 0], ld[:, 5], color="#16a085", label="kızaklardaki ağırlık oranı")
    a.plot(ld[:, 0], ld[:, 7], color="#8e44ad", label="collective")
    a.axhline(0.7, ls="--", color="gray", lw=.8)
    a.set_title("Oturma: ağırlık kızaklara, collective aşağı"); a.set_xlabel("t (s)"); a.legend(fontsize=8); a.grid(alpha=.3)
    res = info.get("command_results") or []
    fig.suptitle(f"AH-1S kalkış / iniş ajanı — {Path(args.model).name}  (iniş: "
                 + ", ".join(f"{c['kind']} {'✓' if c['success'] else '✗'}" for c in res) + ")", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("kaydedildi:", args.out)


if __name__ == "__main__":
    main()
