"""
İleri kalkış (depart) görevinin yapılabilirlik ve ödül tutarlılığı kontrolü — RL DEĞİL, eğitimde kullanılmaz.

Basit bir PI hız / irtifa / rota tutucusu (probe kontrolcüleriyle aynı yapı) depart görevini uçar; aynı episode'u bir
policy de uçar. Amaç: (1) görev fiziksel olarak ve bant ölçütleriyle yapılabilir mi, (2) ödül takip eden davranışı
duranın üstünde tutuyor mu (ödül tasarımı doğru mu).

  python docs/depart/check_depart_scripted.py --model models_takeoff/takeoff_torque.zip --level K11r --seeds 5 8 11
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import evaluate_takeoff as ET  # noqa: E402
from helicopter_env_takeoff import expo_inv  # noqa: E402


def scripted_action(env, integ):
    s = env._state()
    t = env.steps * 0.075
    vn, ve = env._track_ref(t)[2:] if env.track else (0.0, 0.0)
    ps = math.radians(s["psi_deg"])
    vf = vn * math.cos(ps) + ve * math.sin(ps)
    e = env._errors(s)
    eu = (vf + 0.1 * e["fwd"]) - s["ug"]
    ev = 0.1 * e["right"] - s["vg"]
    evs = float(np.clip(0.3 * e["h"], -6, 6)) - s["vs"]
    eps = e["psi"]
    integ[0] = np.clip(integ[0] + 0.01 * evs * 0.075, -0.3, 0.3)
    integ[1] = np.clip(integ[1] - 0.004 * eu * 0.075, -0.3, 0.3)
    integ[2] = np.clip(integ[2] + 0.004 * ev * 0.075, -0.3, 0.3)
    integ[3] = np.clip(integ[3] - 0.003 * eps * 0.075, -0.3, 0.3)
    th_ref = float(np.clip(-0.01 * eu, -0.25, 0.25)) + integ[1]
    ph_ref = float(np.clip(0.03 * ev, -0.25, 0.25)) + integ[2]
    c = np.array([env.trim[0] + 0.03 * evs + integ[0], env.trim[1] + 3.0 * (s["theta"] - th_ref) + 1.6 * s["q"],
                  env.trim[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"], env.trim[3] - 0.02 * eps + 0.8 * s["r"] + integ[3]])
    return np.clip(expo_inv(np.clip((c - env.trim) / env.rng_ctrl, -1, 1), env.cfg.expo), -1, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", default="K11r")
    ap.add_argument("--seeds", nargs="*", type=int, default=[5, 8, 11])
    ap.add_argument("--env", default='{}', help="env ayarları (JSON), modelinkilerin üstüne")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    env, pol, ov = ET.make_env(args.model, json.loads(args.env), level=args.level)
    out = []
    for seed in args.seeds:
        for mode in ("policy", "scripted"):
            obs, info = env.reset(seed=seed)
            task = dict(env.windows[0]["task"])
            integ = np.array([0.0, -0.03, 0.0, 0.0])
            done, total, parts, us = False, 0.0, collections.Counter(), []
            while not done:
                a = pol(obs) if mode == "policy" else scripted_action(env, integ)
                obs, r, te, tr, info = env.step(a)
                total += r
                us.append(info["speed"])
                for k, v in info["reward_parts"].items():
                    parts[k] += 0.1 * v
                done = te or tr
            res = info["command_results"][0]
            out.append(dict(seed=seed, mode=mode, task=task, success=bool(res["success"]), reward=round(total, 1),
                            end_speed_fps=round(float(np.mean(us[-40:])), 1), termination=info["termination"],
                            parts={k: round(v, 1) for k, v in parts.items()}))
            print(f"seed {seed} {mode:8s} başarı {res['success']!s:5s} ödül {total:6.1f}  son hız {np.mean(us[-40:]):5.1f} ft/s  "
                  f"(hedef {task.get('v_kt', 0) * 1.68781:5.1f})  {info['termination']}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
