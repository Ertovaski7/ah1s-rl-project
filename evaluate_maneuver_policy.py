from __future__ import annotations

"""
EVALUATE MANEUVER POLICY — süre hedefli komutlar ve çeviklik ölçüleri
=====================================================================

Manevra ajanını (helicopter_env_maneuver.py) farklı hızlarda tek komutlarla
dener; her komut için:

  oturma   : tüm eksenlerin birlikte banda son girişi (komuttan itibaren, s)
  T / son  : süre hedefi ve son sınır (T·1.25, en az T+1 s) — başarı için oturma ≤ son sınır
  max|φ|, max|θ| : en büyük yatış / yunuslama (komut öncesine göre)
  max|r|, max|q| : en büyük yaw / pitch hızı (°/s),  max|ḣ| (ft/s), max|u̇| (ft/s²)
  kuplaj   : komut verilmeyen eksendeki en büyük sapma

İsteğe bağlı karşılaştırma: eski komut modeli (models_command_curriculum/v2_R1_final.zip,
helicopter_env_command.py, 15 ft/s) aynı büyüklükteki komutlarda ne kadar sürede ve hangi
açılarla oturuyor.

Kullanım
  python evaluate_maneuver_policy.py --model runs/man/models/latest.zip
  python evaluate_maneuver_policy.py --model models_maneuver/m5_final.zip --level M5 --compare-old --json out.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helicopter_env_command import CONTROL_DT  # noqa: E402

KT = 1.6878
AXES = ("heading", "speed", "altitude")


def load(path):
    from stable_baselines3 import PPO
    m = PPO.load(str(path), device="cpu")
    return lambda o: m.predict(o, deterministic=True)[0]


def load_maneuver(path):
    """(policy, ManeuverEnvConfig) — env, modelin eğitildiği ayarlarla (ör. collective yetkisi) kurulur."""
    from helicopter_env_maneuver import load_maneuver_policy
    return load_maneuver_policy(path)


def fly(env, policy, cmd: dict, start: dict, T: float | None, episode_s: float = 45.0, t_cmd: float = 3.0):
    item = (t_cmd, dict(cmd), T) if T else (t_cmd, dict(cmd))
    obs, info = env.reset(seed=0, options=dict(commands=[item], episode_s=episode_s, **start))
    rows, done = [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        rows.append((info["t"], info["roll_deg"], info["pitch_deg"], info["yaw_rate_dps"], info["vertical_speed"],
                     info["speed"], info["err_heading"], info["err_speed"], info["err_altitude"]))
        done = term or trunc
    a = np.array(rows)
    res = (info.get("command_results") or [{}])[0]
    return a, res, info


def metrics(a: np.ndarray, res: dict, t_cmd: float) -> dict:
    t = a[:, 0]
    pre = t < t_cmd
    post = t >= t_cmd
    base_phi = a[pre, 1].mean() if pre.any() else 0.0
    base_th = a[pre, 2].mean() if pre.any() else 0.0
    du = np.gradient(a[:, 5], t) if len(t) > 2 else np.zeros_like(t)
    q = np.gradient(a[:, 2], t) if len(t) > 2 else np.zeros_like(t)
    m = dict(max_roll=float(np.abs(a[post, 1] - base_phi).max()), max_pitch=float(np.abs(a[post, 2] - base_th).max()),
             max_r=float(np.abs(a[post, 3]).max()), max_q=float(np.abs(q[post]).max()), max_vs=float(np.abs(a[post, 4]).max()),
             max_acc=float(np.abs(du[post]).max()))
    settle = res.get("settle_s")
    if settle is None:                               # eski env: tüm eksenlerin banda son girişi
        m["settle_s"] = float("nan")
    else:
        m["settle_s"] = float(settle)
    m.update(success=bool(res.get("success", False)), on_time=res.get("on_time"), T=res.get("T"),
             deadline=res.get("deadline"), coupling={k: float(v) for k, v in (res.get("max_abs_err") or {}).items()})
    return m


def settle_all(a: np.ndarray, t_cmd: float, tol=(1.5, 1.5, 10.0)) -> float:
    """Eski env için: üç eksen birlikte bantta kalmaya başladığı an (komuttan itibaren)."""
    t = a[:, 0]
    inside = (np.abs(a[:, 6]) <= tol[0]) & (np.abs(a[:, 7]) <= tol[1]) & (np.abs(a[:, 8]) <= tol[2])
    post = t >= t_cmd + CONTROL_DT - 1e-9
    idx = np.flatnonzero(post)
    if not inside[idx[-1]]:
        return float("nan")
    k = idx[-1]
    while k - 1 >= idx[0] and inside[k - 1]:
        k -= 1
    return float(t[k] - t_cmd)


BATTERY = [
    # (etiket, başlangıç hızı ft/s, komut)
    ("hover-ish 5 kt", 5 * KT, {"heading": 90.0}), ("hover-ish 5 kt", 5 * KT, {"heading": 180.0}),
    ("hover-ish 5 kt", 5 * KT, {"altitude": 100.0}), ("hover-ish 5 kt", 5 * KT, {"altitude": -100.0}),
    ("hover-ish 5 kt", 5 * KT, {"speed": 40.0}),
    ("30 kt", 30 * KT, {"heading": 90.0}), ("30 kt", 30 * KT, {"speed": -40.0}), ("30 kt", 30 * KT, {"speed": 40.0}),
    ("60 kt", 60 * KT, {"heading": 90.0}), ("60 kt", 60 * KT, {"heading": -180.0}), ("60 kt", 60 * KT, {"altitude": 150.0}),
    ("60 kt", 60 * KT, {"speed": -50.0}), ("60 kt", 60 * KT, {"heading": 60.0, "speed": 15.0, "altitude": 80.0}),
    ("90 kt", 90 * KT, {"heading": 90.0}), ("90 kt", 90 * KT, {"speed": -50.0}), ("90 kt", 90 * KT, {"altitude": -120.0}),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", default="M5", help="süre hedefi bu seviyenin çevikliğinden (M3 rahat, M4 hızlı, M5 agresif)")
    ap.add_argument("--start-alt", type=float, default=900.0)
    ap.add_argument("--compare-old", action="store_true", help="eski komut modeliyle (15 ft/s) karşılaştır")
    ap.add_argument("--old-model", default=str(REPO_ROOT / "models_command_curriculum" / "v2_R1_final.zip"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from helicopter_env_maneuver import HelicopterEnvManeuver
    pol, cfg = load_maneuver(args.model)
    env = HelicopterEnvManeuver(level=args.level, config=cfg)
    out = []
    print(f"MANEVRA — model {Path(args.model).name}, süre hedefi {args.level} çevikliği, başlangıç {args.start_alt:.0f} ft, "
          f"collective yetkisi ±{cfg.coll_scale:g}")
    print(f"{'rejim':15s} {'komut':30s} {'T':>5s} {'son':>5s} {'oturma':>7s} {'ok':>3s} {'max|φ|':>7s} {'max|θ|':>7s} "
          f"{'max|r|':>7s} {'max|q|':>6s} {'max|ḣ|':>6s} {'max|u̇|':>6s}  kuplaj")
    for label, u0, cmd in BATTERY:
        a, res, info = fly(env, pol, cmd, dict(start_speed_fps=u0, start_alt_ft=args.start_alt, start_heading_deg=0.0), None)
        m = metrics(a, res, 3.0)
        coup = " ".join(f"{k[:3]}={v:.1f}" for k, v in m["coupling"].items() if not cmd.get(k))
        ctext = " ".join(f"{k[:3]}{v:+g}" for k, v in cmd.items())
        print(f"{label:15s} {ctext:30s} {m['T']:5.1f} {m['deadline']:5.1f} {m['settle_s']:6.1f}s {'✓' if m['success'] else '✗':>3s} "
              f"{m['max_roll']:6.1f}° {m['max_pitch']:6.1f}° {m['max_r']:5.1f}°/s {m['max_q']:4.1f} {m['max_vs']:6.1f} {m['max_acc']:6.1f}  "
              f"{coup}  {info.get('termination')}")
        out.append(dict(model="maneuver", regime=label, u0=u0, cmd=cmd, **m, termination=info.get("termination")))
    ok = sum(r["success"] for r in out)
    print(f"\nbaşarı (süre hedefi dahil): {ok}/{len(out)}")

    if args.compare_old:
        from helicopter_env_command import HelicopterEnvCommand
        old_env = HelicopterEnvCommand(level="R1")
        old_pol = load(args.old_model)
        print(f"\nKARŞILAŞTIRMA — aynı büyüklükte komut, 15 ft/s, 900 ft (eski model kendi env'inde; yeni model süre hedefli)")
        print(f"{'komut':22s} {'model':10s} {'oturma':>7s} {'max|φ|':>7s} {'max|θ|':>7s} {'max|r|':>7s} {'max|ḣ|':>6s} {'max|u̇|':>6s}")
        for cmd in ({"heading": 90.0}, {"heading": 180.0}, {"altitude": 100.0}, {"speed": 10.0}):
            a, res, info = fly(old_env, old_pol, cmd, dict(start_speed_fps=15.0, start_alt_ft=args.start_alt,
                                                           start_heading_deg=0.0), None, episode_s=60.0)
            m_old = metrics(a, res, 3.0)
            m_old["settle_s"] = settle_all(a, 3.0)
            a2, res2, _ = fly(env, pol, cmd, dict(start_speed_fps=15.0, start_alt_ft=args.start_alt, start_heading_deg=0.0), None)
            m_new = metrics(a2, res2, 3.0)
            ctext = " ".join(f"{k[:3]}{v:+g}" for k, v in cmd.items())
            for name, m in (("eski", m_old), ("manevra", m_new)):
                print(f"{ctext:22s} {name:10s} {m['settle_s']:6.1f}s {m['max_roll']:6.1f}° {m['max_pitch']:6.1f}° "
                      f"{m['max_r']:5.1f}°/s {m['max_vs']:6.1f} {m['max_acc']:6.1f}")
                out.append(dict(model=name, regime="15 ft/s karşılaştırma", u0=15.0, cmd=cmd, **m))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        print(f"kaydedildi: {args.json}")


if __name__ == "__main__":
    main()
