"""
CHECK NATURAL LIMITS (2026-10-01) — doğal komut zarfı ve komut yönlendiricisinin kontrolü (RL değil; ölçüm)
=========================================================================================================

Repo kökünden:
  python docs/flight/check_natural_limits.py                       # flight_v2 ile
  python docs/flight/check_natural_limits.py --model models_flight/flight_final.zip

1) Zarf dışına düşen canlı Δ komutları ters ÇEVRİLMİYOR, doğal sınıra kırpılıyor (2026-09-30'da 40 kt'ta −20 kt → 60 kt idi).
2) Komut yönlendirici rejime göre görev seçiyor; rejime uymayan komut episode'u bitirmiyor (2026-09-30'da 80 kt'ta hover
   dönüşü → 0.1 s'de speed_limit idi).
3) Doğal zarfın uçları ve pirouette (ajanın başarısı; F1–F9'da yoktu).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import evaluate_flight as ef  # noqa: E402
from flight_commands import apply, route_action, route_command, regime  # noqa: E402
from helicopter_env_command import CONTROL_DT  # noqa: E402

KT = 1.6878099


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "models_flight" / "flight_v2.zip"))
    args = ap.parse_args(argv)
    env, pol, _ = ef.make_env(args.model)

    def start(**o):
        base = dict(level="F8", start_heading_deg=0.0, start_perturb=0.0, fuel=(300.0, 300.0),
                    physics=dict(wind_kt=0.0, turb_level="none"), live=True, episode_s=600.0)
        base.update(o)
        obs, _ = env.reset(seed=0, options=base)
        for _ in range(40):
            obs, *_ = env.step(pol(obs))
        return obs

    def fly(obs, secs):
        info = {}
        for _ in range(int(secs / CONTROL_DT)):
            obs, r, term, trunc, info = env.step(pol(obs))
            if term or trunc:
                return obs, info, True
        return obs, info, False

    print("1) Zarf dışı Δ komutları (env'e doğrudan) — beklenen: kırpma, ters çevirme yok")
    ch = dict(kind="cruise", hold=True)
    for title, st, task, want in (
            ("40 kt, Δhız −20 kt", dict(start="cruise", start_speed_kt=40.0, start_alt_ft=400.0, tasks=[ch]),
             dict(kind="cruise", du_kt=-20.0), "20 kt"),
            ("100 kt, Δhız +45 kt", dict(start="cruise", start_speed_kt=100.0, start_alt_ft=400.0, tasks=[ch]),
             dict(kind="cruise", du_kt=45.0), "130 kt (kırpma)"),
            ("900 ft, Δirtifa +800 ft", dict(start="cruise", start_speed_kt=80.0, start_alt_ft=900.0, tasks=[ch]),
             dict(kind="cruise", dh=800.0), "1500 ft (kırpma)"),
            ("250 ft, Δirtifa −300 ft", dict(start="cruise", start_speed_kt=80.0, start_alt_ft=250.0, tasks=[ch]),
             dict(kind="cruise", dh=-300.0), "50 ft (kırpma)"),
            ("hover 30 ft, bob −25 ft", dict(start="hover", start_alt_ft=30.0, tasks=[dict(kind="hold")]),
             dict(kind="bob", dh=-25.0), "12 ft (kırpma)")):
        obs = start(**st)
        env.queue_task(task)
        env.step(pol(obs))
        w = env.windows[-1]
        u_t = env.cruise["u"] / KT if env.cruise else float("nan")
        print(f"   {title:<26} → hedef {u_t:6.1f} kt / {w['target']['h']:6.1f} ft  (beklenen {want}; kırpma {w.get('clipped')})")

    print("2) Komut yönlendirici — beklenen: episode bitmez, istenen rejim / hedef")
    crz = dict(start="cruise", start_speed_kt=80.0, start_alt_ft=300.0, tasks=[ch])
    hov = dict(start="hover", start_alt_ft=100.0, tasks=[dict(kind="hold")])
    for title, st, mk, secs in (
            ("80 kt: iniş", crz, lambda: route_action(env, "land"), 150.0),
            ("80 kt: heading 270 (mutlak)", crz, lambda: route_command(env, heading_deg=270.0), 60.0),
            ("80 kt: hız 0, irtifa 100 ft", crz, lambda: route_command(env, speed_kt=0.0, alt_ft=100.0), 120.0),
            ("hover: Δψ +90, Δh +200", hov, lambda: route_command(env, dheading_deg=90.0, dalt_ft=200.0), 90.0),
            ("hover: hız 60, heading 90", hov, lambda: route_command(env, speed_kt=60.0, heading_deg=90.0), 90.0),
            ("hover: pirouette", hov, lambda: route_action(env, "pirouette"), 80.0)):
        obs = start(**st)
        reg = regime(env)
        res = mk()
        if res.ok:
            apply(env, res)
        obs, info, ended = fly(obs, secs)
        print(f"   {title:<28} [{reg}] {res.message!r} → {secs:.0f} s sonra: bitti={ended} {info.get('termination', '')} "
              f"hız {info.get('airspeed_kt', np.nan):.0f} kt, h {info.get('altitude', np.nan):.0f} ft, "
              f"ψ {info.get('heading_deg', np.nan):.0f}°, kızak {info.get('wow')}")

    print("3) Doğal zarfın uçları — ajanın başarısı (ayrı test takımında da var: evaluate_flight --suite test)")
    res, summ = ef.evaluate(args.model, only=[s[0] for s in ef.TEST_SCENARIOS if s[1] == "test/doğal"], suite="test")
    return summ


if __name__ == "__main__":
    main()
