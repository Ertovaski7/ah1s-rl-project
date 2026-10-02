"""
COMPARE EVAL — evaluate_flight JSON'larını yan yana özetler (markdown tablo)
==========================================================================

  python docs/flight/compare_eval.py flight_final=docs/flight/eval_test_final.json flight_v2=docs/flight/eval_test_v2.json
  # bir koşu için birden çok JSON (ör. takım + seviye istatistikleri): ad=a.json,b.json
"""
import json
import sys


def load(paths):
    out = {}
    for path in paths.split(","):
        d = json.load(open(path, encoding="utf-8"))
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**v, **out[k]}                 # ilk dosya önceliklidir
            else:
                out.setdefault(k, v)
    return out


def frac(x):
    return f"{x[0]}/{x[1]}" if x else "—"


def main(argv):
    runs = [(a.split("=", 1)[0], load(a.split("=", 1)[1])) for a in argv]
    groups = []
    for _, d in runs:
        for g in (d.get("summary") or {}):
            if g not in groups and g not in ("kategori", "tork", "yakit", "hassasiyet"):
                groups.append(g)
    print("| ölçüt | " + " | ".join(n for n, _ in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for g in groups:
        row = []
        for _, d in runs:
            x = (d.get("summary") or {}).get(g)
            row.append(f"{frac(x['episode_ok'])} · {frac(x['tasks_ok'])}" if x else "—")
        print(f"| {g} (tüm görevler · görev) | " + " | ".join(row) + " |")
    for key, label, fmt in (("senaryo_56_ustu", "56 psi aşılan senaryo", "frac"), ("sure_56_ustu_s", "56 psi üstü toplam (s)", ".0f"),
                            ("en_uzun_56_ustu_s", "56 psi üstü en uzun (s)", ".1f"), ("maks_psi", "en yüksek tork (psi)", ".1f")):
        row = []
        for _, d in runs:
            t = (d.get("summary") or {}).get("tork")
            row.append("—" if not t else (frac(t[key]) if fmt == "frac" else format(t[key], fmt)))
        print(f"| {label} | " + " | ".join(row) + " |")
    row = []
    for _, d in runs:
        h = (d.get("summary") or {}).get("hassasiyet") or {}
        dk = h.get("donus_kayma_ft") or [None]
        pr = h.get("pirouette_uzaklik_ft") or [None]
        row.append("—" if dk[0] is None else f"dönüş {dk[0]:.1f} ft" + ("" if pr[0] is None else f", pirouette {pr[0]:.1f} ft"))
    print("| hassasiyet (en büyük kayma) | " + " | ".join(row) + " |")
    lv = []
    for _, d in runs:
        for k in (d.get("levels") or {}):
            if k not in lv:
                lv.append(k)
    for k in lv:
        row = []
        for _, d in runs:
            x = (d.get("levels") or {}).get(k)
            bad = sum(v for t, v in ((x or {}).get("termination") or {}).items() if t not in ("time_limit", "fuel_exhausted"))
            row.append("—" if not x else f"%{100 * x['episode']:.0f} ({x['n']})" + (f", {bad} güvensiz" if bad else ""))
        print(f"| seviye {k} (episode başarısı) | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
