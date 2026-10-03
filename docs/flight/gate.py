"""Regresyon kapısı (2026-10-02): yeni bir model, referans modelin hiçbir becerisini geriletmemeli.

Üç test: seçim takımı (21 senaryo), iniş stres taraması (24 iniş), held-out takım (35). Her test için kategori başına
başarı sayılır. Kural: aday, her kategoride referansın en az (referans − tolerans) kadarını yapmalı; güvenlik (düşme /
sınır aşımı) sayısı referanstan fazla olamaz. Çıktı: kategori tablosu ve GEÇTİ / KALDI.

Kullanım:
  python docs/flight/gate.py --ref models_flight/flight_v3.zip --cand runs/x/models/snap_01500k.zip --out /tmp/gate
  (önceden koşulmuş JSON'lar varsa --ref-json / --cand-json klasörleriyle yeniden koşmaz)
"""
import argparse, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV = '{"torque_density_climb": false, "next_at_deadline": false}'


def run_all(model: str, out: Path, tag: str):
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for suite in ("secim", "test"):
        f = out / f"{suite}_{tag}.json"
        if not f.exists():
            jobs.append(subprocess.Popen([sys.executable, str(REPO / "evaluate_flight.py"), "--model", model, "--suite", suite,
                                          "--env", ENV, "--json", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    f = out / f"landing_{tag}.json"
    if not f.exists():
        jobs.append(subprocess.Popen([sys.executable, str(REPO / "docs/flight/probe_landing.py"), str(f), model],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for j in jobs:
        j.wait()


def categories(out: Path, tag: str) -> dict:
    cat = defaultdict(lambda: [0, 0])
    unsafe = 0
    for suite in ("secim", "test"):
        d = json.load(open(out / f"{suite}_{tag}.json"))
        for r in d["results"]:
            unsafe += int(not r["safe"])
            for w in r["windows"]:
                if w["ads33"] == "kesildi":
                    continue
                k = f"{suite}/{w['category']}"
                cat[k][0] += int(w["success"]); cat[k][1] += 1
    L = json.load(open(out / f"landing_{tag}.json"))
    rows = next(iter(L.values()))["rows"]
    cat["iniş-stres/başarılı"] = [sum(r["ok"] for r in rows), len(rows)]
    unsafe += sum(r["term"] not in ("time_limit", "fuel_exhausted") for r in rows)
    cat["güvensiz (toplam)"] = [unsafe, None]
    return dict(cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--cand", required=True)
    ap.add_argument("--out", default="/tmp/gate"); ap.add_argument("--tol", type=int, default=1,
                    help="kategori başına izin verilen gerileme (başarı sayısı)")
    a = ap.parse_args()
    out = Path(a.out)
    run_all(a.ref, out, "ref"); run_all(a.cand, out, "cand")
    R, C = categories(out, "ref"), categories(out, "cand")
    ok = True
    print(f"{'kategori':28s} {'referans':>10s} {'aday':>10s}  durum")
    for k in sorted(set(R) | set(C)):
        r, c = R.get(k, [0, 0]), C.get(k, [0, 0])
        if k.startswith("güvensiz"):
            bad = c[0] > r[0]
            print(f"{k:28s} {r[0]:>10d} {c[0]:>10d}  {'KALDI' if bad else 'ok'}")
        else:
            bad = c[0] < r[0] - a.tol
            print(f"{k:28s} {r[0]:>4d}/{r[1]:<5d} {c[0]:>4d}/{c[1]:<5d}  {'KALDI' if bad else ('↑' if c[0] > r[0] else 'ok')}")
        ok = ok and not bad
    print("\nSONUÇ:", "GEÇTİ" if ok else "KALDI")
    # Bilgi (2026-10-03): ikili başarı sınırdaki zamanlamada gürültülü (ör. flight_v4 −100 ft komutunu son sınırdan
    # 0.0–2.2 s önce bitiriyor; 0.1 s gecikme "başarısız" sayılıyor). Kategori başına "yeterli veya iyi" sayısı ve banda
    # giriş / son sınır medyanı — gerçek gerilemeyi zamanlama gürültüsünden ayırmak için. Karar kuralını değiştirmez.
    print("\nbilgi: yeterli+ sayısı ve banda giriş / son sınır medyanı (referans → aday)")
    Ra, Ca = adequacy(out, "ref"), adequacy(out, "cand")
    for k in sorted(set(Ra) | set(Ca)):
        r, c = Ra.get(k, [0, 0, None]), Ca.get(k, [0, 0, None])
        f = lambda x: "—" if x is None else f"{x:.2f}"          # noqa: E731
        flag = "  ← yeterli+ geriledi" if c[0] < r[0] - a.tol else ""
        print(f"  {k:26s} yeterli+ {r[0]:>3d}/{r[1]:<3d} → {c[0]:>3d}/{c[1]:<3d}   giriş/sınır {f(r[2])} → {f(c[2])}{flag}")
    return 0 if ok else 1


def adequacy(out: Path, tag: str) -> dict:
    import statistics
    cat = defaultdict(lambda: [0, 0, []])
    for suite in ("secim", "test"):
        for r in json.load(open(out / f"{suite}_{tag}.json"))["results"]:
            for w in r["windows"]:
                if w["ads33"] == "kesildi":
                    continue
                k = f"{suite}/{w['category']}"
                cat[k][0] += int(w["ads33"] in ("istenen", "yeterli")); cat[k][1] += 1
                if w.get("settle_s") is not None and w.get("deadline"):
                    cat[k][2].append(w["settle_s"] / w["deadline"])
    return {k: [v[0], v[1], statistics.median(v[2]) if v[2] else None] for k, v in cat.items()}


if __name__ == "__main__":
    sys.exit(main())
