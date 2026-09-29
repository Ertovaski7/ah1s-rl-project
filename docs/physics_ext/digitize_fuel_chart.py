"""TM 55-1520-234-10 (AH-1S operatör el kitabı, 1976, Change 30'a kadar) şekil 7-8 sayfa 7'nin (seyir, 4 TOW, 324 rotor /
6600 motor rpm, JP-4, ECU kapalı, OGE, FAT +15 °C; deniz seviyesi / 2000 / 4000 / 6000 ft panelleri) tork ↔ yakıt akışı
ölçeklerinin sayısallaştırılması.

Kaynak PDF: https://archive.org/details/TM55152023410AH1S (PDF sayfa 205 = el kitabı sayfası 7-28). Sayfa 400 dpi
render edildi (pdftoppm -r 400); her panelde üst eksen (yakıt akışı) ve alt eksen (tork, psi) çentiklerinin x pikselleri
görüntü işlemeyle bulundu (çentik = eksen çizgisinin hemen üstünde / altında koyu sütun). El kitabı: "torque pressure may
be converted directly to fuel flow without regard for other chart information" → aynı x'teki iki ölçek okuması doğrudan
eşleşir. Aşağıdaki piksel listeleri o ölçümün kaydıdır; bu script onlardan psi = 10…50'de yakıt akışını ve her irtifa için
doğru uydurmasını üretir (physics_ext.FUEL_CHART_A / FUEL_CHART_B bunlardan).

Çıktı: fuel_chart_digitized.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
# panel: (yakıt akışı etiketleri lb/h, çentik x pikselleri), tork çentikleri (10, 20, … 60 psi) x pikselleri
PANELS = {
    0.0: ((400, 450, 500, 550, 600, 650, 700, 750, 800), (560, 658, 760, 862, 972, 1084, 1189, 1300, 1403),
          (528, 726, 928, 1126, 1326, 1527)),
    2000.0: ((400, 450, 500, 550, 600, 650, 700, 750, 800), (1996, 2094, 2197, 2305, 2414, 2524, 2635, 2740, 2843),
             (1926, 2126, 2324, 2526, 2724, 2926)),
    4000.0: ((350, 400, 450, 500, 550, 600, 650, 700, 750, 800),
             (539, 637, 740, 844, 952, 1062, 1165, 1275, 1376, 1476), (527, 728, 928, 1127, 1326, 1526)),
    6000.0: ((350, 400, 450, 500, 550, 600, 650, 700, 750, 800),
             (1966, 2072, 2174, 2283, 2393, 2500, 2602, 2710, 2810, 2900), (1928, 2126, 2325, 2525, 2726, 2926)),
}
PSI_TICKS = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)


def flow_at(x: float, ff, xs) -> float:
    ff, xs = np.asarray(ff, float), np.asarray(xs, float)
    if x < xs[0]:
        return float(ff[0] + (x - xs[0]) * (ff[1] - ff[0]) / (xs[1] - xs[0]))
    if x > xs[-1]:
        return float(ff[-1] + (x - xs[-1]) * (ff[-1] - ff[-2]) / (xs[-1] - xs[-2]))
    return float(np.interp(x, xs, ff))


def main():
    rows = []
    print("irtifa ft |  a (lb/h)   b (lb/h/psi) | artık en çok")
    for alt, (ff, xs, xt) in PANELS.items():
        vals = [flow_at(x, ff, xs) for x in xt[:5]]                # 10…50 psi (60 psi eksen dışı uzatma, alınmadı)
        A = np.vstack([np.ones(5), PSI_TICKS[:5]]).T
        coef = np.linalg.lstsq(A, vals, rcond=None)[0]
        res = np.asarray(vals) - A @ coef
        print(f"{alt:8.0f}  | {coef[0]:8.1f}   {coef[1]:8.3f}     | {np.abs(res).max():.1f}")
        for p, v in zip(PSI_TICKS[:5], vals):
            rows.append(dict(pressure_alt_ft=alt, psi=p, fuel_flow_lbh=round(v, 1), fit_a=round(coef[0], 1),
                             fit_b=round(coef[1], 3)))
    with open(OUT / "fuel_chart_digitized.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
