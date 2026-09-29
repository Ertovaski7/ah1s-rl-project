from __future__ import annotations

"""
TRIM TABLE — hava hızına dayalı trim (gövde ekseninde ileri HAVA hızı × ağırlık), physics-ext probe (e)
=======================================================================================================

JSBSim AH-1S'in `steady_flight_data.xml` tabloları 8500 lbs için ve YER hızıyla bakılıyor → rüzgârda (hover-in-wind:
yer hızı 0 ama hava hızı rüzgâr kadar) ve yana uçuşta yanlış; ayrıca uzunlamasına / yanal cyclic sütunları bizim AFCS
ayarımızda (yalnızca SAS) ~0.1'e kadar farklı (docs/physics_ext/probe_e_trim_table.txt). Bu tablo aynı modelde, düz ve
dengeli uçuşta (yana hava hızı 0, heading sabit, 300 ft AGL, iki tank eşit) ölçüldü: u_air −20…120 kt × 8500…10,280 lbs.
u_air = 0'da tablo sabit hover trimine eşit (8500 lbs: 0.605 / −0.151 / 0.193 / 0.412 ≈ HOVER_TRIM).

Kullanım:
    tt = AirspeedTrimTable()                   # docs/physics_ext/trim_table_airspeed.json
    c = tt(u_air_kt=35.0, weight_lbs=9400.0)   # → [collective, elevator, aileron, rudder]
    tt.attitude(35.0, 9400.0)                  # → (θ, φ) derece
İrtifa: 1000 ft AGL'de collective ~+0.010, pedal ~+0.002 (probe e); `h_agl_ft` verilirse doğrusal düzeltme.
"""

import json
from pathlib import Path

import numpy as np

DEFAULT_TABLE = Path(__file__).resolve().parent / "docs" / "physics_ext" / "trim_table_airspeed.json"
# 1000 ft − 300 ft AGL farkı (probe e, 8500 / 10,280 lbs ortalaması), 1000 ft başına: collective, elevator, aileron, rudder
ALT_SLOPE_PER_1000FT = np.array([0.0143, 0.0045, 0.0044, 0.0027])


class AirspeedTrimTable:
    def __init__(self, path: str | Path = DEFAULT_TABLE):
        d = json.loads(Path(path).read_text())
        self.u = np.asarray(d["u_kt"], dtype=np.float64)
        self.w = np.asarray(d["weight_lbs"], dtype=np.float64)
        self.h_ref = float(d.get("h_agl_ft", 300.0))
        self.ctrl = np.stack([np.asarray(d[k], dtype=np.float64) for k in ("collective", "elevator", "aileron", "rudder")],
                             axis=-1)                                          # [ağırlık, hız, 4]
        self.att = np.stack([np.asarray(d[k], dtype=np.float64) for k in ("theta_deg", "phi_deg")], axis=-1)
        self.psi = np.asarray(d["torque_psi"], dtype=np.float64)
        self.source = d.get("source", "")

    def _interp(self, grid: np.ndarray, u_kt: float, w_lbs: float) -> np.ndarray:
        u = float(np.clip(u_kt, self.u[0], self.u[-1]))
        w = float(np.clip(w_lbs, self.w[0], self.w[-1]))
        i = int(np.clip(np.searchsorted(self.u, u) - 1, 0, len(self.u) - 2))
        j = int(np.clip(np.searchsorted(self.w, w) - 1, 0, len(self.w) - 2))
        tu = (u - self.u[i]) / (self.u[i + 1] - self.u[i])
        tw = (w - self.w[j]) / (self.w[j + 1] - self.w[j])
        a = grid[j, i] * (1 - tu) + grid[j, i + 1] * tu
        b = grid[j + 1, i] * (1 - tu) + grid[j + 1, i + 1] * tu
        return a * (1 - tw) + b * tw

    def __call__(self, u_air_kt: float, weight_lbs: float, h_agl_ft: float | None = None) -> np.ndarray:
        c = self._interp(self.ctrl, u_air_kt, weight_lbs)
        if h_agl_ft is not None:
            c = c + ALT_SLOPE_PER_1000FT * (float(h_agl_ft) - self.h_ref) / 1000.0
        return c

    def attitude(self, u_air_kt: float, weight_lbs: float) -> np.ndarray:
        return self._interp(self.att, u_air_kt, weight_lbs)

    def torque_psi(self, u_air_kt: float, weight_lbs: float) -> float:
        return float(self._interp(self.psi[..., None], u_air_kt, weight_lbs)[0])


if __name__ == "__main__":
    tt = AirspeedTrimTable()
    print("hover 8500:", np.round(tt(0.0, 8500.0), 4), " 60 kt 10280:", np.round(tt(60.0, 10280.0), 4),
          " θ/φ @ 100 kt 9000:", np.round(tt.attitude(100.0, 9000.0), 2), " psi @ 70 kt 8500:", round(tt.torque_psi(70, 8500), 1))
