from __future__ import annotations

"""
MANEUVER CURRICULUM — Δ komutları + süre hedefi, 0–100 kt
=========================================================

Komut curriculum'unun (command_curriculum.py) devamı. Arayüz aynı: Δheading /
Δhız / Δirtifa. Yeni olan: her komutun bir **süre hedefi** T var. Ajan hedef
bandına T (+ pay) içinde girip orada kalmalı. Süre kısaldıkça helikopterin tek
yolu açıları kullanmak: ileri uçuşta yatışlı dönüş, hızlanmada burun aşağı,
yavaşlamada burun yukarı, tırmanışta collective + yunuslama.

Süre hedefi seviyenin "çeviklik" değerlerinden hesaplanır (komut anındaki hıza
göre; `time_target`):

    T = gecikme + max(|Δψ| / ψ̇_hedef(V), |Δv| / ivme_hedef, |Δh| / tırmanış_hedef)

    ψ̇_hedef(V) = min(pedal_ψ̇, g·tan(yatış_hedef) / V)     (V: ileri hız)

Düşük hızda dönüş pedalla (pedal_ψ̇), ileri uçuşta yatışla sınırlıdır: 60 kt'ta
45° yatış ≈ 18°/s, 100 kt'ta ≈ 11°/s. Arayüz T'yi doğrudan da verebilir.

Referans (ADS-33, askerî helikopter uçuş niteliği standardı): hover / düşük hız
büyük açı değişimlerinde "orta" çeviklik ≈ pitch 13°/s, yaw 22°/s; "agresif"
≈ pitch 30°/s, yaw 50°/s. Bu modelde (JSBSim AH-1S) ölçülen tavanlar: yatışa
geçiş ~22–27°/s, pitch ~15°/s, pedal dönüşü ~25°/s, collective ile ~30 ft/s.
"""

import math
from dataclasses import dataclass

import numpy as np

from command_curriculum import AXES, AxisRange, Level

G_FPS2 = 32.174
KT = 1.6878                                   # ft/s / kt


@dataclass(frozen=True)
class ManeuverLevel(Level):
    """Level + çeviklik hedefleri. start_speed_fps: başlangıç hızı aralığı (ft/s)."""
    pedal_rate_dps: float = 12.0              # düşük hızda dönüş hızı hedefi
    bank_deg: float = 20.0                    # ileri uçuşta dönüş için yatış hedefi
    accel_fps2: float = 2.0                   # hız değişimi
    climb_fps: float = 6.0                    # irtifa değişimi
    lag_s: float = 2.0                        # tepki gecikmesi payı
    n_commands: int = 3                       # episode başına komut sayısı
    speed_bounds_fps: tuple = (0.0, 170.0)    # hedef hız bu aralıkta kalır (Δv işareti buna göre seçilir)

    def time_target(self, cmd: dict, speed_fps: float) -> float:
        return time_target(cmd, speed_fps, self.pedal_rate_dps, self.bank_deg, self.accel_fps2, self.climb_fps,
                           self.lag_s)


def heading_rate_target(speed_fps: float, pedal_rate_dps: float, bank_deg: float) -> float:
    v = max(abs(speed_fps), 1.0)
    bank_rate = math.degrees(G_FPS2 * math.tan(math.radians(bank_deg)) / v)
    return float(min(pedal_rate_dps, bank_rate))


def time_target(cmd: dict, speed_fps: float, pedal_rate_dps: float, bank_deg: float, accel_fps2: float,
                climb_fps: float, lag_s: float) -> float:
    need = [0.0]
    if cmd.get("heading"):
        need.append(abs(cmd["heading"]) / heading_rate_target(speed_fps, pedal_rate_dps, bank_deg))
    if cmd.get("speed"):
        need.append(abs(cmd["speed"]) / accel_fps2)
    if cmd.get("altitude"):
        need.append(abs(cmd["altitude"]) / climb_fps)
    return float(lag_s + max(need))


# ---------------------------------------------------------------------------
# Seviyeler
# ---------------------------------------------------------------------------

_H = lambda mx, prev=0.0, mn=5.0: AxisRange(max_abs=mx, prev_max=prev, min_abs=mn)     # noqa: E731
_V = lambda mx, prev=0.0, mn=3.0: AxisRange(max_abs=mx, prev_max=prev, min_abs=mn)     # noqa: E731
_A = lambda mx, prev=0.0, mn=15.0: AxisRange(max_abs=mx, prev_max=prev, min_abs=mn)    # noqa: E731

_MIX3 = {"heading": 0.45, "speed": 0.30, "altitude": 0.25}

DEFAULT_MANEUVER_LEVELS: list[ManeuverLevel] = [
    ManeuverLevel(
        name="M1", description="Temel kontrol, düşük hız (0–30 kt): küçük komutlar, rahat süre",
        heading=_H(60.0), speed=_V(15.0), altitude=_A(60.0), axis_probs=_MIX3,
        start_alt_ft=(500.0, 900.0), start_speed_fps=(0.0, 50.0),
        pedal_rate_dps=10.0, bank_deg=15.0, accel_fps2=2.0, climb_fps=6.0, lag_s=3.0),
    ManeuverLevel(
        name="M2", description="Düşük hızda çeviklik: hızlı pedal dönüşü, ivmelenme/yavaşlama, bob-up/down",
        heading=_H(180.0, 60.0), speed=_V(30.0, 15.0), altitude=_A(100.0, 60.0), axis_probs=_MIX3,
        start_alt_ft=(500.0, 1000.0), start_speed_fps=(0.0, 50.0),
        pedal_rate_dps=18.0, bank_deg=25.0, accel_fps2=4.0, climb_fps=12.0, lag_s=2.5),
    ManeuverLevel(
        name="M3", description="İleri uçuş (40–80 kt): yatışlı dönüş (30°), hız ve irtifa değişimi",
        heading=_H(120.0, 0.0, 10.0), speed=_V(30.0, 0.0, 5.0), altitude=_A(120.0, 0.0, 20.0), axis_probs=_MIX3,
        start_alt_ft=(600.0, 1200.0), start_speed_fps=(40.0 * KT, 80.0 * KT),
        pedal_rate_dps=18.0, bank_deg=30.0, accel_fps2=4.0, climb_fps=12.0, lag_s=2.5,
        mix_single=0.5, single_ranges={"heading": _H(180.0), "speed": _V(30.0), "altitude": _A(100.0)}),
    ManeuverLevel(
        name="M4", description="Tüm zarf (0–100 kt): 45° yatış, 22°/s pedal, 5 ft/s² ivme, 16 ft/s tırmanış",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 30.0, 5.0), altitude=_A(150.0, 100.0, 20.0), axis_probs=_MIX3,
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT),
        pedal_rate_dps=22.0, bank_deg=45.0, accel_fps2=5.0, climb_fps=16.0, lag_s=2.2),
    ManeuverLevel(
        name="M5", description="Agresif + birleşik: 55° yatış, 25°/s pedal, 6 ft/s² ivme, 20 ft/s tırmanış",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 0.0, 5.0), altitude=_A(150.0, 0.0, 20.0), axis_probs=_MIX3,
        combined=True, mix_single=0.5,
        single_ranges={"heading": _H(180.0), "speed": _V(50.0), "altitude": _A(150.0)},
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT),
        pedal_rate_dps=25.0, bank_deg=55.0, accel_fps2=6.0, climb_fps=20.0, lag_s=2.0),
]


def find_maneuver_level(level, levels=None) -> int:
    levels = levels or DEFAULT_MANEUVER_LEVELS
    if isinstance(level, (int, np.integer)):
        return int(np.clip(level, 0, len(levels) - 1))
    for i, lv in enumerate(levels):
        if lv.name.lower() == str(level).lower():
            return i
    raise KeyError(f"seviye yok: {level}")


def maneuver_level_names(levels=None) -> list[str]:
    return [lv.name for lv in (levels or DEFAULT_MANEUVER_LEVELS)]


__all__ = ["AXES", "ManeuverLevel", "DEFAULT_MANEUVER_LEVELS", "find_maneuver_level", "maneuver_level_names",
           "time_target", "heading_rate_target", "KT", "G_FPS2"]
