"""Uygulama katmanının sabitleri: platformlar, modüller, başlangıç konumları, başlangıç koşullarının sınırları.

Sınırlar simülasyonun kendi sabitlerinden okunur (flight_curriculum / takeoff_curriculum; yalnızca numpy ister),
böylece formdaki aralıklar canlı sunucunun (command_viz.py) kabul ettikleriyle aynı kalır.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flight_curriculum import (CMD_MAX_ALT_FT, CMD_MAX_KT, CRUISE_MAX_ALT_FT, CRUISE_MAX_KT,  # noqa: E402
                               CRUISE_MIN_ALT_FT, CRUISE_MIN_KT)
from takeoff_curriculum import TANK_CAPACITY_LBS  # noqa: E402

LIVE_SERVER = REPO_ROOT / "command_viz.py"
# tek ajan (kalkış → hover → ileri uçuş → iniş); command_viz.FLIGHT_POLICY ile aynı seçim
FLIGHT_MODEL = next((p for p in (REPO_ROOT / "models_flight" / "flight_v2.zip",
                                 REPO_ROOT / "models_flight" / "flight_final.zip") if p.exists()),
                    REPO_ROOT / "models_flight" / "flight_final.zip")


# ---------------------------------------------------------------------
# ekranlar: platformlar ve AH-1S modülleri (renkler şimdilik düz yer tutucu)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Tile:
    key: str
    title: str
    subtitle: str
    color: tuple[int, int, int]
    ready: bool = False


PLATFORMS = (
    Tile("ah1s", "AH-1S Cobra", "Taarruz helikopteri", (78, 84, 52), ready=True),
    Tile("t129", "T129 ATAK", "Taarruz ve taktik keşif helikopteri", (104, 98, 80)),
    Tile("t70", "T70", "Genel maksat helikopteri", (78, 90, 104)),
)
AH1S_MODULES = (
    Tile("flight", "Sürüş", "Başlangıç koşullarını seç, uç", (63, 95, 74), ready=True),
    Tile("refuel", "Yakıt İkmali", "", (107, 91, 62)),
    Tile("asym", "Asimetrik Savaş Senaryoları", "", (98, 66, 72)),
    Tile("emergency", "Motor Arızası ve Acil Durum Prosedürleri", "", (74, 79, 99)),
)
COMING_SOON = "Coming soon"


# ---------------------------------------------------------------------
# başlangıç konumları (Türkiye haritasında önceden seçili)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Location:
    key: str
    name: str
    region: str
    lat: float
    lon: float
    elev_m: float          # gerçek yükseklik
    note: str = ""


# Fizik modelinin zemini 2283.5 ft (696 m; aircraft/ah1s/reset00.xml) ve değişmiyor: konum yalnızca enlem / boylamı
# taşır (JSBSim, Tacview / ACMI kaydı). Diyarbakır (Havalimanı, 686 m) bu yüksekliğe en yakın Güneydoğu Anadolu
# konumu: hava yoğunluğu ve güç payı gerçekteki gibi kalır. 3D arazi prosedürel (bölgenin görünümü, gerçek topografya değil).
SIM_GROUND_M = 2283.5 * 0.3048
LOCATIONS = (
    Location("diyarbakir", "Diyarbakır", "Güneydoğu Anadolu", 37.894, 40.201, 686.0,
             "Havalimanı çevresi; yükseklik fizik modelinin zeminiyle aynı (≈ 690 m)."),
)
DEFAULT_LOCATION = LOCATIONS[0]


# ---------------------------------------------------------------------
# başlangıç koşulları
# ---------------------------------------------------------------------
START_MODES = (("ground", "Yerde"), ("hover", "Hover"), ("cruise", "İleri uçuş"))
TURB_LEVELS = (("none", "Yok"), ("light", "Hafif"), ("moderate", "Orta"), ("severe", "Şiddetli"))

# canlı sunucunun kabul ettiği aralıklar (command_viz.LiveFlight._request_reset_takeoff ile aynı)
LIMITS = dict(
    alt_ft=(12.0, CMD_MAX_ALT_FT),            # havada başlangıç (AGL, CG)
    speed_kt=(15.0, CMD_MAX_KT),              # ileri uçuşta başlangıç hava hızı
    fuel_total_lbs=(0.0, 2 * TANK_CAPACITY_LBS),
    wind_kt=(0.0, 40.0),
    temp_dc=(-30.0, 40.0),
)
# ajanın eğitildiği aralıklar (dışı çalışır ama uyarı gösterilir; command_viz._limits_flight "trained")
TRAINED = dict(
    hover_alt_ft=(12.0, 1000.0),
    cruise_alt_ft=(CRUISE_MIN_ALT_FT, CRUISE_MAX_ALT_FT),
    cruise_kt=(CRUISE_MIN_KT, CRUISE_MAX_KT),
    fuel_tank_lbs=(150.0, 600.0),
    wind_kt=(0.0, 25.0),
    turb=("none", "light", "moderate"),
    temp_dc=(-10.0, 30.0),
)


@dataclass
class StartConditions:
    mode: str = "ground"                 # ground | hover | cruise
    alt_ft: float = 100.0                # AGL (hover / ileri uçuş)
    speed_kt: float = 60.0               # ileri uçuş
    heading_deg: float = 0.0
    fuel_total_lbs: float = 600.0        # iki tanka eşit
    wind_kt: float = 0.0
    wind_from_deg: float = 0.0           # rüzgârın geldiği yön, kuzeye göre (mutlak)
    turb: str = "none"
    gusts: bool = False
    temp_dc: float = 0.0                 # standart günden sıcaklık farkı (°C)
    location: Location = field(default_factory=lambda: DEFAULT_LOCATION)

    def copy(self, **kw) -> "StartConditions":
        return replace(self, **kw)

    @property
    def fuel_tanks(self) -> tuple[float, float]:
        half = round(self.fuel_total_lbs / 2.0, 1)
        return half, half

    @property
    def wind_dir_relative(self) -> float:
        """Canlı sunucu rüzgâr yönünü başlangıç heading'ine göre bekler."""
        return (self.wind_from_deg - self.heading_deg) % 360.0

    def warnings(self) -> list[str]:
        """Eğitim aralığının dışında kalan seçimler (uçuş yine başlar)."""
        w, T = [], TRAINED
        if self.mode == "hover" and not T["hover_alt_ft"][0] <= self.alt_ft <= T["hover_alt_ft"][1]:
            w.append(f"Hover irtifası eğitimde {T['hover_alt_ft'][0]:.0f}–{T['hover_alt_ft'][1]:.0f} ft.")
        if self.mode == "cruise":
            if not T["cruise_alt_ft"][0] <= self.alt_ft <= T["cruise_alt_ft"][1]:
                w.append(f"İleri uçuş irtifası eğitimde {T['cruise_alt_ft'][0]:.0f}–{T['cruise_alt_ft'][1]:.0f} ft.")
            if not T["cruise_kt"][0] <= self.speed_kt <= T["cruise_kt"][1]:
                w.append(f"İleri uçuş hızı eğitimde {T['cruise_kt'][0]:.0f}–{T['cruise_kt'][1]:.0f} kt.")
        tank = self.fuel_tanks[0]
        if not T["fuel_tank_lbs"][0] <= tank <= T["fuel_tank_lbs"][1]:
            w.append(f"Yakıt eğitimde tank başına {T['fuel_tank_lbs'][0]:.0f}–{T['fuel_tank_lbs'][1]:.0f} lbs "
                     f"(toplam {2 * T['fuel_tank_lbs'][0]:.0f}–{2 * T['fuel_tank_lbs'][1]:.0f}).")
        if self.wind_kt > T["wind_kt"][1]:
            w.append(f"Rüzgâr eğitimde en çok {T['wind_kt'][1]:.0f} kt.")
        if self.turb not in T["turb"]:
            w.append("Şiddetli türbülans eğitimde yoktu.")
        if not T["temp_dc"][0] <= self.temp_dc <= T["temp_dc"][1]:
            w.append(f"Sıcaklık farkı eğitimde {T['temp_dc'][0]:+.0f}…{T['temp_dc'][1]:+.0f} °C.")
        if self.fuel_total_lbs <= 0.0:
            w.append("Tanklar boş: motor ilk saniyede durur (yakıt bitti).")
        return w
