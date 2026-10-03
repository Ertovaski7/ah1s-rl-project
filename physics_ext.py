from __future__ import annotations

"""
PHYSICS EXT — ortak fizik katmanı: tork göstergesi / cezası, yakıt tüketimi, rüzgâr / gust / türbülans
=======================================================================================================

Env'den bağımsız; kalkış (`helicopter_env_takeoff.py`), manevra (`helicopter_env_maneuver.py`) ve tek ajan
(`helicopter_env_flight.py`) env'leri aynı modülü kullanır. Her özellik config ile açılır; **varsayılan: hepsi kapalı**
→ eski modeller (`takeoff_final.zip`, `takeoff_torque.zip`, `maneuver_robust_final.zip`, ...) birebir aynı davranır.
Env ayarı olarak modelin zip'inde taşınır (`ah1s_env_overrides = {"physics": {...}}`, `PhysicsExtConfig.to_dict`).

Kullanım (env içinde):
    ext = PhysicsExt(cfg.physics, control_dt)        # None / {} → pasif (hiçbir yöntem bir şey yapmaz)
    ext.begin_episode(rng, heading_deg, options)     # reset başında bir kez: bölüm parametreleri (rüzgâr, türbülans...)
    ext.attach(fdm)                                  # her yeni FDM'de (reset denemeleri): sabit rüzgâr
    ext.before_step(fdm)                             # her fizik koşusundan önce (reset PID'i dahil): rüzgâr yeniden
                                                     # yazılır (run_ic siler), türbülans / gust
    ext.start_disturbances(fdm)                      # handover: türbülans ve gust'lar başlar, yakıt sayacı başlar
    events = ext.after_step(fdm)                     # bölüm adımından sonra: yakıt, motor, tork istatistiği
    ext.summary()                                    # tepe psi, 50 / 56 psi üstü süre, yakıt, olaylar

1) Tork (psi)
   `ah1s.xml` "bell instruments": psi = 0.00416·Q_ana_rotor − 7.33 (model yazarı ±%20 hata olabilir diyor); el
   kitabı sınırları 50 psi sürekli, 56 psi 30 dk (= %100 tork). `torque_psi`, `psi_to_throttle` ve `torque_penalty`
   `ground-effect-torque` branch'indeki kalkış env'i uygulamasının kendisi (buraya taşındı; env'ler buradan çağırır,
   sonuçlar bit düzeyinde aynı). Gözlem / ceza / güç tavanı env config alanlarıyla (`torque_obs`, `pen_torque_cont`,
   `pen_torque_over`, `pen_rpm_low`, `aircraft="repo"` + `power_cap_psi`). Model psi'si el kitabının hover ve seyir
   grafikleriyle ~1 psi içinde (docs/physics_ext/README.md, probe a).

2) Yakıt
   JSBSim AH-1S motoru `electric_1500hp` → yakıt yakmıyor. Burada her kontrol adımında W_f·dt tanklardan düşülür.
   model = "chart" (varsayılan, kullanıcı seçimi 2026-09-29): W_f(psi, basınç irtifası) = a(h) + b(h)·psi·(N_r/324) —
     AH-1S operatör el kitabı TM 55-1520-234-10 şekil 7-8 sayfa 7 (seyir, T53-L-703, 4 TOW, FAT +15 °C, JP-4, ECU
     kapalı, 324 rotor rpm): tork ve yakıt akışı ölçekleri aynı grafikte karşılıklı; 400 dpi taramadan eksen
     çentikleriyle okundu (`docs/physics_ext/fuel_chart_digitized.csv`). Eşdeğer SFC (1290 shp = 56 psi): 50 psi 0.64,
     30 psi 0.81 lb/shp/h; doğru 1800 shp'ye uzatılınca 0.573 → yayımlanmış kalkış SFC'si 0.568 ile tutarlı.
   model = "sfc": W_f = sfc · P, sfc sabit (0.568 lb/shp/h: T53-L-703 kalkış gücünde, Purdue AAE propulsion
     veritabanı). P = "gauge" (psi → shp: 1290 shp @ 56 psi, aircav) ya da "rotor" (ana + kuyruk rotoru Q·Ω).
   Tank çekimi: el kitabı iki hücre (ön / arka) ve her birinde pompa tarif ediyor; ağırlık-denge grafiği yakıt
     momentini tek bir doğru olarak veriyor (kol ~200–203 in) → iki hücre birlikte boşalıyor. Çekim sırası açıkça
     yazılmıyor → VARSAYIM: iki tanktan eşit ("equal"; biri boşalırsa kalan diğerinden). Tanklar boşalınca motor
     ayrılır: `fcs/rpm-governor-active-norm = 0` (governor gazı 0'a çeker, FGTransmission serbest tekerlek) ve
     "fuel_exhausted" olayı kaydedilir.

3) Rüzgâr / gust / türbülans
   Sabit rüzgâr: `atmosphere/wind-{north,east}-fps` (yön rüzgârın GELDİĞİ yön; mutlak ya da başlangıç heading'ine
   göre). JSBSim `run_ic()` rüzgârı IC rüzgârıyla (0) eziyor (FGFDMExec::Initialize → Winds->SetWindNED) → her fizik
   koşusundan önce yeniden yazılır. İsteğe bağlı yer yakını kesme (MIL-F-8785C log profili, W20 = 20 ft'teki rüzgâr).
   Gust: JSBSim 1−cos gust'ı (`atmosphere/cosine-gust/*`, yerel NED), Poisson zamanlı ya da sabit liste.
   Türbülans:
     backend "dryden_agl" (varsayılan; kullanıcı seçimi: yok → hafif → orta): MIL-F-8785C alçak irtifa Dryden,
       yükseklik = AGL (CG), σ_w = 0.1·W20, L_w = h, σ_u = σ_v = σ_w/(0.177 + 0.000823h)^0.4,
       L_u = L_v = h/(0.177 + 0.000823h)^1.2; bileşenler ortalama rüzgâr eksenlerinde; kesin (üstel) ayrıklaştırma,
       kontrol adımında; `atmosphere/gust-{north,east,down}-fps`'e yazılır (öteleme; açısal türbülans yok). Donmuş
       alan hızı V = max(hava hızı, v_min).
     backend "jsbsim_milspec" / "jsbsim_tustin": JSBSim'in kendi modeli (turb-type 3 / 4). BU UÇAKTA KULLANILAMAZ
       (probe c): yükseklik MSL (Edwards'ta hep orta irtifa dalı, W20 etkisiz), açısal türbülans 10.75 ft kanat
       açıklığıyla ölçekleniyor → hafifte bile açık döngüde 6–900 °/s; sakin hover'da NaN. Yalnızca karşılaştırma için.
   MIL-F-8785C şiddet referansı (alçak irtifa): hafif W20 = 15 kt, orta 30 kt, şiddetli 45 kt.
"""

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

# =====================================================================================================================
# Sabitler (kaynaklar modül başlığında ve docs/physics_ext/README.md'de)
# =====================================================================================================================
KT_TO_FPS = 1.6878099
PSI_CONT = 50.0
PSI_LIMIT = 56.0
NOMINAL_RPM = 324.0
ELECTRIC_HP = 1500.0                  # Engines/electric_1500hp.xml (T53 yerine)
SHP_AT_100 = 1290.0                   # aircav.com T53-L-703: "limited to 1290 shp at 100% torque"
PSI_AT_100 = 56.0                     # aircav.com: 35 psi = %62.5 → %100 = 56 psi
T53_703_TO_SFC = 0.568                # lb/shp/h @ 1800 shp kalkış (Purdue AAE propulsion DB, T53 sayfası)

# TM 55-1520-234-10 şekil 7-8 sayfa 7: yakıt akışı W_f = a + b·psi (lb/h), basınç irtifasına göre (ft)
FUEL_CHART_ALT_FT = (0.0, 2000.0, 4000.0, 6000.0)
FUEL_CHART_A = (292.6, 273.9, 251.7, 238.3)
FUEL_CHART_B = (9.446, 9.416, 9.501, 9.376)

TANK_X_IN = (146.0, 206.0)            # ah1s.xml: tank 0 önde, tank 1 arkada (z = 38 in), her biri 890 lbs
TANK_CAPACITY_LBS = 890.0

# MIL-F-8785C alçak irtifa türbülans şiddeti → W20 (20 ft'te rüzgâr hızı, kt)
TURB_W20_KT = {"none": 0.0, "light": 15.0, "moderate": 30.0, "severe": 45.0}
# JSBSim milspec/tustin şiddet indeksi (olasılık eğrisi, MIL-F-8785C şekil 7): hafif 3, orta 4, şiddetli 6
TURB_JSBSIM_SEVERITY = {"none": 0, "light": 3, "moderate": 4, "severe": 6}


# =====================================================================================================================
# TORK (ground-effect-torque kalkış env'inden taşındı — aynı ifadeler, aynı sonuçlar)
# =====================================================================================================================

def torque_psi(q_lbsft):
    """Modelin kendi tork göstergesi (ah1s.xml 'bell instruments'): el kitabı sürekli 50 psi, %100 (30 dk) 56 psi."""
    return 0.00416 * q_lbsft - 7.33


def psi_to_throttle(psi: float) -> float:
    """Bu tork (nominal devirde) kadar güç → governor gaz tavanı (elektrik motoru: güç = gaz · 1500 hp)."""
    q = (psi + 7.33) / 0.00416
    return q * (NOMINAL_RPM * 2.0 * math.pi / 60.0) / 550.0 / ELECTRIC_HP


def torque_penalty(psi: float, rpm: float, pen_cont: float = 0.0, pen_over: float = 0.0, pen_rpm_low: float = 0.0,
                   cont_psi: float = PSI_CONT, max_psi: float = PSI_LIMIT, rpm_low: float = 314.0) -> float:
    """50–56 psi: pen_cont·((psi − 50)/6)² (kalkış gücü bölgesi, hafif), 56 üstü: pen_over·min(((psi − 56)/3)², 9),
    rotor devri rpm_low altında: pen_rpm_low·((rpm_low − rpm)/10)². (kalkış env'i 2026-09-28 ile birebir.)"""
    torque = 0.0
    if pen_cont > 0.0 or pen_over > 0.0:
        span = max(1e-6, max_psi - cont_psi)
        torque = (pen_cont * min(1.0, max(0.0, psi - cont_psi) / span) ** 2
                  + pen_over * min(9.0, (max(0.0, psi - max_psi) / 3.0) ** 2))
    if pen_rpm_low > 0.0 and rpm < rpm_low:
        torque += pen_rpm_low * ((rpm_low - rpm) / 10.0) ** 2
    return torque


def read_torque_psi(fdm) -> float:
    return float(torque_psi(float(fdm["propulsion/engine/torque-lbsft"])))


# =====================================================================================================================
# CONFIG
# =====================================================================================================================

@dataclass
class FuelExtConfig:
    enable: bool = False
    model: str = "chart"                  # "chart" (TM 55-1520-234-10) | "sfc" (sabit SFC × güç)
    sfc: float = T53_703_TO_SFC           # model="sfc"
    power_source: str = "gauge"           # model="sfc": "gauge" (psi → shp) | "rotor" (Q·Ω ana + kuyruk)
    shp_at_100: float = SHP_AT_100
    psi_at_100: float = PSI_AT_100
    draw: str = "equal"                   # "equal" | "proportional" | "fwd_first" | "aft_first"
    unusable_lbs: float = 0.0             # toplam bunun altına inince motor ayrılır
    engine_out_on_empty: bool = True
    burn_scale: float = 1.0               # 1 = gerçek; >1 yalnızca test / hızlandırılmış senaryo için


@dataclass
class WindExtConfig:
    enable: bool = False
    speed_kt: tuple = (0.0, 0.0)          # bölüm başında düzgün dağılımdan (kesme açıksa W20)
    p_calm: float = 0.0                   # bu olasılıkla rüzgâr 0 (curriculum: rüzgârlı / rüzgârsız karışım)
    dir_deg: tuple = (0.0, 360.0)         # rüzgârın GELDİĞİ yön
    dir_relative: bool = True             # True: başlangıç heading'ine göre (0 = karşıdan, 90 = sağdan)
    shear: bool = False                   # MIL-F-8785C: W(h) = W20·ln(h/z0)/ln(20/z0), h ≤ 1000 ft (CG AGL)
    z0_ft: float = 0.15
    shear_min_frac: float = 0.3           # yerde (h → z0) sıfıra inmesin


@dataclass
class GustExtConfig:
    enable: bool = False
    rate_per_min: float = 0.0             # Poisson; 0 → yalnızca `schedule`
    magnitude_kt: tuple = (5.0, 15.0)
    rise_s: tuple = (1.0, 3.0)
    hold_s: tuple = (0.0, 2.0)
    decay_s: tuple = (1.0, 3.0)
    vertical_frac: tuple = (0.0, 0.0)     # dikey bileşen / toplam (işaret rastgele)
    along_wind_prob: float = 0.5          # ortalama rüzgâr yönünde (diğerleri rastgele yatay yön)
    first_after_s: float = 5.0
    schedule: tuple = ()                  # sabit gust listesi: ({"t": s, "mag_kt":, "dir_deg": (gidiş, mutlak),
                                          #   "vert": frac, "rise":, "hold":, "decay":}, ...)


@dataclass
class TurbExtConfig:
    enable: bool = False
    backend: str = "dryden_agl"           # "dryden_agl" | "jsbsim_milspec" | "jsbsim_tustin"
    levels: tuple = ("light",)            # bölüm başında seçilir (none / light / moderate / severe)
    level_probs: tuple = ()               # boş → eşit olasılık
    scale: float = 1.0                    # dryden_agl: σ çarpanı (curriculum için)
    v_min_fps: float = 15.0               # dryden_agl: donmuş alan hızı alt sınırı (sakin hover)
    h_min_ft: float = 10.0                # MIL / JSBSim alt sınırı
    seed_jsbsim: bool = True              # jsbsim_*: atmosphere/randomseed bölüm RNG'sinden


@dataclass
class AtmoExtConfig:
    """Hava sıcaklığı (2026-10-01): JSBSim `atmosphere/delta-T` — standart günden sapma. Edwards'ta (pist 2283 ft MSL)
    +15 °C → yoğunluk irtifası ~3900 ft, +30 °C → ~5400 ft. Yalnızca aerodinamik (yoğunluk); elektrik motorunun gücü
    sıcaklık / irtifayla düşmüyor (T53 güç kaybı modellenmedi)."""
    enable: bool = False
    delta_T_C: tuple = (0.0, 0.0)         # bölüm başında düzgün dağılımdan (°C)


@dataclass
class PhysicsExtConfig:
    fuel: FuelExtConfig = field(default_factory=FuelExtConfig)
    wind: WindExtConfig = field(default_factory=WindExtConfig)
    gust: GustExtConfig = field(default_factory=GustExtConfig)
    turb: TurbExtConfig = field(default_factory=TurbExtConfig)
    atmo: AtmoExtConfig = field(default_factory=AtmoExtConfig)

    @property
    def active(self) -> bool:
        return any(getattr(self, f.name).enable for f in fields(self))

    def to_dict(self, only_changes: bool = True) -> dict:
        """JSON'a uygun dict (model zip'indeki ah1s_env_overrides["physics"] için); varsayılandan farklı olanlar."""
        out = {}
        for f in fields(self):
            sub = getattr(self, f.name)
            d = asdict(sub)
            if only_changes:
                ref = asdict(type(sub)())
                d = {k: v for k, v in d.items() if v != ref[k]}
            if d:
                out[f.name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
        return out

    @classmethod
    def from_dict(cls, d: dict | None) -> "PhysicsExtConfig":
        cfg = cls()
        for name, sub in (d or {}).items():
            if name not in {f.name for f in fields(cls)}:
                raise ValueError(f"bilinmeyen physics bölümü: {name}")
            obj = getattr(cfg, name)
            known = {f.name for f in fields(obj)}
            for k, v in (sub or {}).items():
                if k not in known:
                    raise ValueError(f"bilinmeyen physics.{name} ayarı: {k}")
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)
        return cfg


def as_physics_config(x) -> PhysicsExtConfig:
    if x is None:
        return PhysicsExtConfig()
    if isinstance(x, PhysicsExtConfig):
        return x
    if isinstance(x, dict):
        return PhysicsExtConfig.from_dict(x)
    raise TypeError(f"physics config: {type(x)}")


# =====================================================================================================================
# SAF FONKSİYONLAR
# =====================================================================================================================

def fuel_flow_chart_lbh(psi: float, rpm: float, pressure_alt_ft: float) -> float:
    """TM 55-1520-234-10 şekil 7-8 (sayfa 7) doğruları; psi, rotor devriyle güce eşdeğerlenir (psi·N/324)."""
    h = float(np.clip(pressure_alt_ft, FUEL_CHART_ALT_FT[0], FUEL_CHART_ALT_FT[-1]))
    a = float(np.interp(h, FUEL_CHART_ALT_FT, FUEL_CHART_A))
    b = float(np.interp(h, FUEL_CHART_ALT_FT, FUEL_CHART_B))
    return a + b * max(0.0, psi) * max(0.0, rpm) / NOMINAL_RPM


def shaft_power_gauge_shp(psi: float, rpm: float, shp_at_100: float = SHP_AT_100, psi_at_100: float = PSI_AT_100):
    return max(0.0, psi) * shp_at_100 / psi_at_100 * max(0.0, rpm) / NOMINAL_RPM


def shaft_power_rotor_hp(fdm) -> float:
    """Modelin fiziksel gücü: ana + kuyruk rotoru aerodinamik torku × açısal hız (hp)."""
    p = 0.0
    for i in (0, 1):
        q = float(fdm[f"propulsion/engine[{i}]/torque-lbsft"])
        w = float(fdm[f"propulsion/engine[{i}]/rotor-rpm"]) * 2.0 * math.pi / 60.0
        p += q * w / 550.0
    return max(0.0, p)


def wind_ned_fps(speed_fps: float, from_deg: float) -> tuple[float, float]:
    """Meteorolojik yön (rüzgârın geldiği yön) → hava kütlesinin hızı (kuzey, doğu), ft/s."""
    a = math.radians(from_deg)
    return -speed_fps * math.cos(a), -speed_fps * math.sin(a)


def shear_factor(h_ft: float, z0_ft: float = 0.15, min_frac: float = 0.3) -> float:
    """MIL-F-8785C: W(h)/W20 = ln(h/z0)/ln(20/z0), 1000 ft üstü sabit."""
    h = float(np.clip(h_ft, z0_ft * 1.01, 1000.0))
    return max(min_frac, math.log(h / z0_ft) / math.log(20.0 / z0_ft))


def dryden_low_alt_params(h_ft: float, w20_fps: float) -> dict:
    """MIL-F-8785C alçak irtifa (h < 1000 ft) ölçek uzunlukları (ft) ve şiddetleri (ft/s)."""
    h = float(np.clip(h_ft, 10.0, 1000.0))
    k = 0.177 + 0.000823 * h
    sig_w = 0.1 * w20_fps
    return dict(L_u=h / k ** 1.2, L_v=h / k ** 1.2, L_w=h, sig_u=sig_w / k ** 0.4, sig_v=sig_w / k ** 0.4, sig_w=sig_w)


def draw_from_tanks(contents, amount: float, mode: str = "equal") -> np.ndarray:
    """contents (lbs, tank 0 ön / tank 1 arka) − amount; boşalan tanktan eksik kalan diğerinden."""
    c = np.maximum(np.asarray(contents, dtype=np.float64), 0.0)
    amount = min(max(0.0, float(amount)), float(c.sum()))
    if amount <= 0.0:
        return c
    if mode == "proportional":
        return c - amount * c / c.sum()
    if mode in ("fwd_first", "aft_first"):
        order = (0, 1) if mode == "fwd_first" else (1, 0)
        rest = amount
        for i in order:
            d = min(c[i], rest)
            c[i] -= d
            rest -= d
        return c
    if mode != "equal":
        raise ValueError(f"bilinmeyen tank çekim sırası: {mode}")
    rest = amount
    for _ in range(2):                                 # eşit pay; biri yetmezse kalan diğerinden
        live = [i for i in (0, 1) if c[i] > 1e-12]
        if not live or rest <= 1e-12:
            break
        share = rest / len(live)
        for i in live:
            d = min(c[i], share)
            c[i] -= d
            rest -= d
    return np.maximum(c, 0.0)


def air_ground_velocities(fdm) -> dict:
    """Gövde ekseninde hava hızı (u, v, w: hava kütlesine göre) ve yer hızı (ileri, yana), ft/s."""
    psi = float(fdm["attitude/psi-rad"])
    vn, ve = float(fdm["velocities/v-north-fps"]), float(fdm["velocities/v-east-fps"])
    return dict(u_air=float(fdm["velocities/u-aero-fps"]), v_air=float(fdm["velocities/v-aero-fps"]),
                w_air=float(fdm["velocities/w-aero-fps"]),
                u_gnd=vn * math.cos(psi) + ve * math.sin(psi), v_gnd=-vn * math.sin(psi) + ve * math.cos(psi))


def fuel_total(fdm) -> float:
    return float(fdm["propulsion/tank[0]/contents-lbs"]) + float(fdm["propulsion/tank[1]/contents-lbs"])


# =====================================================================================================================
# DURUMLU KATMAN
# =====================================================================================================================

class PhysicsExt:
    """Bir env örneğine bağlı fizik katmanı. Varsayılan config (ya da begin_episode çağrılmadan) → hiçbir şey yapmaz,
    env'in rastgele sayı akışına da dokunmaz (eski modeller birebir)."""

    def __init__(self, cfg=None, control_dt: float = 0.075):
        self.cfg = as_physics_config(cfg)
        self.dt = float(control_dt)
        self.rng = np.random.default_rng(0)
        self.t = 0.0
        self.params: dict = {}
        self.events: list[dict] = []
        self._armed = False                          # begin_episode ile (config açıksa ya da options verildiyse)
        self._touched = False                        # FDM'e rüzgâr / gust / türbülans yazıldı mı (FDM yeniden kullanılırsa
                                                     # sonraki bölümde temizlenir: run_ic gust / türbülansı SIFIRLAMIYOR)
        self._jsbsim_turb = False
        self._dist_on = False                        # start_disturbances ile (türbülans, gust, yakıt, istatistik)
        self._turb_on = False
        self._xi = np.zeros(3)
        self._gust_next_t = np.inf
        self._gust_end_t = -np.inf                   # çalışan 1−cos gust'ın bitişi (üst üste binmeyi önlemek için)
        self._gust_queue: list[dict] = []
        self._engine_out = False
        self._wind_ned = (0.0, 0.0)
        self._wind_target = None                     # canlı rüzgâr değişikliği: (kuzey, doğu) ft/s hedefi (set_live)
        self.stats = self._new_stats()
        self._last = dict(psi=0.0, rpm=NOMINAL_RPM, fuel_flow_lbh=0.0, shaft_shp=0.0)

    @property
    def active(self) -> bool:
        return self.cfg.active

    @property
    def armed(self) -> bool:
        return self._armed

    @staticmethod
    def _new_stats() -> dict:
        return dict(t=0.0, peak_psi=-np.inf, t_over_cont=0.0, t_over_limit=0.0, min_rpm=np.inf, fuel_used_lbs=0.0,
                    fuel0_lbs=None, engine_out_t=None, gusts=0)

    # -----------------------------------------------------------------------------------------------------------
    # bölüm başı
    # -----------------------------------------------------------------------------------------------------------
    def begin_episode(self, rng, heading_deg: float = 0.0, options: dict | None = None) -> dict:
        """Reset başında bir kez: bölüm parametrelerini seçer. rng: env'in np_random'u — YALNIZCA katman açıksa bir
        tohum çekilir (kapalıyken env'in rastgele akışı değişmez). options (değerlendirme / canlı; config'i ezer):
        {"wind_kt":, "wind_dir_deg":, "wind_dir_relative":, "wind_shear":, "turb_level":, "turb_backend":, "turb_w20_kt":,
        "turb_from_wind": (True | çarpan), "delta_T_C":,
        "gusts": [...]}."""
        cfg, o = self.cfg, dict(options or {})
        self.t = 0.0
        self.events, self._engine_out, self._turb_on, self._dist_on = [], False, False, False
        self._xi = np.zeros(3)
        self._gust_next_t = np.inf
        self._gust_end_t = -np.inf
        self._gust_queue = []
        self.stats = self._new_stats()
        self._wind_ned = (0.0, 0.0)
        self._wind_target = None
        self.params = {}
        self._armed = cfg.active or bool(o)
        if not self._armed:
            return {}
        self.rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
        p = {}
        if cfg.wind.enable or "wind_kt" in o:
            w = cfg.wind
            spd = float(self.rng.uniform(*w.speed_kt))
            if w.p_calm > 0.0 and self.rng.random() < w.p_calm:
                spd = 0.0
            d = float(self.rng.uniform(*w.dir_deg))
            spd = float(o.get("wind_kt", spd))
            d = float(o.get("wind_dir_deg", d))
            rel = bool(o.get("wind_dir_relative", w.dir_relative))
            d_abs = (heading_deg + d) % 360.0 if rel else d % 360.0
            p.update(wind_kt=spd, wind_dir_deg=d, wind_dir_relative=rel, wind_from_deg=d_abs)
            self._wind_ned = wind_ned_fps(spd * KT_TO_FPS, d_abs)
            if "wind_shear" in o:                                  # 2026-10-01: seçenekle yer yakını kesme (W20 = 20 ft)
                p["wind_shear"] = bool(o["wind_shear"])
        if cfg.turb.enable or "turb_level" in o:
            tb = cfg.turb
            probs = np.asarray(tb.level_probs, dtype=np.float64) if tb.level_probs else None
            lvl = tb.levels[int(self.rng.choice(len(tb.levels), p=probs / probs.sum() if probs is not None else None))]
            lvl = o.get("turb_level", lvl)
            w20 = float(o.get("turb_w20_kt", TURB_W20_KT[lvl]))
            if o.get("turb_from_wind") and "wind_kt" in p and lvl != "none":
                # 2026-10-01 seçeneği: şiddet ortalama rüzgârdan (MIL-F-8785C: σ_w = 0.1·W20; W20 = 20 ft'teki rüzgâr) —
                # seviyenin W20'si alt sınır değil, rüzgâr × çarpan (turb_from_wind sayısı; True → 1)
                w20 = float(p["wind_kt"]) * (1.0 if o["turb_from_wind"] is True else float(o["turb_from_wind"]))
            p.update(turb_level=lvl, turb_backend=o.get("turb_backend", tb.backend), turb_w20_kt=w20)
        if cfg.gust.enable or "gusts" in o or "gust_rate_per_min" in o:
            self._gust_queue = sorted((dict(g) for g in (o.get("gusts", cfg.gust.schedule) or ())),
                                      key=lambda g: float(g["t"]))
            p["gust_rate_per_min"] = float(o.get("gust_rate_per_min", cfg.gust.rate_per_min if cfg.gust.enable else 0.0))
            p["gust_kt"] = tuple(o.get("gust_kt", cfg.gust.magnitude_kt))
        if "delta_T_C" in o:                                      # sıcaklık (yalnızca istenince; yoksa RNG'ye dokunmaz)
            p["delta_T_C"] = float(o["delta_T_C"])
        elif cfg.atmo.enable:
            p["delta_T_C"] = float(self.rng.uniform(*cfg.atmo.delta_T_C))
        self.params = p
        return dict(p)

    def attach(self, fdm):
        """Her reset denemesinde (yeni ya da yeniden kullanılan FDM): önceki bölümün rüzgâr / gust / türbülansını temizler
        (run_ic yalnızca sabit rüzgârı IC'den yeniden kurar; gust, 1−cos gust ve türbülans FDM'de kalır), sonra bu
        bölümün sabit rüzgârını yazar (reset PID'i rüzgârda otursun). Hiç dokunulmamışsa ve kapalıysa hiçbir şey yazmaz."""
        if self._touched:
            self._clear(fdm)
        if self._armed:
            self.apply_wind(fdm)
            if self.params.get("delta_T_C"):                       # run_ic delta-T'yi silmiyor (ölçüldü) → bir kez
                fdm["atmosphere/delta-T"] = 1.8 * float(self.params["delta_T_C"])   # °C → °R
                self._touched = True

    def _clear(self, fdm):
        fdm["atmosphere/delta-T"] = 0.0
        for k in ("wind", "gust"):
            for a in ("north", "east", "down"):
                fdm[f"atmosphere/{k}-{a}-fps"] = 0.0
        if self._jsbsim_turb:
            fdm["atmosphere/turb-type"] = 0
            self._jsbsim_turb = False
        # çalışan 1−cos gust'ı sıfır genlikli kısa bir gust'la bitir (start=0 son değeri dondurur)
        fdm["atmosphere/cosine-gust/magnitude-ft_sec"] = 0.0
        fdm["atmosphere/cosine-gust/startup-duration-sec"] = 0.01
        fdm["atmosphere/cosine-gust/steady-duration-sec"] = 0.0
        fdm["atmosphere/cosine-gust/end-duration-sec"] = 0.01
        fdm["atmosphere/cosine-gust/start"] = 1
        self._touched = False

    def apply_wind(self, fdm, h_agl_ft: float | None = None):
        vn, ve = self._wind_ned
        if self.params.get("wind_shear", self.cfg.wind.shear) and (vn or ve):
            h = float(fdm["position/h-agl-ft"]) if h_agl_ft is None else h_agl_ft
            k = shear_factor(h, self.cfg.wind.z0_ft, self.cfg.wind.shear_min_frac)
            vn, ve = vn * k, ve * k
        if "wind_kt" in self.params:
            fdm["atmosphere/wind-north-fps"] = vn
            fdm["atmosphere/wind-east-fps"] = ve
            fdm["atmosphere/wind-down-fps"] = 0.0
            self._touched = True

    def start_disturbances(self, fdm):
        """Handover'da: türbülans + gust zamanlaması başlar; yakıt ve tork istatistiği bölüm saatini başlatır."""
        if not self._armed:
            return
        self._dist_on = True
        self.t = 0.0
        self.stats["fuel0_lbs"] = fuel_total(fdm)
        p = self.params
        if "turb_level" in p and p["turb_level"] != "none":
            self._turb_on = True
            be = p["turb_backend"]
            if be.startswith("jsbsim"):
                self._touched = self._jsbsim_turb = True
                fdm["atmosphere/turb-type"] = 3 if be == "jsbsim_milspec" else 4
                fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = p["turb_w20_kt"] * KT_TO_FPS
                fdm["atmosphere/turbulence/milspec/severity"] = TURB_JSBSIM_SEVERITY[p["turb_level"]]
                if self.cfg.turb.seed_jsbsim:
                    fdm["atmosphere/randomseed"] = int(self.rng.integers(1, 2 ** 31 - 1))
            elif be != "dryden_agl":
                raise ValueError(f"bilinmeyen türbülans backend'i: {be}")
        rate = float(p.get("gust_rate_per_min", 0.0))
        if rate > 0.0:
            self._gust_next_t = self.cfg.gust.first_after_s + float(self.rng.exponential(60.0 / rate))

    # -----------------------------------------------------------------------------------------------------------
    # adım
    # -----------------------------------------------------------------------------------------------------------
    def before_step(self, fdm):
        """Her fizik koşusundan (kontrol adımı) önce — reset PID'leri dahil. Sabit rüzgâr HER adımda yeniden yazılır
        (env'ler teleport / bozucu için run_ic çağırıyor, run_ic rüzgârı siliyor)."""
        if not self._armed:
            return
        if self._wind_target is not None:                          # canlı rüzgâr değişikliği: hedefe rampa
            self._ramp_wind()
        self.apply_wind(fdm)
        if not self._dist_on:
            return
        if self._turb_on and self.params.get("turb_backend") == "dryden_agl":
            self._dryden_step(fdm)
        self._gust_step(fdm)

    def _dryden_step(self, fdm):
        tb, p = self.cfg.turb, self.params
        h = max(tb.h_min_ft, float(fdm["position/h-agl-ft"]))
        w20 = p["turb_w20_kt"] * KT_TO_FPS * tb.scale
        q = dryden_low_alt_params(h, w20)
        V = max(tb.v_min_fps, float(fdm["velocities/vt-fps"]))
        # MIL-HDBK-1797 / Yeager: u ~ τ_u = L_u/V, v ~ τ_u/2, w ~ τ_w/2 (birinci derece yaklaşık), kesin ayrıklaştırma
        taus = np.array([q["L_u"] / V, 0.5 * q["L_v"] / V, 0.5 * q["L_w"] / V])
        sig = np.array([q["sig_u"], q["sig_v"], q["sig_w"]])
        a = np.exp(-self.dt / taus)
        self._xi = a * self._xi + sig * np.sqrt(1.0 - a * a) * self.rng.standard_normal(3)
        vn, ve = self._wind_ned
        psiw = math.atan2(ve, vn) if (vn or ve) else math.radians(float(fdm["attitude/psi-deg"]))
        c, s = math.cos(psiw), math.sin(psiw)
        u, v, w = self._xi
        fdm["atmosphere/gust-north-fps"] = c * u - s * v
        fdm["atmosphere/gust-east-fps"] = s * u + c * v
        fdm["atmosphere/gust-down-fps"] = w
        self._touched = True

    def _gust_step(self, fdm):
        g = self.cfg.gust
        due = None
        if self._gust_queue and self.t >= float(self._gust_queue[0]["t"]) - 1e-9:
            due = self._gust_queue.pop(0)
        elif self.t >= self._gust_next_t:
            due = self._random_gust()
            self._gust_next_t = self.t + float(self.rng.exponential(60.0 / float(self.params["gust_rate_per_min"])))
        if due is None:
            return
        if self.t < self._gust_end_t:
            # JSBSim'in 1−cos gust'ı sürerken yenisi başlatılırsa yönü güncellemiyor, genliği basamakla değiştiriyor ve
            # süreyi yeniden başlatmıyor (2026-09-30 ölçümü) → yeni gust öncekinin bitişine ertelenir (kaybolmaz)
            self._gust_queue.append(dict(due, t=self._gust_end_t + 0.1))
            self._gust_queue.sort(key=lambda x: float(x["t"]))
            return
        mag = float(due["mag_kt"]) * KT_TO_FPS
        d = math.radians(float(due["dir_deg"]))                   # gust'ın GİTTİĞİ yön (mutlak)
        vf = float(due.get("vert", 0.0))
        hvec = math.sqrt(max(0.0, 1.0 - vf * vf))
        self._gust_end_t = self.t + float(due.get("rise", 2.0)) + float(due.get("hold", 1.0)) + float(due.get("decay", 2.0))
        fdm["atmosphere/cosine-gust/startup-duration-sec"] = float(due.get("rise", 2.0))
        fdm["atmosphere/cosine-gust/steady-duration-sec"] = float(due.get("hold", 1.0))
        fdm["atmosphere/cosine-gust/end-duration-sec"] = float(due.get("decay", 2.0))
        fdm["atmosphere/cosine-gust/magnitude-ft_sec"] = mag
        fdm["atmosphere/cosine-gust/frame"] = 3                    # yerel NED
        fdm["atmosphere/cosine-gust/X-velocity-ft_sec"] = hvec * math.cos(d)
        fdm["atmosphere/cosine-gust/Y-velocity-ft_sec"] = hvec * math.sin(d)
        fdm["atmosphere/cosine-gust/Z-velocity-ft_sec"] = vf
        fdm["atmosphere/cosine-gust/start"] = 1
        self._touched = True
        self.stats["gusts"] += 1
        self.events.append(dict(t=self.t, kind="gust", **{k: v for k, v in due.items() if k != "t"}))

    def _random_gust(self) -> dict:
        g, rng = self.cfg.gust, self.rng
        vn, ve = self._wind_ned
        if (vn or ve) and rng.random() < g.along_wind_prob:
            d = math.degrees(math.atan2(ve, vn))
        else:
            d = float(rng.uniform(0.0, 360.0))
        vf = float(rng.uniform(*g.vertical_frac)) * (1.0 if rng.random() < 0.5 else -1.0)
        return dict(t=self.t, mag_kt=float(rng.uniform(*self.params.get("gust_kt", g.magnitude_kt))), dir_deg=d % 360.0,
                    vert=vf,
                    rise=float(rng.uniform(*g.rise_s)), hold=float(rng.uniform(*g.hold_s)),
                    decay=float(rng.uniform(*g.decay_s)))

    # -----------------------------------------------------------------------------------------------------------
    # canlı uygulama: uçuş sırasında hava değişikliği (2026-10-03)
    # -----------------------------------------------------------------------------------------------------------
    LIVE_WIND_RATE_KT_S = 4.0                    # rüzgâr değişikliği bu hızla rampalanır (basamak gerçekçi değil)

    def _ramp_wind(self):
        (vn, ve), (tn, te) = self._wind_ned, self._wind_target
        dn, de = tn - vn, te - ve
        d = math.hypot(dn, de)
        step = self.LIVE_WIND_RATE_KT_S * KT_TO_FPS * self.dt
        if d <= step:
            self._wind_ned, self._wind_target = (tn, te), None
        else:
            self._wind_ned = (vn + dn * step / d, ve + de * step / d)

    def set_live(self, fdm, wind_kt=None, wind_from_deg=None, turb_level=None, gusts=None, gust_now=None) -> dict:
        """Uçuş sırasında rüzgâr / türbülans / gust değiştir (canlı uygulama; eğitim ve değerlendirme kullanmaz).
        wind_from_deg: rüzgârın GELDİĞİ mutlak yön (°, kuzeyden). Rüzgâr yeni değerine LIVE_WIND_RATE_KT_S hızla rampalanır.
        turb_level: none / light / moderate / severe. gusts: True → dakikada ~1, 5–12 kt rastgele gust; False → kapalı.
        gust_now: {"mag_kt":, "dir_deg": (gust'ın GİTTİĞİ mutlak yön), "rise":, "hold":, "decay":} → hemen bir gust.
        Dönen: güncel parametreler."""
        self._armed = True
        p = self.params
        if wind_kt is not None or wind_from_deg is not None:
            spd = float(p.get("wind_kt", 0.0) if wind_kt is None else wind_kt)
            d_abs = float(p.get("wind_from_deg", 0.0) if wind_from_deg is None else wind_from_deg) % 360.0
            p.update(wind_kt=spd, wind_from_deg=d_abs, wind_dir_deg=d_abs, wind_dir_relative=False)
            self._wind_target = wind_ned_fps(spd * KT_TO_FPS, d_abs)
        if turb_level is not None:
            lvl = str(turb_level)
            if lvl not in TURB_W20_KT:
                raise ValueError(f"bilinmeyen türbülans seviyesi: {lvl}")
            be = p.get("turb_backend", self.cfg.turb.backend)
            p.update(turb_level=lvl, turb_backend=be, turb_w20_kt=TURB_W20_KT[lvl])
            on = lvl != "none"
            if be.startswith("jsbsim"):
                self._jsbsim_turb = self._touched = True
                fdm["atmosphere/turb-type"] = (3 if be == "jsbsim_milspec" else 4) if on else 0
                fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = p["turb_w20_kt"] * KT_TO_FPS
                fdm["atmosphere/turbulence/milspec/severity"] = TURB_JSBSIM_SEVERITY[lvl]
            elif not on:                                          # dryden: süzgeç durumu ve çıkış sıfırlanır
                self._xi = np.zeros(3)
                for a in ("north", "east", "down"):
                    fdm[f"atmosphere/gust-{a}-fps"] = 0.0
            self._turb_on = on and self._dist_on
        if gusts is not None:
            if gusts:
                p["gust_rate_per_min"] = 1.0
                p["gust_kt"] = tuple(p.get("gust_kt") or (5.0, 12.0))
                self._gust_next_t = self.t + float(self.rng.exponential(60.0))
            else:
                p["gust_rate_per_min"] = 0.0
                self._gust_next_t = np.inf
        if gust_now:
            g = dict(gust_now)
            g.setdefault("rise", 1.5)
            g.setdefault("hold", 1.0)
            g.setdefault("decay", 2.0)
            g["t"] = self.t
            self._gust_queue.insert(0, g)
            p.setdefault("gust_rate_per_min", 0.0)
            p.setdefault("gust_kt", (5.0, 12.0))
        return dict(p)

    def after_step(self, fdm, dt: float | None = None) -> list[dict]:
        """Bölüm adımından sonra: tork istatistiği, yakıt yakma, yakıt bitince motor ayrılması. Yeni olayları döndürür."""
        if not (self._armed and self._dist_on):
            return []
        dt = self.dt if dt is None else float(dt)
        n0 = len(self.events)
        self.t += dt
        psi, rpm = read_torque_psi(fdm), float(fdm["propulsion/engine/rotor-rpm"])
        st = self.stats
        st["t"] = self.t
        st["peak_psi"] = max(st["peak_psi"], psi)
        st["min_rpm"] = min(st["min_rpm"], rpm)
        if psi > PSI_CONT:
            st["t_over_cont"] += dt
        if psi > PSI_LIMIT:
            st["t_over_limit"] += dt
        self._last.update(psi=psi, rpm=rpm)
        if self.cfg.fuel.enable:
            self._burn(fdm, psi, rpm, dt)
        return self.events[n0:]

    def _burn(self, fdm, psi: float, rpm: float, dt: float):
        fc = self.cfg.fuel
        if self._engine_out:
            self._last.update(fuel_flow_lbh=0.0, shaft_shp=0.0)
            return
        if fc.model == "chart":
            ff = fuel_flow_chart_lbh(psi, rpm, float(fdm["atmosphere/pressure-altitude"]))
            shp = shaft_power_gauge_shp(psi, rpm, fc.shp_at_100, fc.psi_at_100)
        elif fc.model == "sfc":
            shp = (shaft_power_gauge_shp(psi, rpm, fc.shp_at_100, fc.psi_at_100) if fc.power_source == "gauge"
                   else shaft_power_rotor_hp(fdm))
            ff = fc.sfc * shp
        else:
            raise ValueError(f"bilinmeyen yakıt modeli: {fc.model}")
        if float(fdm["fcs/rpm-governor-active-norm"]) <= 0.0:      # motor kapalı / boşta sayılmaz
            ff = 0.0
        burn = ff * fc.burn_scale * dt / 3600.0
        c = np.array([float(fdm["propulsion/tank[0]/contents-lbs"]), float(fdm["propulsion/tank[1]/contents-lbs"])])
        usable = max(0.0, c.sum() - fc.unusable_lbs)
        burn = min(burn, usable)
        c2 = draw_from_tanks(c, burn, fc.draw)
        fdm["propulsion/tank[0]/contents-lbs"] = float(c2[0])
        fdm["propulsion/tank[1]/contents-lbs"] = float(c2[1])
        self.stats["fuel_used_lbs"] += burn
        self._last.update(fuel_flow_lbh=ff, shaft_shp=shp)
        if fc.engine_out_on_empty and c2.sum() - fc.unusable_lbs <= 1e-9 and ff > 0.0:
            self.engine_out(fdm, reason="fuel_exhausted")

    def engine_out(self, fdm, reason: str = "engine_failure"):
        """Motor ayrılır (28 Eylül motor arızası mekanizması): governor kapalı → gaz 0 → serbest tekerlek."""
        if self._engine_out:
            return
        fdm["fcs/rpm-governor-active-norm"] = 0.0
        self._engine_out = True
        self.stats["engine_out_t"] = self.t
        self.events.append(dict(t=self.t, kind=reason))

    @property
    def engine_is_out(self) -> bool:
        return self._engine_out

    # -----------------------------------------------------------------------------------------------------------
    # gözlem / bilgi / özet
    # -----------------------------------------------------------------------------------------------------------
    @property
    def wind_ned(self) -> tuple[float, float]:
        return self._wind_ned

    def info(self, fdm=None) -> dict:
        d = dict(self._last)
        d["engine_out"] = self._engine_out
        d["wind_n_fps"], d["wind_e_fps"] = self._wind_ned
        d["delta_T_C"] = float(self.params.get("delta_T_C", 0.0))
        if fdm is not None:
            d["fuel_lbs"] = fuel_total(fdm)
            d["density_altitude_ft"] = float(fdm["atmosphere/density-altitude"])
            d["gust_n_fps"] = float(fdm["atmosphere/total-wind-north-fps"]) - self._wind_ned[0]
            d["gust_e_fps"] = float(fdm["atmosphere/total-wind-east-fps"]) - self._wind_ned[1]
            d["gust_d_fps"] = float(fdm["atmosphere/total-wind-down-fps"])
        return d

    def summary(self) -> dict:
        s = dict(self.stats)
        s["params"] = dict(self.params)
        s["events"] = list(self.events)
        if s["peak_psi"] == -np.inf:
            s["peak_psi"] = float("nan")
        if s["min_rpm"] == np.inf:
            s["min_rpm"] = float("nan")
        return s


# =====================================================================================================================
# HIZLI TEST:  python physics_ext.py
# =====================================================================================================================

if __name__ == "__main__":
    # 1) varsayılan config pasif ve RNG'ye dokunmuyor
    cfg = PhysicsExtConfig()
    assert not cfg.active and cfg.to_dict() == {}
    g = np.random.default_rng(5)
    s0 = g.bit_generator.state
    ext = PhysicsExt(None)
    assert ext.begin_episode(g, 0.0) == {} and not ext.armed and g.bit_generator.state == s0
    # 2) dict ↔ config gidiş-dönüş (model zip'indeki ah1s_env_overrides["physics"])
    d = {"fuel": {"enable": True}, "wind": {"enable": True, "speed_kt": [0, 20]}, "turb": {"enable": True},
         "atmo": {"enable": True, "delta_T_C": [-10, 30]}}
    c2 = PhysicsExtConfig.from_dict(d)
    assert c2.active and c2.wind.speed_kt == (0, 20) and PhysicsExtConfig.from_dict(c2.to_dict()).to_dict() == c2.to_dict()
    assert c2.atmo.delta_T_C == (-10, 30)
    # sıcaklık yalnızca istenince örneklenir (kapalıyken bölüm RNG akışı aynı)
    e1, e2 = PhysicsExt({"wind": {"enable": True, "speed_kt": [0, 20]}}), PhysicsExt({"wind": {"enable": True, "speed_kt": [0, 20]}})
    p1 = e1.begin_episode(np.random.default_rng(3), 0.0)
    p2 = e2.begin_episode(np.random.default_rng(3), 0.0, {"delta_T_C": 25.0})
    assert "delta_T_C" not in p1 and p2["delta_T_C"] == 25.0 and p1["wind_kt"] == p2["wind_kt"]
    # 3) tank çekimi
    assert np.allclose(draw_from_tanks((890, 890), 100, "equal"), (840, 840))
    assert np.allclose(draw_from_tanks((10, 890), 100, "equal"), (0, 800))
    assert np.allclose(draw_from_tanks((300, 900), 120, "proportional"), (270, 810))
    assert np.allclose(draw_from_tanks((890, 890), 1000, "fwd_first"), (0, 780))
    assert np.allclose(draw_from_tanks((890, 890), 5000, "aft_first"), (0, 0))
    # 4) yakıt akışı: el kitabı doğruları (2000 ft, 43 psi ≈ 678 lb/h @ +15 °C; el kitabı örneği −30 °C'de 648)
    ff = fuel_flow_chart_lbh(43.0, 324.0, 2000.0)
    assert 670 < ff < 690, ff
    psi_1800 = 1800.0 / (SHP_AT_100 / PSI_AT_100)
    sfc_1800 = fuel_flow_chart_lbh(psi_1800, 324.0, 0.0) / 1800.0
    assert abs(sfc_1800 - T53_703_TO_SFC) < 0.03, sfc_1800
    # 5) rüzgâr yönü: kuzeyden 10 kt → hava kütlesi güneye
    vn, ve = wind_ned_fps(10 * KT_TO_FPS, 0.0)
    assert vn < 0 and abs(ve) < 1e-9
    # 6) MIL-F-8785C: 100 ft, W20 = 15 kt → σ_w = 2.53, σ_u = 4.35 ft/s
    q = dryden_low_alt_params(100.0, 15 * KT_TO_FPS)
    assert abs(q["sig_w"] - 2.53) < 0.01 and abs(q["sig_u"] - 4.35) < 0.02
    # 7) tork cezası (kalkış env'i 2026-09-28 ile aynı): 53 psi → 0.3·(3/6)² = 0.075; 59 psi → 0.3 + 2·1
    assert torque_penalty(49.0, 324.0, 0.3, 2.0, 1.0) == 0.0
    assert abs(torque_penalty(53.0, 324.0, 0.3, 2.0, 1.0) - 0.075) < 1e-12
    assert abs(torque_penalty(59.0, 324.0, 0.3, 2.0, 1.0) - 2.3) < 1e-12
    assert abs(torque_penalty(40.0, 304.0, 0.3, 2.0, 1.0) - 1.0) < 1e-12
    assert abs(psi_to_throttle(56.0) - 0.6261) < 1e-3
    print(f"physics_ext: tamam (43 psi / 2000 ft → {ff:.0f} lb/h; 1800 shp'de eşdeğer SFC {sfc_1800:.3f})")
