from __future__ import annotations

"""
COMMAND VIZ — komut / manevra ajanını 3D ve metriklerle canlı izle
==================================================================

Tek bir JSBSim uçuşu (eğitilmiş PPO + env) arka planda gerçek zamanlı akar.
Tarayıcıdaki sayfadan istediğin an Δheading / Δhız / Δirtifa komutu verirsin;
helikopteri (low-poly AH-1S modeli, `viz/ah1s_model.js`) 3D izler, komutun
metriklerini (yükselme, aşma, oturma, kuplaj, eğitimdeki başarı kararı) ve
zaman serilerini görürsün. Aynı sayfa kayıtlı uçuşları da oynatır.

İki görev (model observation boyutundan anlaşılır):
  manevra (varsayılan: dayanıklı model models_maneuver/maneuver_robust_final.zip, README 29;
      önceki models_maneuver/maneuver_M5_final.zip): helicopter_env_maneuver.py,
      0–100 kt, her komutun bir süre hedefi (T) var — boş bırakılırsa seçili çeviklikten
      (M3 rahat / M4 hızlı / M5 agresif). Sayfada attitude göstergesi ve ajanın yatış /
      yunuslama komutları görünür. Manevra sürerken yeni komut verilebilir (önceki «kesilir»).
      Env, modelin zip'inde taşıdığı ayarlarla kurulur (dayanıklı model: collective ±0.45).
  komut (models_command_curriculum/v2_R1_final.zip): helicopter_env_command.py, 15 ft/s civarı.
  kalkış (models_takeoff/…): helicopter_env_takeoff.py — yerden kalkış, hover manevraları, iniş (4 kumanda doğrudan).
  uçuş (models_flight/flight_v2.zip 2026-10-01, yoksa flight_final.zip 2026-09-29): helicopter_env_flight.py — tek ajan:
      kalkış → hover → ileri uçuşa geçiş → ileri uçuşta Δhız / Δheading / Δirtifa → duruş → iniş; rüzgâr / türbülans /
      yakıt / tork. Sayfada rüzgâr oku, yakıt ve tork göstergeleri; yeniden başlatmada rüzgâr, türbülans, yakıt ve hava
      sıcaklığı seçilir. Komutlar flight_commands.py yönlendiricisinden geçer (doğal zarf: 10–130 kt, 50–1500 ft).

Kullanım
--------
  Yerel (repo kökünde):
    python command_viz.py                       # dayanıklı manevra modeli → http://127.0.0.1:8765
    python command_viz.py --model models_maneuver/maneuver_M5_final.zip                    # önceki manevra modeli
    python command_viz.py --model models_command_curriculum/v2_R1_final.zip --port 8766   # eski komut modeli
    python command_viz.py --model models_flight/flight_final.zip --start ground --wind-kt 15 --turb light   # tek ajan
  Colab (hücrede, satır içi; localhost / paylaşım linki yok):
    import command_viz; command_viz.colab()
  Kayıtlı demo uçuşları üret (sayfanın "kayıt" modu için):
    python command_viz.py record --out viz/demo_flights.json

Uçuş dışa aktarma: sayfadaki JSON / ACMI düğmeleri (ACMI = Tacview kaydı).
Komut uygulama yolu eğitim ve değerlendirmeyle aynıdır (env.queue_command →
env._issue_due_commands): güvenli aralığın dışına taşan Δ'nın işareti çevrilir.
"""

import argparse
import json
import math
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from command_curriculum import AXES  # noqa: E402
from helicopter_env_command import CONTROL_DT, CommandEnvConfig, HelicopterEnvCommand  # noqa: E402
from helicopter_env_maneuver import (ENV_OVERRIDES_ATTR, OBS_DIM_M, HelicopterEnvManeuver,  # noqa: E402
                                     ManeuverEnvConfig)
from helicopter_env_takeoff import OBS_DIM_T, HelicopterEnvTakeoff, TakeoffEnvConfig  # noqa: E402
from helicopter_env_flight import FLIGHT_OBS_DIMS, OBS_DIM_F, FlightEnvConfig, HelicopterEnvFlight  # noqa: E402
from flight_commands import apply as apply_route, route_command, route_task  # noqa: E402
from flight_curriculum import (CMD_MAX_ALT_FT, CMD_MAX_KT, CMD_MIN_ALT_FT, CMD_MIN_KT, CRUISE_MAX_ALT_FT,  # noqa: E402
                               CRUISE_MAX_KT, CRUISE_MIN_ALT_FT, CRUISE_MIN_KT)
from maneuver_curriculum import DEFAULT_MANEUVER_LEVELS  # noqa: E402
from takeoff_curriculum import DEFAULT_TAKEOFF_LEVELS, GROUND_H_FT, TANK_CAPACITY_LBS  # noqa: E402

VIZ_DIR = REPO_ROOT / "viz"
PAGE_FILE = VIZ_DIR / "command_viz.html"
COMMAND_POLICY = REPO_ROOT / "models_command_curriculum" / "v2_R1_final.zip"
MANEUVER_POLICY = REPO_ROOT / "models_maneuver" / "maneuver_M5_final.zip"
ROBUST_POLICY = REPO_ROOT / "models_maneuver" / "maneuver_robust_final.zip"     # collective ±0.45 + S4 ince ayar (README 29)
TAKEOFF_POLICY = REPO_ROOT / "models_takeoff" / "takeoff_final.zip"              # yerden kalkış / hover / iniş, 4 kumanda (README 30)
# tek ajan: kalkış → ileri uçuş → iniş (README 32); 2026-10-01: flight_v2 (doğal zarf, sıcak gün, hover hassasiyeti —
# docs/flight/README.md bölüm 8), yoksa flight_final
FLIGHT_POLICY = next((p for p in (REPO_ROOT / "models_flight" / "flight_v5.zip",                 # 2026-10-03 (README 35.5, rejim uzmanlı)
                                  REPO_ROOT / "models_flight" / "flight_v4.zip",
                                  REPO_ROOT / "models_flight" / "flight_v3.zip",
                                  REPO_ROOT / "models_flight" / "flight_v2.zip",
                                  REPO_ROOT / "models_flight" / "flight_final.zip") if p.exists()),
                     REPO_ROOT / "models_flight" / "flight_final.zip")
# varsayılan: dayanıklı manevra modeli; yoksa M5 manevra modeli; o da yoksa eski komut modeli
DEFAULT_POLICY = next((p for p in (ROBUST_POLICY, MANEUVER_POLICY) if p.exists()), COMMAND_POLICY)
# görev başına varsayılan başlangıç (eğitim aralığının içinde)
DEFAULT_START = {"command": dict(alt=300.0, speed=15.0, heading=0.0, seed=0),
                 "maneuver": dict(alt=800.0, speed=101.3, heading=0.0, seed=0),       # 800 ft, 60 kt
                 # kalkış: yerde (rotor warm-up sonrası) ya da havada hover; yakıt tank başına lbs
                 "takeoff": dict(start="ground", alt=100.0, speed=0.0, heading=0.0, seed=0, fuel=[0.0, 0.0]),
                 # tek ajan: + ileri uçuşta başlangıç (hava hızı kt), rüzgâr (kt, heading'e göre yön °), türbülans
                 "flight": dict(start="ground", alt=100.0, speed=0.0, speed_kt=60.0, heading=0.0, seed=0,
                                fuel=[300.0, 300.0], wind_kt=0.0, wind_dir=0.0, turb="none", gusts=False)}
FORMAT = "ah1s-command-flight/1"
EARTH_RADIUS_FT = 20_902_231.0
LIVE_EPISODE_S = 4 * 3600.0              # canlı uçuşta zaman sınırı pratikte yok
MAX_ROWS_PER_POLL = 4000

# Telemetri sütunları (her kontrol adımında bir satır, 0.075 s) ve yuvarlama basamağı
COLUMNS = [
    ("t", 3),                                            # s, reset'ten beri
    ("x", 1), ("y", 1), ("h", 2),                        # doğu ft, kuzey ft, irtifa AGL ft
    ("psi", 2), ("u", 3), ("v", 3), ("vs", 3),           # heading °, ileri / yanal hız, dikey hız ft/s
    ("phi", 2), ("theta", 2), ("r", 2), ("rpm", 1),      # roll °, pitch °, yaw rate °/s, rotor rpm
    ("psi_ref", 2), ("u_ref", 3), ("h_ref", 2),          # hedefler (heading 0–360)
    ("e_psi", 3), ("e_u", 3), ("e_h", 3),                # hata = hedef − ölçülen
    ("a0", 3), ("a1", 3), ("a2", 3), ("a3", 3),          # PPO action (−1…1)
    ("f0", 3), ("f1", 3), ("f2", 3), ("f3", 3),          # filtrelenmiş action = trim'e eklenen residual
    ("c0", 4), ("c1", 4), ("c2", 4), ("c3", 4),          # kumanda: collective, elevator, aileron, rudder (norm)
    ("rew", 4),                                          # PPO'nun gördüğü ödül (ölçekli)
    ("phi_c", 2), ("theta_c", 2),                        # manevra env'i: attitude komutu (°); komut env'inde boş
    ("tx", 1), ("ty", 1), ("wow", 0), ("wf", 2),         # kalkış env'i: hedef konum (doğu / kuzey ft), yerdeki kızak sayısı,
                                                         # kızaklardaki ağırlık oranı
    # uçuş env'i (tek ajan): hava hızı ileri kt, hedef hava hızı kt (ileri uçuş penceresinde), trim çizelgesinin hızı kt,
    # toplam rüzgâr kuzey / doğu ft/s (ortalama + türbülans + gust), tork psi, yakıt lbs, yana hava hızı ft/s,
    # hız hatası ft/s (süzülmüş), ileri uçuş bayrağı
    ("ua", 2), ("uat", 2), ("uff", 2), ("wn", 2), ("we", 2), ("tq", 2), ("fuel", 1), ("va", 2), ("esp", 2), ("cr", 0),
    # rota tutma (canlı, 2026-10-02): hedef yer izi ° (ileri uçuşta; yoksa boş)
    ("crs", 1),
]
COLS = [c for c, _ in COLUMNS]
AXIS_ERR_COL = {"heading": "e_psi", "speed": "e_u", "altitude": "e_h"}


def _limits_takeoff(cfg: TakeoffEnvConfig) -> dict:
    """Kalkış görevi: sayfanın başarı / güvenlik metinleri ve bant hesabı için (env ile aynı sayılar)."""
    lv = {x.name: x for x in DEFAULT_TAKEOFF_LEVELS}
    return dict(
        task="takeoff",
        timing=dict(grace_frac=0.25, grace_min_s=2.0,
                    levels=[dict(name=x.name, description=x.description, climb_fps=x.climb_fps, descent_fps=x.descent_fps,
                                 move_fps=x.move_fps, accel_fps2=x.accel_fps2, yaw_rate_dps=x.yaw_rate_dps, lag_s=x.lag_s,
                                 hold_s=x.hold_s, hold_first_s=x.hold_first_s, hold_T_s=x.hold_T_s, recover_s=x.recover_s)
                            for x in DEFAULT_TAKEOFF_LEVELS]),
        authority=dict(hover_trim=list(cfg.hover_trim), ctrl_range=list(cfg.ctrl_range), expo=cfg.expo,
                       rate_limit=list(cfg.rate_limit), direct=True),
        # bant: konum / irtifa toleransı hedef irtifaya göre (clip(a + b·(h* − 6.3), a, c))
        takeoff=dict(tol_h=list(cfg.tol_h), tol_xy=list(cfg.tol_xy), tol_psi_deg=cfg.tol_psi_deg, tol_v_fps=cfg.tol_v_fps,
                     tol_vs_fps=cfg.tol_vs_fps, land_tol_xy_ft=cfg.land_tol_xy_ft, land_tol_psi_deg=cfg.land_tol_psi_deg,
                     land_weight_frac=cfg.land_weight_frac, land_hold_s=cfg.land_hold_s,
                     touchdown_ok_fps=cfg.touchdown_ok_fps, ground_h_ft=GROUND_H_FT, sink_base_fps=cfg.sink_base_fps,
                     sink_slope=cfg.sink_slope, tank_capacity_lbs=TANK_CAPACITY_LBS),
        tol={"heading": cfg.tol_psi_deg, "speed": cfg.tol_v_fps, "altitude": cfg.tol_h[0]},
        coupling={"xy": cfg.coupling_limits[0], "altitude": cfg.coupling_limits[1], "heading": cfg.coupling_limits[2]},
        hold_s=lv["K9"].hold_s,
        safety=dict(max_roll_deg=cfg.max_roll_deg, max_pitch_deg=cfg.max_pitch_deg, max_roll_rate_dps=cfg.max_rate_dps,
                    max_yaw_rate_dps=cfg.max_yaw_rate_dps, ground_max_att_deg=cfg.ground_max_att_deg,
                    crash_vs_fps=cfg.crash_vs_fps, flyaway_ft=cfg.flyaway_ft, alt_over_ft=cfg.alt_over_ft,
                    max_speed_fps=cfg.max_speed_fps, rpm_limits=list(cfg.rpm_limits), min_agl_ft=0.0),
        trained=dict(takeoff_alt_ft=[12.0, 1000.0], turn_deg=lv["K5"].turn_deg[1], move_ft=lv["K5"].move_ft[1],
                     bob_ft=lv["K5"].bob_ft[1], start_alt_ft=[12.0, 400.0], fuel_lbs=[0.0, TANK_CAPACITY_LBS]),
    )


def _limits_flight(cfg: FlightEnvConfig) -> dict:
    """Tek ajan: kalkış görevinin sınırları + ileri uçuş bandı, tork / yakıt, türbülans bant katsayıları."""
    L = _limits_takeoff(cfg)
    L["task"] = "flight"
    L["course_hold"] = bool(getattr(cfg, "course_hold", False))        # sayfa: heading komutu rota mı
    L["flight"] = dict(tol_u_fps=cfg.tol_u_fps, tol_psi_cruise_deg=cfg.tol_psi_cruise_deg, tol_h_cruise_ft=cfg.tol_h_cruise_ft,
                       tol_vs_cruise_fps=cfg.tol_vs_cruise_fps, coupling_cruise=list(cfg.coupling_cruise),
                       tol_turb_scale=dict(cfg.tol_turb_scale), max_airspeed_kt=cfg.max_airspeed_kt,
                       max_roll_cruise_deg=cfg.max_roll_cruise_deg, torque_cont_psi=cfg.torque_cont_psi,
                       torque_max_psi=cfg.torque_max_psi, fuel_capacity_lbs=2.0 * TANK_CAPACITY_LBS,
                       trim_schedule=cfg.trim_schedule, power_cap_psi=cfg.power_cap_psi)
    L["safety"].update(max_airspeed_kt=cfg.max_airspeed_kt, max_roll_cruise_deg=cfg.max_roll_cruise_deg,
                       flyaway_track_ft=cfg.flyaway_track_ft)
    L["trained"].update(cruise_kt=[CRUISE_MIN_KT, CRUISE_MAX_KT], cruise_alt_ft=[CRUISE_MIN_ALT_FT, CRUISE_MAX_ALT_FT],
                        du_kt=35.0, dpsi_deg=180.0, dh_ft=300.0, accel_kt=[40.0, 80.0], wind_kt=[0.0, 25.0],
                        turb=["none", "light", "moderate"], weight_lbs=[8800.0, 9700.0], fuel_lbs=[150.0, 600.0],
                        start_alt_ft=[12.0, 1000.0])
    # doğal komut zarfı (2026-10-01; flight_commands bu sınırlara kırpar — F10 ile eğitildi)
    L["command"] = dict(speed_kt=[0.0, CMD_MAX_KT], cruise_min_kt=CMD_MIN_KT, cruise_alt_ft=[CMD_MIN_ALT_FT, CMD_MAX_ALT_FT],
                        hover_alt_ft=[12.0, CMD_MAX_ALT_FT], dpsi_deg=360.0)
    return L


def _limits(cfg: CommandEnvConfig, task: str = "command") -> dict:
    if task == "flight":
        return _limits_flight(cfg)
    if task == "takeoff":
        return _limits_takeoff(cfg)
    if task == "maneuver":
        trained = {"heading": 180.0, "speed": 50.0, "altitude": 150.0,
                   "start_alt_ft": [600.0, 1500.0], "start_speed_fps": [0.0, 169.0]}
        timing = dict(grace_frac=cfg.deadline_grace_frac, grace_min_s=cfg.deadline_grace_min_s,
                      yaw_decel_dps2=cfg.yaw_decel_dps2, vs_decel_fps2=cfg.vs_decel_fps2,
                      accel_jerk_fps3=cfg.accel_jerk_fps3,
                      levels=[dict(name=lv.name, description=lv.description, pedal_rate_dps=lv.pedal_rate_dps,
                                   bank_deg=lv.bank_deg, accel_fps2=lv.accel_fps2, climb_fps=lv.climb_fps,
                                   lag_s=lv.lag_s) for lv in DEFAULT_MANEUVER_LEVELS])
    else:
        # R1 seviyesinin eğitim aralıkları (command_curriculum.py)
        trained = {"heading": 180.0, "speed": 10.0, "altitude": 100.0,
                   "start_alt_ft": [200.0, 1000.0], "start_speed_fps": [10.0, 25.0]}
        timing = None
    return dict(
        task=task, timing=timing,
        # manevra: kumanda yetkisi (collective trim ± coll_scale, pedal ± pedal_scale, attitude komutu ± φ / θ)
        authority=(dict(coll_scale=cfg.coll_scale, pedal_scale=cfg.pedal_scale, roll_cmd_deg=cfg.roll_cmd_deg,
                        pitch_cmd_deg=cfg.pitch_cmd_deg) if task == "maneuver" else None),
        tol={"heading": cfg.tol_heading_deg, "speed": cfg.tol_speed_fps, "altitude": cfg.tol_alt_ft},
        coupling=(dict(zip(AXES, cfg.coupling_limits)) if cfg.coupling_limits is not None else None),
        hold_s=cfg.success_hold_s,
        safety=dict(max_roll_deg=cfg.max_roll_deg, max_pitch_deg=cfg.max_pitch_deg, min_agl_ft=cfg.min_agl_ft,
                    max_yaw_rate_dps=cfg.max_yaw_rate_dps, max_roll_rate_dps=cfg.max_roll_rate_dps,
                    rpm_limits=list(cfg.rpm_limits),
                    alt_margin_ft=cfg.alt_margin_ft, speed_margin_fps=cfg.speed_margin_fps,
                    heading_margin_deg=cfg.heading_margin_deg,
                    cmd_speed_range=[0.0, 170.0] if task == "maneuver" else list(cfg.cmd_speed_range),
                    cmd_min_alt_ft=cfg.cmd_min_alt_ft),
        trained=trained,
    )


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def task_text(t: dict) -> str:
    """Kalkış görevinin kısa metni (ACMI olayları / log)."""
    k = t.get("kind")
    if k in ("takeoff", "climb_to"):
        return f"{'kalkış' if k == 'takeoff' else 'yeni irtifa'} {t.get('h', 0):.0f} ft"
    if k == "turn":
        return f"dönüş {t.get('dpsi', 0):+.0f}°"
    if k == "move":
        return f"kayma ileri {t.get('dx', 0):+.0f} / sağa {t.get('dy', 0):+.0f} ft"
    if k == "bob":
        return f"bob {t.get('dh', 0):+.0f} ft"
    if k == "recover":
        return f"bozucu {t.get('dist', '')}"
    if k == "cruise":
        if t.get("hold"):
            return "ileri uçuşu tut"
        p = []
        if "u_kt" in t:
            p.append(f"{'hızlan' if t.get('accel') else 'hız'} → {t['u_kt']:.0f} kt")
        if t.get("du_kt"):
            p.append(f"Δhız {t['du_kt']:+.0f} kt")
        if t.get("dpsi"):
            p.append(f"Δψ {t['dpsi']:+.0f}°")
        if t.get("dh"):
            p.append(f"Δh {t['dh']:+.0f} ft")
        if t.get("h") is not None:
            p.append(f"irtifa → {t['h']:.0f} ft")
        return "ileri uçuş: " + (", ".join(p) or "devam")
    if k == "pirouette":
        return f"pirouette {t.get('radius', 100):.0f} ft / {t.get('circle_s', 45):.0f} s"
    if k == "stop":
        return "dur (hover)"
    return {"land": "iniş", "hold": "hover tut"}.get(k, str(k))


def _clean(x):
    """JSON için: NaN/inf → None, numpy → python, iç içe yapılar."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_policy(path: str | Path, with_obs_dim: bool = False):
    """Deterministik policy. with_obs_dim → (fn, obs boyutu, env ayarları): görev obs boyutundan anlaşılır;
    env ayarları modelin eğitildiği, varsayılandan farklı ayarlar (ör. {"coll_scale": 0.45}; eski modellerde {})."""
    from stable_baselines3 import PPO
    model = PPO.load(str(path), device="cpu")
    fn = lambda obs: model.predict(obs, deterministic=True)[0]          # noqa: E731
    if with_obs_dim:
        return fn, int(model.observation_space.shape[0]), dict(getattr(model, ENV_OVERRIDES_ATTR, None) or {})
    return fn


# =====================================================================
# KOMUT METRİKLERİ (sayfadaki JS ile aynı tanım)
# =====================================================================

def command_metrics(t: np.ndarray, err: dict, applied: dict, t0: float, t_end: float,
                    limits: dict, terminated: bool = False, deadline: float | None = None) -> dict:
    """Bir komut penceresinin basamak cevabı metrikleri.

    t: zaman dizisi; err: {eksen: hata dizisi}; applied: uygulanan Δ; pencere
    (t0, t_end]. Komut verilen eksen: yükselme (%10→%90), aşma %, oturma (bant
    içine girip bir daha çıkmadığı an), son hold_s (komut 10 s, manevra 5 s) ortalama hata. Komut verilmeyen
    eksen: pencere boyunca en büyük |hata| (kuplaj) ve sınırı. Hepsi birlikte:
    sondaki "tüm eksenler tolerans içinde" süresi (streak).
    """
    tol, lim = limits["tol"], limits["coupling"]
    sel = (t >= t0 + CONTROL_DT - 1e-9) & (t <= t_end + 1e-9)
    ts = t[sel]
    out = dict(t0=t0, t_end=t_end, axes={}, streak_s=0.0, coupling_exceeded=False)
    if ts.size == 0:
        return out
    inside_all = np.ones(ts.size, dtype=bool)
    last10 = ts >= ts[-1] - float(limits.get("hold_s") or 10.0) - 1e-9     # son hata: son hold_s (komut 10 s, manevra 5 s)
    for a in AXES:
        e = np.asarray(err[a], dtype=np.float64)[sel]
        inside_all &= np.abs(e) <= tol[a]
        d = float(applied.get(a, 0.0))
        if d != 0.0:
            frac = (d - e) / d
            i10 = np.flatnonzero(frac >= 0.1)
            i90 = np.flatnonzero(frac >= 0.9)
            rise = float(ts[i90[0]] - ts[i10[0]]) if i10.size and i90.size and i90[0] >= i10[0] else None
            outside = np.abs(e) > tol[a]
            if not outside.any():
                settle = 0.0
            elif not outside[-1]:
                settle = float(ts[np.flatnonzero(outside)[-1] + 1] - t0)
            else:
                settle = None
            out["axes"][a] = dict(commanded=True, delta=d, rise_s=rise,
                                  overshoot_pct=float(max(0.0, frac.max() - 1.0) * 100.0),
                                  settle_s=settle, final=float(e[last10].mean()), now=float(e[-1]))
        else:
            peak = float(np.abs(e).max())
            limit = None if lim is None else float(lim[a])
            exceeded = limit is not None and peak > limit
            out["coupling_exceeded"] |= exceeded
            out["axes"][a] = dict(commanded=False, coupling=peak, limit=limit, exceeded=exceeded,
                                  final=float(e[last10].mean()), now=float(e[-1]))
    run = 0
    for ok in inside_all[::-1]:
        if not ok:
            break
        run += 1
    out["streak_s"] = run * CONTROL_DT
    # tüm eksenlerin birlikte banda son girişi (manevra env'inin "oturma"sı) ve süre hedefi
    out["settle_all_s"] = float(ts[ts.size - run] - t0) if run else None
    on_time = True
    if deadline is not None:
        out["deadline"] = float(deadline)
        on_time = out["settle_all_s"] is not None and out["settle_all_s"] <= deadline + 1e-9
        out["on_time"] = bool(on_time)
    out["success_live"] = bool(not terminated and out["streak_s"] >= limits["hold_s"] - 1e-9
                               and not out["coupling_exceeded"] and on_time)
    return out


# =====================================================================
# CANLI UÇUŞ
# =====================================================================

class LiveFlight:
    """Tek FDM üzerinde PPO + komutlar. Tüm JSBSim çağrıları tek iş parçacığında.

    Dışarıdan (HTTP / Colab) çağrılanlar: hello, state_since, request_command,
    request_reset, set_control, export_dict, export_acmi. Bunlar yalnızca kilit
    altında Python verisi okur / istek kuyruğa koyar.
    """

    def __init__(self, policy_path: str | Path = DEFAULT_POLICY, env_config: str = "v2", level: str | None = None,
                 policy=None, start: dict | None = None, task: str | None = None, course_hold: bool = False):
        self.policy_path = Path(policy_path)
        self.env_config = env_config
        obs_dim, self.env_overrides = None, {}
        if policy is None:
            policy, obs_dim, self.env_overrides = load_policy(self.policy_path, with_obs_dim=True)
        # görev modelin observation boyutundan anlaşılır (komut 19, manevra 24, kalkış 29; kalkışa 2026-09-28'de
        # tork (+1) ve hareketli hedef hızı (+2) eklendi → 30 / 31 / 32)
        takeoff_dims = {OBS_DIM_T, OBS_DIM_T + 1, OBS_DIM_T + 2, OBS_DIM_T + 3}
        self.task = task or ("flight" if obs_dim in FLIGHT_OBS_DIMS else "takeoff" if obs_dim in takeoff_dims
                             else {OBS_DIM_M: "maneuver"}.get(obs_dim, "command"))
        self.is_to = self.task in ("takeoff", "flight")          # görev listesi / pencereler kalkış env'inin yapısında
        if self.task == "flight":
            # course_hold: ileri uçuşta heading komutu yer izi; burun rüzgâra göre düzeltilir (eğitimde kapalıydı)
            cfg = FlightEnvConfig(**{**self.env_overrides, "course_hold": bool(course_hold)})
            self.env = HelicopterEnvFlight(level=level or "F8", config=cfg)
        elif self.task == "takeoff":
            cfg = TakeoffEnvConfig(**self.env_overrides)
            self.env = HelicopterEnvTakeoff(level=level or "K9", config=cfg)
        elif self.task == "maneuver":
            cfg = ManeuverEnvConfig(**self.env_overrides)          # modelin eğitildiği ayarlar (ör. collective ±0.45)
            self.env = HelicopterEnvManeuver(level=level or "M5", config=cfg)
        else:
            cfg = CommandEnvConfig.v1() if env_config == "v1" else CommandEnvConfig()
            self.env = HelicopterEnvCommand(level=level or "R1", config=cfg)
        self.limits = _limits(cfg, self.task)
        self.policy = policy
        self.lock = threading.RLock()
        self.start_opts = dict(DEFAULT_START[self.task])
        if start:
            self.start_opts.update({k: v for k, v in start.items() if v is not None})
        self.flight_id = 0
        self.rows: list[list] = []
        self.commands: list[dict] = []
        self.events: list[dict] = []
        self.meta: dict = {}
        self.state = "starting"
        self.message = "JSBSim hazırlanıyor…"
        self.termination = None
        self.paused = False
        self.time_scale = 1.0
        self.sim_rate = 0.0
        self._pending_cmds: list[dict] = []
        self._pending_reset: dict | None = dict(self.start_opts)
        self._obs = None
        self._done = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lat0 = self._lon0 = 0.0
        self._ground_msl = 0.0

    # ---------------- dış API (iş parçacığı güvenli) ----------------

    def hello(self) -> dict:
        with self.lock:
            return _clean(dict(ok=True, format=FORMAT, mode="live", cols=COLS, control_dt=CONTROL_DT, task=self.task,
                               model=self.policy_path.name, env_config=self.env_config, limits=self.limits,
                               start=self.start_opts, flight=self.flight_id, status=self._status()))

    def state_since(self, flight: int | None = None, since: int = 0, max_rows: int = MAX_ROWS_PER_POLL) -> dict:
        with self.lock:
            since = int(since or 0)
            if flight is None or int(flight) != self.flight_id or since > len(self.rows):
                since = 0                                   # yeni uçuş: baştan gönder
            rows = self.rows[since:since + max_rows]
            return _clean(dict(flight=self.flight_id, n0=since, n=len(self.rows), rows=rows,
                               commands=self.commands, events=self.events, meta=self.meta,
                               status=self._status()))

    def request_command(self, delta: dict) -> dict:
        if self.is_to:
            return self._request_task(delta or {})
        try:
            d = {a: float((delta or {}).get(a, 0.0) or 0.0) for a in AXES}
            T = (delta or {}).get("T")
            T = None if T in (None, "", 0) else float(T)
        except (TypeError, ValueError):
            return dict(ok=False, message="Komut sayı olmalı.")
        if T is not None and not (math.isfinite(T) and 1.0 <= T <= 120.0):
            return dict(ok=False, message="Süre hedefi 1–120 s olmalı.")
        if not all(math.isfinite(v) for v in d.values()):
            return dict(ok=False, message="Komut sayı olmalı.")
        if all(v == 0.0 for v in d.values()):
            return dict(ok=False, message="En az bir eksende sıfırdan farklı Δ gir.")
        with self.lock:
            if self._done or self.state in ("starting", "resetting"):
                return dict(ok=False, message="Uçuş çalışmıyor; önce yeniden başlat.")
            if len(self._pending_cmds) >= 3:
                return dict(ok=False, message="Kuyrukta zaten bekleyen komutlar var.")
            if T is not None and self.task == "maneuver":
                d["T"] = T
            self._pending_cmds.append(d)
            return dict(ok=True, queued=d, message="Komut kuyruğa alındı.")

    def request_reset(self, alt=None, speed=None, heading=None, seed=None, start=None, fuel=None, **extra) -> dict:
        if self.is_to:
            return self._request_reset_takeoff(alt, heading, seed, start, fuel, **extra)
        opts = dict(self.start_opts)
        try:
            if alt is not None:
                opts["alt"] = float(alt)
            if speed is not None:
                opts["speed"] = float(speed)
            opts["heading"] = None if heading in (None, "", "random") else float(heading) % 360.0
            if seed is not None:
                opts["seed"] = int(seed)
        except (TypeError, ValueError):
            return dict(ok=False, message="Başlangıç değerleri sayı olmalı.")
        vmax = 170.0 if self.task == "maneuver" else 60.0
        if not 60.0 <= opts["alt"] <= 5000.0 or not 0.0 <= opts["speed"] <= vmax:
            return dict(ok=False, message=f"Başlangıç: irtifa 60–5000 ft, hız 0–{vmax:.0f} ft/s olmalı.")
        with self.lock:
            self._pending_reset = opts
            self.state = "resetting"
            self.message = "Yeniden başlatılıyor…"
            return dict(ok=True, start=opts)

    # ---------------- kalkış görevi (yerden kalkış / hover / iniş) ----------------

    TASK_KINDS = ("takeoff", "climb_to", "turn", "move", "bob", "land", "recover")
    FLIGHT_KINDS = ("cruise", "accel", "stop", "hold", "komut", "pirouette")

    def _request_task(self, d: dict) -> dict:
        kind = str(d.get("kind") or "")
        if kind not in self.TASK_KINDS + (self.FLIGHT_KINDS if self.task == "flight" else ()):
            return dict(ok=False, message="Bilinmeyen görev.")
        try:
            num = lambda k, default=0.0: float(d.get(k, default) if d.get(k, default) not in (None, "") else default)  # noqa: E731
            if kind == "komut":                     # 2026-10-01: rejimden bağımsız tek komut (flight_commands yönlendirir)
                task = dict(kind="komut")
                for k2 in ("speed_kt", "dspeed_kt", "heading_deg", "dheading_deg", "alt_ft", "dalt_ft"):
                    if d.get(k2) not in (None, ""):
                        task[k2] = num(k2)
                if len(task) == 1:
                    return dict(ok=False, message="Komut: hız, heading ya da irtifa ver (mutlak ya da Δ).")
            elif kind == "pirouette":
                task = dict(kind="pirouette", radius=num("radius", 100.0), circle_s=num("circle_s", 45.0),
                            direction=1.0 if num("direction", 1.0) >= 0 else -1.0)
                if not (50.0 <= task["radius"] <= 200.0 and 30.0 <= task["circle_s"] <= 90.0):
                    return dict(ok=False, message="Pirouette: yarıçap 50–200 ft, süre 30–90 s.")
            elif kind == "cruise":                  # ileri uçuşta Δ (ölçülen duruma göre; verilmeyen eksen hedefine devam)
                task = dict(kind="cruise")
                for k2, lim in (("du_kt", 130.0), ("dpsi", 360.0), ("dh", 1500.0)):
                    v = num(k2)
                    if v:
                        if abs(v) > lim:
                            return dict(ok=False, message=f"{k2} en fazla ±{lim:.0f}.")
                        task[k2] = v
                if len(task) == 1:
                    return dict(ok=False, message="İleri uçuş: en az bir eksende (Δhız / Δψ / Δh) sıfırdan farklı değer gir.")
            elif kind == "accel":                   # hover'dan / düşük hızdan ileri uçuşa geçiş
                task = dict(kind="cruise", u_kt=num("u_kt", 60.0), dh=num("dh", 100.0), accel=True)
                if not CMD_MIN_KT <= task["u_kt"] <= CMD_MAX_KT:
                    return dict(ok=False, message=f"Hızlanma hedefi {CMD_MIN_KT:.0f}–{CMD_MAX_KT:.0f} kt olmalı.")
                if not -1500.0 <= task["dh"] <= 1500.0:
                    return dict(ok=False, message="Hızlanırken tırmanış −1500…+1500 ft olmalı.")
            elif kind == "stop":                    # ileri uçuştan duruş (hover)
                task = dict(kind="stop", decel=num("decel", 2.5))
                if not 1.0 <= task["decel"] <= 4.0:
                    return dict(ok=False, message="Yavaşlama 1–4 ft/s² olmalı.")
            elif kind == "hold":
                task = dict(kind="hold")
            elif kind in ("takeoff", "climb_to"):
                task = dict(kind=kind, h=num("h"))
                if not 12.0 <= task["h"] <= CMD_MAX_ALT_FT:
                    return dict(ok=False, message=f"Hedef irtifa 12–{CMD_MAX_ALT_FT:.0f} ft (CG, yerden) olmalı.")
            elif kind == "turn":
                task = dict(kind=kind, dpsi=num("dpsi"))
                if not (0.0 < abs(task["dpsi"]) <= 360.0):
                    return dict(ok=False, message="Δψ 0 ile ±360° arasında olmalı.")
            elif kind == "move":
                task = dict(kind=kind, dx=num("dx"), dy=num("dy"))
                if task["dx"] == 0.0 and task["dy"] == 0.0 or max(abs(task["dx"]), abs(task["dy"])) > 200.0:
                    return dict(ok=False, message="İleri / sağa kayma ±200 ft içinde ve sıfırdan farklı olmalı.")
            elif kind == "bob":
                task = dict(kind=kind, dh=num("dh"))
                if not (0.0 < abs(task["dh"]) <= CMD_MAX_ALT_FT):
                    return dict(ok=False, message=f"Δh 0 ile ±{CMD_MAX_ALT_FT:.0f} ft arasında olmalı.")
            elif kind == "land":
                task = dict(kind=kind)
            else:
                dist = str(d.get("dist") or "")
                if dist == "kick":
                    task = dict(kind=kind, dist=dist, axis=int(num("axis", 2)), mag=num("mag", 0.3), dur=num("dur", 0.8))
                    if not (0 <= task["axis"] <= 3 and abs(task["mag"]) <= 0.5 and 0.1 <= task["dur"] <= 2.0):
                        return dict(ok=False, message="Kumanda darbesi: eksen 0–3, |büyüklük| ≤ 0.5, süre 0.1–2 s.")
                elif dist == "push":
                    task = dict(kind=kind, dist=dist, fwd=num("fwd"), right=num("right"), dw=num("dw"))
                    if max(abs(task["fwd"]), abs(task["right"]), abs(task["dw"])) > 20.0:
                        return dict(ok=False, message="İtki: her bileşen ±20 ft/s içinde olmalı.")
                elif dist == "tilt":
                    task = dict(kind=kind, dist=dist, dphi=num("dphi"), dtheta=num("dtheta"))
                    if max(abs(task["dphi"]), abs(task["dtheta"])) > 20.0:
                        return dict(ok=False, message="Attitude bozucusu ±20° içinde olmalı.")
                else:
                    return dict(ok=False, message="Bozucu türü: kick / push / tilt.")
        except (TypeError, ValueError):
            return dict(ok=False, message="Görev değerleri sayı olmalı.")
        with self.lock:
            if self._done or self.state in ("starting", "resetting"):
                return dict(ok=False, message="Uçuş çalışmıyor; önce yeniden başlat.")
            if len(self._pending_cmds) >= 3:
                return dict(ok=False, message="Kuyrukta zaten bekleyen görevler var.")
            self._pending_cmds.append(dict(task=task))
            return dict(ok=True, queued=task, message="Görev kuyruğa alındı.")

    def _request_reset_takeoff(self, alt, heading, seed, start, fuel, speed_kt=None, wind_kt=None, wind_dir=None,
                               turb=None, gusts=None, temp_dc=None, **_ignored) -> dict:
        opts = dict(self.start_opts)
        fl = self.task == "flight"
        try:
            if start is not None:
                if start not in (("ground", "hover", "cruise") if fl else ("ground", "hover")):
                    return dict(ok=False, message="Başlangıç: ground (yerde), hover (havada)"
                                + (" ya da cruise (ileri uçuşta)." if fl else "."))
                opts["start"] = start
            if fl:
                if speed_kt is not None:
                    opts["speed_kt"] = float(speed_kt)
                if wind_kt is not None:
                    opts["wind_kt"] = float(wind_kt)
                if wind_dir is not None:
                    opts["wind_dir"] = float(wind_dir) % 360.0
                if turb is not None:
                    if turb not in ("none", "light", "moderate", "severe"):
                        return dict(ok=False, message="Türbülans: none / light / moderate / severe.")
                    opts["turb"] = turb
                if gusts is not None:
                    opts["gusts"] = bool(gusts)
                if temp_dc is not None:                      # standart günden sıcaklık farkı (°C; 2026-10-01)
                    opts["temp_dc"] = float(temp_dc)
                    if not -30.0 <= opts["temp_dc"] <= 40.0:
                        return dict(ok=False, message="Sıcaklık farkı −30…+40 °C olmalı (F10 eğitimi −10…+30).")
                if not 0.0 <= float(opts.get("wind_kt", 0.0)) <= 40.0:
                    return dict(ok=False, message="Rüzgâr 0–40 kt olmalı (eğitim 0–25 kt).")
                if opts.get("start") == "cruise" and not 15.0 <= float(opts.get("speed_kt", 60.0)) <= CMD_MAX_KT:
                    return dict(ok=False, message=f"İleri uçuşta başlangıç hızı 15–{CMD_MAX_KT:.0f} kt olmalı.")
            if alt is not None:
                opts["alt"] = float(alt)
            opts["heading"] = None if heading in (None, "", "random") else float(heading) % 360.0
            if fuel is not None:
                f = [float(x) for x in (fuel if isinstance(fuel, (list, tuple)) else (fuel, fuel))][:2]
                if len(f) != 2 or not all(0.0 <= x <= TANK_CAPACITY_LBS for x in f):
                    return dict(ok=False, message=f"Yakıt: tank başına 0–{TANK_CAPACITY_LBS:.0f} lbs.")
                opts["fuel"] = f
            if seed is not None:
                opts["seed"] = int(seed)
        except (TypeError, ValueError):
            return dict(ok=False, message="Başlangıç değerleri sayı olmalı.")
        if opts.get("start") in ("hover", "cruise") and not 12.0 <= float(opts.get("alt", 100.0)) <= CMD_MAX_ALT_FT:
            return dict(ok=False, message=f"Havada başlangıç irtifası 12–{CMD_MAX_ALT_FT:.0f} ft olmalı.")
        with self.lock:
            self._pending_reset = opts
            self.state = "resetting"
            self.message = "Yeniden başlatılıyor…"
            return dict(ok=True, start=opts)

    def _sync_windows_takeoff(self):
        env = self.env
        w2c = self.__dict__.setdefault("_win2cmd", {})
        if not self.commands:
            w2c.clear()                                  # yeni uçuş
        for k, w in enumerate(env.windows):
            if w["task"].get("auto"):                    # görev bitince env'in açtığı "hedefte kal" penceresi (komut değil)
                continue
            if k not in w2c:
                w2c[k] = len(self.commands)
                self.commands.append(dict(
                    id=len(self.commands) + 1, t=float(w["t"]), kind=w["kind"], task=dict(w["task"], kind=w["kind"]), T=w["T"],
                    deadline=w["deadline"], hold_s=w["hold_s"], allow_s=w["allow_s"], target=dict(w["target"]),
                    tol=list(w["tol"]), active=list(w["active"]), h_start=w.get("h_start"), verdict=None,
                    interrupted=False, streak_s=0.0, max_err=None, coupling_ok=True, settle_s=None, on_time=False,
                    touchdown_vs=None, final_err=None, category=w.get("category", w["kind"]),
                    u_target_kt=(w["u_target"] / 1.6878099 if w.get("u_target") else None),
                    tol_scale=float(getattr(env, "tol_scale", 1.0)),
                    stop_t_end=(float(env.track["t_end"]) if w["kind"] == "stop" and env.track and "t_end" in env.track
                                else None)))
            c = self.commands[w2c[k]]
            st = w["settle_s"]
            c.update(streak_s=float(w["streak"]), max_err=dict(w["max_err"]), coupling_ok=bool(w["coupling_ok"]),
                     settle_s=float(st) if _finite(st) else None, on_time=bool(w["on_time"]),
                     touchdown_vs=w["touchdown_vs"])
            if w["closed"] and c["verdict"] is None:
                c.update(verdict=bool(w["success"]), interrupted=bool(w["interrupted"]), final_err=w.get("final_err"))

    def set_control(self, paused=None, time_scale=None, **_ignored) -> dict:
        with self.lock:
            if paused is not None:
                self.paused = bool(paused)
            if time_scale is not None:
                try:
                    self.time_scale = float(min(20.0, max(0.25, float(time_scale))))
                except (TypeError, ValueError):
                    pass
            return dict(ok=True, paused=self.paused, time_scale=self.time_scale)

    # ---------------- iş parçacığı ----------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ah1s-live-flight", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self):
        next_wall = time.perf_counter()
        rate_t, rate_n = time.perf_counter(), 0
        while not self._stop.is_set():
            try:
                with self.lock:
                    reset = self._pending_reset
                    self._pending_reset = None
                if reset is not None:
                    self._do_reset(reset)
                    next_wall = time.perf_counter()
                    continue
                with self.lock:
                    idle = self.paused or self._done
                    scale = self.time_scale
                if idle:
                    time.sleep(0.03)
                    next_wall = time.perf_counter()
                    continue
                self._step()
                rate_n += 1
                now = time.perf_counter()
                if now - rate_t >= 1.0:
                    with self.lock:
                        self.sim_rate = rate_n * CONTROL_DT / (now - rate_t)
                    rate_t, rate_n = now, 0
                next_wall += CONTROL_DT / scale
                delay = next_wall - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:                           # geride kaldıysa yakalamaya çalışma
                    next_wall = time.perf_counter()
            except Exception as exc:                         # noqa: BLE001
                traceback.print_exc()
                with self.lock:
                    self.state, self._done = "error", True
                    self.message = f"Hata: {exc}"
                    self.events.append(dict(t=self._t(), type="error", message=str(exc)))
                time.sleep(0.2)

    def _t(self) -> float:
        return float(self.env.steps * CONTROL_DT) if self.env.fdm is not None and not self._done else (
            float(self.rows[-1][0]) if self.rows else 0.0)

    # ---------------- simülasyon (yalnızca uçuş iş parçacığında) ----------------

    def _do_reset(self, opts: dict):
        with self.lock:
            self.state, self.message = "resetting", "Yeniden başlatılıyor…"
        loc = opts.get("location") or None                 # {name, lat, lon}: uçuş bu enlem / boylamda başlar
        self.env.start_location = (float(loc["lat"]), float(loc["lon"])) if loc else None
        if self.is_to:
            options = dict(tasks=[dict(t) for t in (opts.get("tasks") or [])], episode_s=LIVE_EPISODE_S, live=True,
                           start=opts.get("start", "ground"), start_alt_ft=float(opts.get("alt", 100.0)),
                           fuel=tuple(opts.get("fuel") or (0.0, 0.0)), start_perturb=0.0)
            if self.task == "flight":
                options["start_speed_kt"] = float(opts.get("speed_kt", 60.0))
                if options["start"] == "cruise" and not options["tasks"]:
                    options["tasks"] = [dict(kind="cruise", hold=True)]    # ileri uçuşta başla: hız / heading / irtifayı tut
                phys = dict(wind_kt=float(opts.get("wind_kt", 0.0)), wind_dir_deg=float(opts.get("wind_dir", 0.0)),
                            wind_dir_relative=True, turb_level=str(opts.get("turb", "none")))
                if opts.get("gusts"):
                    phys.update(gust_rate_per_min=1.0, gust_kt=(5.0, 12.0))
                if opts.get("temp_dc"):
                    phys["delta_T_C"] = float(opts["temp_dc"])
                options["physics"] = phys
        else:
            options = dict(commands=[], episode_s=LIVE_EPISODE_S, start_alt_ft=float(opts["alt"]),
                           start_speed_fps=float(opts["speed"]))
        if opts.get("heading") is not None:
            options["start_heading_deg"] = float(opts["heading"])
        obs, info = self.env.reset(seed=int(opts.get("seed", 0)), options=options)
        f = self.env.fdm
        lat0, lon0 = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
        ground = float(f["position/h-sl-ft"]) - float(f["position/h-agl-ft"])
        with self.lock:
            self.start_opts = {k: v for k, v in opts.items() if k != "tasks"}    # görev listesi yeniden başlatmada tekrarlanmasın
            self._obs = obs
            self._lat0, self._lon0, self._ground_msl = lat0, lon0, ground
            self.flight_id += 1
            self.rows = [self._row(info, np.zeros(4), np.zeros(4), self.env.trim, 0.0)]
            self.commands, self.events, self._pending_cmds = [], [], []
            self.meta = dict(model=self.policy_path.name, env_config=self.env_config,
                             start={k: v for k, v in opts.items() if k != "tasks"},
                             env_overrides=dict(self.env_overrides),
                             weight_lbs=info.get("weight_lbs"), cg_x_in=(info.get("setup") or {}).get("cg_x_in"),
                             lat0_deg=math.degrees(lat0), lon0_deg=math.degrees(lon0), ground_msl_ft=ground,
                             setup=dict(mode=info["setup"].get("mode"), heading_deg=info["heading_deg"]),
                             created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            if self.task == "flight":
                self.meta.update(physics=dict(self.env.ext.params), fuel0_lbs=info.get("fuel_lbs"),
                                 tol_scale=float(self.env.tol_scale))
            if self.is_to:
                where = ("yerde (rotor warm-up sonrası)" if opts.get("start", "ground") == "ground"
                         else f"ileri uçuşta {info.get('airspeed_kt', 0):.0f} kt, {info['altitude']:.0f} ft"
                         if opts.get("start") == "cruise" else f"havada {info['altitude']:.0f} ft hover")
                msg = f"Başladı: {where}, heading {info['heading_deg']:.0f}°, {info.get('weight_lbs', 0):.0f} lbs"
                if self.task == "flight":
                    pp = self.env.ext.params
                    msg += (f", rüzgâr {pp.get('wind_kt', 0):.0f} kt ({pp.get('wind_from_deg', 0):.0f}°'den)"
                            if pp.get("wind_kt") else ", rüzgâr yok")
                    if pp.get("turb_level", "none") != "none":
                        msg += f", türbülans {pp['turb_level']}"
            else:
                msg = f"Başladı: {opts['alt']:.0f} ft, {opts['speed']:.0f} ft/s, heading {info['heading_deg']:.0f}°"
            self.events.append(dict(t=0.0, type="start", message=msg))
            self.termination = None
            self._done = False
            self.state, self.message = "running", "Uçuyor — komut verebilirsin."

    def _step(self):
        env = self.env
        with self.lock:
            pending, self._pending_cmds = self._pending_cmds, []
        for d in pending:
            if self.is_to:
                task = dict(d["task"])
                if task.get("kind") == "recover" and task.get("dist") == "push":
                    ps = math.radians(env._state()["psi_deg"])      # burun eksenine göre itki → kuzey / doğu
                    fw, rt = float(task.pop("fwd", 0.0)), float(task.pop("right", 0.0))
                    task.update(dn=fw * math.cos(ps) - rt * math.sin(ps), de=fw * math.sin(ps) + rt * math.cos(ps))
                if self.task == "flight" and task.get("kind") != "recover":
                    # 2026-10-01: rejime göre güvenli görev(ler) (ileri uçuşta hover görevi → önce duruş, hover'da cruise Δ →
                    # dönüş / irtifa / hızlanma; doğal sınırlara kırpma) — flight_commands
                    if task.get("kind") == "komut":
                        res = route_command(env, **{k: v for k, v in task.items() if k != "kind"})
                    else:
                        res = route_task(env, task)
                    if res.ok:
                        apply_route(env, res)
                    with self.lock:
                        self.route_seq = getattr(self, "route_seq", 0) + 1
                        self.route_msg = dict(id=self.route_seq, ok=bool(res.ok), message=res.message or "Görev verildi.",
                                              tasks=[t2.get("kind") for t2 in res.tasks], regime=res.regime)
                        self.events.append(dict(t=self._t(), type="route", ok=bool(res.ok), message=res.message))
                    continue
                env.queue_task(task)
                continue
            axes_d = {a: d[a] for a in AXES}
            t_cmd = env.queue_command(axes_d, d.get("T")) if self.task == "maneuver" else env.queue_command(axes_d)
            with self.lock:
                self.commands.append(dict(id=len(self.commands) + 1, t=t_cmd, requested=axes_d, T_req=d.get("T"),
                                          T=None, deadline=None, applied=None,
                                          ref=None, start=None, verdict=None, coupling_ok=None,
                                          final_err=None, max_abs_err=None, streak_s=0.0))
        action = np.asarray(self.policy(self._obs), dtype=np.float64).reshape(-1)[:4]
        obs, reward, term, trunc, info = env.step(action)
        with self.lock:
            self._obs = obs
            self.rows.append(self._row(info, np.clip(action, -1, 1), env.filt, info["controls"], reward))
            self._sync_windows_takeoff() if self.is_to else self._sync_windows()
            if term or trunc:
                self._done = True
                self.termination = info.get("termination") or ("time_limit" if trunc else "?")
                for k, r in enumerate(info.get("command_results") or []):
                    if k < len(self.commands):
                        self.commands[k].update(verdict=bool(r["success"]), coupling_ok=r.get("coupling_ok"),
                                                final_err=r.get("final_err"), max_abs_err=r.get("max_abs_err"),
                                                settle_s=r.get("settle_s"), on_time=r.get("on_time"),
                                                interrupted=bool(r.get("interrupted")))
                        if self.is_to:
                            self.commands[k].update(max_err=r.get("max_abs_err"), touchdown_vs=r.get("touchdown_vs"))
                self.state = "done"
                self.message = f"Uçuş bitti: {self.termination}"
                self.events.append(dict(t=float(info["t"]), type="end", reason=self.termination,
                                        message=self.message))

    def _sync_windows(self):
        """env.windows ↔ komut kayıtları (aynı sırayla açılıyorlar)."""
        env = self.env
        for k, w in enumerate(env.windows):
            if k >= len(self.commands):
                break
            c = self.commands[k]
            if c["applied"] is None:                          # pencere bu adımda açıldı
                c.update(applied=dict(w["cmd"]), ref=dict(env.ref), start=dict(env.cmd_start), t=float(w["t"]),
                         T=w.get("T"), deadline=w.get("deadline"), allow_s=w.get("allow_s"),
                         eff=dict(w["eff"]) if w.get("eff") else None)
                c["ref"]["heading_wrapped"] = env.ref["heading"] % 360.0
                if any(abs(c["applied"][a] - c["requested"][a]) > 1e-9 for a in AXES):
                    self.events.append(dict(t=float(w["t"]), type="adjusted", command=c["id"], message=(
                        f"K{c['id']}: güvenli aralık için Δ değiştirildi → " + ", ".join(
                            f"{a} {c['applied'][a]:+g}" for a in AXES if c['applied'][a] != c['requested'][a]))))
            c["streak_s"] = float(w["streak"])
            c["max_abs_err"] = dict(w["max_abs_err"])
            if w["success"] is not None and c["verdict"] is None:   # pencere kapandı → env'in kararı
                c.update(verdict=bool(w["success"]), coupling_ok=w.get("coupling_ok"), final_err=w.get("final_err"),
                         settle_s=w.get("settle_s"), on_time=w.get("on_time"), interrupted=bool(w.get("interrupted")))

    def _row(self, info: dict, action, filt, controls, reward) -> list:
        if self.is_to:                                     # konum env'in pad çerçevesinde (doğu / kuzey ft)
            tg = info["target"]
            vals = [info["t"], info["east"], info["north"], info["altitude"], info["heading_deg"], info["speed"],
                    info["lateral_speed"], info["vertical_speed"], info["roll_deg"], info["pitch_deg"],
                    info["yaw_rate_dps"], info["rotor_rpm"], tg["psi"] % 360.0, 0.0, tg["h"],
                    info["err_heading"], info["err_xy"], info["err_altitude"],
                    *[float(v) for v in action], *[float(v) for v in filt], *[float(v) for v in controls], reward,
                    None, None, tg["e"], tg["n"], info["wow"], info["weight_on_skids"]]
            if self.task == "flight":
                f = self.env.fdm
                vals += [info["airspeed_kt"], info["u_target_kt"] if info.get("cruise") else None, info["trim_speed_kt"],
                         float(f["atmosphere/total-wind-north-fps"]), float(f["atmosphere/total-wind-east-fps"]),
                         info["torque_psi"], info["fuel_lbs"], info["lateral_airspeed"], info["err_speed"],
                         1.0 if info.get("cruise") else 0.0, info.get("course_deg")]
            vals += [None] * (len(COLUMNS) - len(vals))
            return [round(float(v), nd) if _finite(v) else None for v, (_, nd) in zip(vals, COLUMNS)]
        f = self.env.fdm
        lat, lon = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
        x = (lon - self._lon0) * math.cos(self._lat0) * EARTH_RADIUS_FT
        y = (lat - self._lat0) * EARTH_RADIUS_FT
        ref = info["ref"]
        vals = [info["t"], x, y, info["altitude"], info["heading_deg"], info["speed"], info["lateral_speed"],
                info["vertical_speed"], info["roll_deg"], info["pitch_deg"], info["yaw_rate_dps"], info["rotor_rpm"],
                ref["heading"] % 360.0, ref["speed"], ref["altitude"],
                info["err_heading"], info["err_speed"], info["err_altitude"],
                *[float(v) for v in action], *[float(v) for v in filt], *[float(v) for v in controls], reward,
                *(info.get("att_cmd_deg") or (None, None))]
        vals += [None] * (len(COLUMNS) - len(vals))
        return [round(float(v), nd) if _finite(v) else None for v, (_, nd) in zip(vals, COLUMNS)]

    def _status(self) -> dict:
        return dict(state=self.state, message=self.message, paused=self.paused, time_scale=self.time_scale,
                    t=self.rows[-1][0] if self.rows else 0.0, termination=self.termination, flight=self.flight_id,
                    sim_rate=round(self.sim_rate, 2), pending=len(self._pending_cmds),
                    route=getattr(self, "route_msg", None))

    # ---------------- görev (kayıt) ----------------

    def run_mission(self, start: dict, schedule: list, duration_s: float) -> dict:
        """Gerçek zamansız: aynı _step yolu, komutlar zamanında kuyruğa girer.
        Kalkış görevi: schedule = görev listesi; env sırayla verir (biri bitince sıradaki, `_frac` → kesme)."""
        if self.is_to:
            self._do_reset(dict(self.start_opts, **start, tasks=[dict(d) for d in schedule]))
            while not self._done and self.env.steps * CONTROL_DT < duration_s - 1e-9:
                self._step()
            with self.lock:
                for w in self.env.windows:
                    self.env._close_window(w)
                self._sync_windows_takeoff()
                if not self._done:
                    self.state, self.message = "done", "Görev süresi doldu."
                    self.events.append(dict(t=self._t(), type="end", reason="mission_end", message=self.message))
            return self.export_dict()
        self._do_reset(dict(self.start_opts, **start))
        sched = sorted(((float(t), d) for t, d in schedule), key=lambda x: x[0])
        i = 0
        while not self._done and self.env.steps * CONTROL_DT < duration_s - 1e-9:
            t_now = self.env.steps * CONTROL_DT
            while i < len(sched) and t_now >= sched[i][0] - 1e-9:
                d = {a: float(sched[i][1].get(a, 0.0)) for a in AXES}
                if sched[i][1].get("T"):
                    d["T"] = float(sched[i][1]["T"])
                self._pending_cmds.append(d)
                i += 1
            self._step()
        with self.lock:
            for w in self.env.windows:                        # kalan pencereleri env kuralıyla kapat
                self.env._close_window(w)
            self._sync_windows()
            if not self._done:
                self.state, self.message = "done", "Görev süresi doldu."
                self.events.append(dict(t=self._t(), type="end", reason="mission_end", message=self.message))
        return self.export_dict()

    # ---------------- dışa aktarma ----------------

    def export_dict(self, every: int = 1, title: str | None = None, description: str | None = None) -> dict:
        with self.lock:
            rows = self.rows[::max(1, int(every))]
            if self.rows and rows[-1] is not self.rows[-1]:
                rows = rows + [self.rows[-1]]
            data = {c: [r[i] for r in rows] for i, c in enumerate(COLS)}
            t = np.asarray([r[0] for r in self.rows], dtype=np.float64)
            err = {a: np.asarray([np.nan if r[COLS.index(AXIS_ERR_COL[a])] is None else r[COLS.index(AXIS_ERR_COL[a])]
                                  for r in self.rows], dtype=np.float64) for a in AXES}
            cmds = []
            for k, c in enumerate(self.commands if not self.is_to else []):
                if c["applied"] is None:
                    continue
                t_end = self.commands[k + 1]["t"] if k + 1 < len(self.commands) else float(t[-1])
                last = k + 1 == len(self.commands)
                m = command_metrics(t, err, c.get("eff") or c["applied"], c["t"], t_end, self.limits,
                                    terminated=bool(last and self.termination and self.termination != "time_limit"),
                                    deadline=c.get("deadline"))
                cmds.append(dict(c, metrics=m))
            if self.is_to:                               # görev sonuçları env'den (bant / tutma / kuplaj / temas)
                cmds = [dict(c) for c in self.commands]
            return _clean(dict(format=FORMAT, title=title, description=description, cols=COLS,
                               control_dt=CONTROL_DT, every=int(every), meta=self.meta, limits=self.limits,
                               termination=self.termination, commands=cmds, events=self.events, data=data))

    def export_acmi(self) -> str:
        """Tacview ACMI 2.1 metni (T = boylam | enlem | irtifa m | roll | pitch | yaw)."""
        with self.lock:
            rows = list(self.rows)
            meta, cmds = dict(self.meta), list(self.commands)
            lat0, lon0, ground = self._lat0, self._lon0, self._ground_msl
        ref_time = meta.get("created", "2026-01-01T00:00:00+00:00").replace("+00:00", "Z")
        out = ["FileType=text/acmi/tacview", "FileVersion=2.1",
               f"0,ReferenceTime={ref_time}", "0,DataSource=JSBSim AH-1S + PPO (ah1s-rl-project)",
               f"0,Title=AH-1S komut uçuşu ({meta.get('model', '?')})",
               "a01,Type=Air+Rotorcraft,Name=AH-1S,Pilot=PPO,Color=Blue"]
        ix, iy, ih = COLS.index("x"), COLS.index("y"), COLS.index("h")
        ip, it, ips = COLS.index("phi"), COLS.index("theta"), COLS.index("psi")
        next_cmd = 0
        for r in rows:
            if any(r[i] is None for i in (ix, iy, ih, ip, it, ips)):
                continue
            lat = math.degrees(lat0 + r[iy] / EARTH_RADIUS_FT)
            lon = math.degrees(lon0 + r[ix] / (EARTH_RADIUS_FT * math.cos(lat0)))
            alt_m = (ground + r[ih]) * 0.3048
            out.append(f"#{r[0]:.3f}")
            while next_cmd < len(cmds) and cmds[next_cmd]["t"] <= r[0] + 1e-9:
                c = cmds[next_cmd]
                if c.get("task"):
                    txt = task_text(c["task"])
                else:
                    txt = " ".join(f"{a}{(c['applied'] or c['requested'])[a]:+g}" for a in AXES
                                   if (c["applied"] or c["requested"])[a])
                out.append(f"0,Event=Message|a01|K{c['id']} {txt}")
                next_cmd += 1
            out.append(f"a01,T={lon:.7f}|{lat:.7f}|{alt_m:.2f}|{r[ip]:.2f}|{r[it]:.2f}|{r[ips]:.2f}")
        return "\n".join(out) + "\n"

    def save(self, directory: str | Path = ".") -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pj = directory / f"ah1s_flight_{stamp}.json"
        pa = directory / f"ah1s_flight_{stamp}.acmi"
        pj.write_text(json.dumps(self.export_dict(), separators=(",", ":")), encoding="utf-8")
        pa.write_text(self.export_acmi(), encoding="utf-8")
        return dict(ok=True, json=str(pj.resolve()), acmi=str(pa.resolve()))


# =====================================================================
# SAYFA
# =====================================================================

def inline_local_modules(html: str) -> str:
    """Sayfadaki `import { … } from "./<ad>.js";` satırlarını (viz/ altındaki modüller) modüllerin kendisiyle değiştirir.

    Colab sayfayı satır içi gösterir, yanında dosya sunamaz; yerel sunucu da aynı sayfayı verir. Her modül kendi
    `import * as THREE` satırı olmadan, sayfanın THREE'si ile ayrı bir kapsamda çalışır; adları sayfayla çakışmaz.
    """
    def repl(m: re.Match) -> str:
        names = ", ".join(n.strip() for n in m.group(1).split(",") if n.strip())
        path = VIZ_DIR / m.group(2)
        src = path.read_text(encoding="utf-8")
        src, n_three = re.subn(r'^import \* as THREE from "[^"]+";\n', "", src, flags=re.M)
        if n_three != 1 or re.search(r"^import ", src, flags=re.M):
            raise RuntimeError(f"{path.name}: beklenen tek import three.js olmalı")
        src = re.sub(r"^export (const|class|function) ", r"\1 ", src, flags=re.M)
        return f"const {{ {names} }} = (() => {{\n{src}\nreturn {{ {names} }};\n}})();"
    return re.sub(r'^import \{([^}]*)\} from "\./([\w-]+\.js)";$', repl, html, flags=re.M)


def page_fragment(boot: dict | None = None) -> str:
    html = inline_local_modules(PAGE_FILE.read_text(encoding="utf-8"))
    if boot:
        html = f"<script>window.AH1S_VIZ_BOOT = {json.dumps(boot)};</script>\n" + html
    return html


# CDN → yerel kopya (viz/vendor; aynı sürümler). Yerel sunucu sayfayı bunlarla verir: masaüstü penceresi / tarayıcı
# internetsiz de açılır. Colab satır içi sayfası CDN'den yükler (yanında dosya sunulamaz).
VENDOR = {
    "https://cdn.jsdelivr.net/npm/three@0.169.0/+esm": "./vendor/three.module.min.js",
    "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/controls/OrbitControls.js/+esm": "./vendor/OrbitControls.js",
    "https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js": "./vendor/uPlot.iife.min.js",
}


def page_document(boot: dict | None = None, local_vendor: bool = True) -> str:
    html = page_fragment(boot)
    if local_vendor:
        for cdn, local in VENDOR.items():
            if (VIZ_DIR / local.removeprefix("./")).is_file():
                html = html.replace(cdn, local)
    return ("<!doctype html>\n<html lang=\"tr\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + html + "\n</body>\n</html>\n")


# =====================================================================
# YEREL HTTP SUNUCU (yalnızca standart kütüphane)
# =====================================================================

STATIC_TYPES = {".json": "application/json", ".js": "text/javascript",
                ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".acmi": "text/plain"}


def make_handler(flight: LiveFlight):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AH1SCommandViz/1.0"

        def log_message(self, fmt, *args):             # sessiz
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, separators=(",", ":")).encode("utf-8"), "application/json")

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            path = url.path
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, page_document(dict(transport="http")).encode("utf-8"),
                                      "text/html; charset=utf-8")
                if path == "/api/hello":
                    return self._json(flight.hello())
                if path == "/api/state":
                    fid = q.get("flight", [None])[0]
                    return self._json(flight.state_since(None if fid in (None, "", "null") else int(fid),
                                                         int(q.get("since", ["0"])[0] or 0)))
                if path == "/api/export.json":
                    body = json.dumps(flight.export_dict(), separators=(",", ":")).encode("utf-8")
                    return self._send(200, body, "application/json",
                                      {"Content-Disposition": "attachment; filename=ah1s_flight.json"})
                if path == "/api/export.acmi":
                    return self._send(200, flight.export_acmi().encode("utf-8"), "text/plain; charset=utf-8",
                                      {"Content-Disposition": "attachment; filename=ah1s_flight.txt.acmi"})
                name = path.lstrip("/")
                target = (VIZ_DIR / name).resolve()
                if (VIZ_DIR.resolve() in target.parents and target.is_file()
                        and target.suffix in STATIC_TYPES):
                    return self._send(200, target.read_bytes(), STATIC_TYPES[target.suffix])
                return self._json(dict(ok=False, message="bulunamadı"), 404)
            except Exception as exc:                      # noqa: BLE001
                traceback.print_exc()
                return self._json(dict(ok=False, message=str(exc)), 500)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if path == "/api/command":
                    return self._json(flight.request_command(body))
                if path == "/api/reset":
                    return self._json(flight.request_reset(**body))
                if path == "/api/control":
                    return self._json(flight.set_control(**body))
                return self._json(dict(ok=False, message="bulunamadı"), 404)
            except Exception as exc:                      # noqa: BLE001
                return self._json(dict(ok=False, message=str(exc)), 400)

    return Handler


def serve(flight: LiveFlight, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False):
    srv = ThreadingHTTPServer((host, port), make_handler(flight))
    srv.daemon_threads = True
    flight.start()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    print(f"AH-1S komut uçuşu: {url}   (model: {flight.policy_path.name}; durdurmak için Ctrl+C)", flush=True)
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:                                 # noqa: BLE001
            pass
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        flight.stop()
        srv.server_close()


# =====================================================================
# COLAB (satır içi; kernel callback'leri — localhost / paylaşım linki yok)
# =====================================================================

def colab(model: str | Path = DEFAULT_POLICY, env_config: str = "v2", start: dict | None = None,
          save_dir: str | Path = "/content/ah1s_flights") -> LiveFlight:
    try:
        from google.colab import output
        from IPython.display import HTML, JSON, display
    except ImportError as exc:
        raise RuntimeError("colab() yalnızca Colab içinde çalışır; yerelde `python command_viz.py`.") from exc
    import builtins
    old = getattr(builtins, "_ah1s_command_viz", None)
    if old is not None:
        try:
            old.stop()
        except Exception:                                 # noqa: BLE001
            pass
    flight = LiveFlight(model, env_config=env_config, start=start)
    builtins._ah1s_command_viz = flight
    output.register_callback("ah1s_viz.hello", lambda: JSON(flight.hello()))
    output.register_callback("ah1s_viz.state", lambda fid=None, since=0: JSON(flight.state_since(fid, since)))
    output.register_callback("ah1s_viz.command", lambda d=None: JSON(flight.request_command(d or {})))
    output.register_callback("ah1s_viz.reset", lambda d=None: JSON(flight.request_reset(**(d or {}))))
    output.register_callback("ah1s_viz.control", lambda d=None: JSON(flight.set_control(**(d or {}))))
    output.register_callback("ah1s_viz.save", lambda: JSON(flight.save(save_dir)))
    flight.start()
    display(HTML(page_fragment(dict(transport="colab"))))
    return flight


# =====================================================================
# KAYITLI DEMO UÇUŞLARI
# =====================================================================

# policy="maneuver" → manevra modeli (MANEUVER_POLICY ya da `record --maneuver-model`); policy="robust" → dayanıklı
# model (ROBUST_POLICY ya da `record --robust-model`); diğerleri models_command_curriculum/ içindeki komut modelleri.
# Her model kendi env ayarıyla uçar (ör. dayanıklı model collective ±0.45). Manevra görevlerinde süre hedefi (T) verilmezse
# env onu M5 çevikliğinden hesaplar (55° yatış, 25°/s pedal, 6 ft/s² ivme, 20 ft/s tırmanış, +2 s tepki).
GOREV6 = [(5, {"heading": 90}), (40, {"speed": 8}), (75, {"altitude": 100}),
          (110, {"heading": -45, "altitude": -60}), (145, {"speed": -10}), (180, {"heading": 180})]
TO = lambda h: dict(kind="takeoff", h=float(h))                                     # noqa: E731
ACC = lambda kt, dh: dict(kind="cruise", u_kt=float(kt), dh=float(dh), accel=True)  # noqa: E731
CRZ = lambda **k: dict(kind="cruise", **{a: float(b) for a, b in k.items()})         # noqa: E731
FL_CHAIN = [TO(40), ACC(70, 150), CRZ(dpsi=90), CRZ(du_kt=-20, dh=100), dict(kind="stop", decel=2.5), dict(kind="land")]
MISSIONS = [
    # ---- tek ajan (README 32): kalkış → ileri uçuşa geçiş → Δ komutlar → duruş → iniş; rüzgâr / türbülans / yakıt ----
    dict(id="fl_zincir_sakin", title="Tek ajan: yerden kalkış → 70 kt → dönüş → yavaşla / tırman → dur → in (sakin)",
         policy="flight",
         description="Rotor warm-up'tan sonra 40 ft hover, 70 kt'a hızlanırken +150 ft, +90° dönüş, −20 kt ile +100 ft, "
                     "duruş (hover) ve pad'e iniş. Tork cezası, güç tavanı (56 psi) ve yakıt tüketimi açık.",
         start=dict(start="ground", heading=0.0, fuel=[300.0, 300.0], wind_kt=0.0, turb="none"), duration=420.0,
         schedule=FL_CHAIN),
    dict(id="fl_zincir_ruzgar", title="Tek ajan: aynı zincir, 15 kt rüzgâr (sağ ön) + hafif türbülans",
         policy="flight",
         description="Aynı görev zinciri; 15 kt rüzgâr başlangıç heading'ine göre 45° sağdan, hafif türbülans (Dryden, W20 "
                     "15 kt). Rüzgârın kendisi gözlemde yok — ajan hava / yer hızı farkından görüyor.",
         start=dict(start="ground", heading=0.0, fuel=[300.0, 300.0], wind_kt=15.0, wind_dir=45.0, turb="light"),
         duration=420.0, schedule=FL_CHAIN),
    dict(id="fl_ileri_ucus", title="Tek ajan: ileri uçuşta Δ komutlar (80 kt, 400 ft), duruş ve iniş",
         policy="flight",
         description="80 kt'ta: +20 kt, −180° dönüş, +200 ft, birleşik (−30 kt, +60°, −150 ft); sonra dur ve in.",
         start=dict(start="cruise", alt=400.0, speed_kt=80.0, heading=0.0, fuel=[300.0, 300.0]), duration=420.0,
         schedule=[dict(kind="cruise", hold=True), CRZ(du_kt=20), CRZ(dpsi=-180), CRZ(dh=200),
                   CRZ(du_kt=-30, dpsi=60, dh=-150), dict(kind="stop", decel=2.5), dict(kind="land")]),
    dict(id="fl_turb_agir", title="Tek ajan: ağır (9700 lbs), 20 kt rüzgâr + orta türbülans: hover → 80 kt → dur → in",
         policy="flight",
         description="100 ft hover'dan 80 kt'a (+100 ft), +90°, duruş ve iniş; 20 kt arkadan-yandan rüzgâr, orta türbülans "
                     "ve gust'lar. Başarı bantları orta türbülansta 2× (ADS-33 'yeterli' gibi).",
         start=dict(start="hover", alt=100.0, heading=0.0, fuel=[600.0, 600.0], wind_kt=20.0, wind_dir=150.0,
                    turb="moderate", gusts=True), duration=420.0,
         schedule=[dict(kind="hold"), ACC(80, 100), CRZ(dpsi=90), dict(kind="stop", decel=2.5), dict(kind="land")]),
    # ---- kalkış görevi (4 kumanda doğrudan, README 30): görevler sırayla, biri bitince (bant + tutma) sıradaki ----
    dict(id="to_kalkis_300", title="Kalkış: yerden 300 ft hover (dört kumanda doğrudan)", policy="takeoff",
         description="Rotor warm-up'tan sonra yerden 300 ft'e; pad'in üstünde, heading'i koruyarak hover. Tork değişimini pedal, "
                     "tail rotor itkisinin yana kaydırmasını yanal cyclic, burnu boylamsal cyclic karşılıyor.",
         start=dict(start="ground", heading=0.0, fuel=[0.0, 0.0]), duration=48.0, schedule=[TO(300)]),
    dict(id="to_hover_manevra", title="Hover manevraları: 360° dönüş, sağa / sola kayma, bob-up / down (40 ft)", policy="takeoff",
         description="Kalkış 40 ft → yerinde +360° (pirouette) → 40 ft sağa → 40 ft sola → +30 ft → −30 ft; her görev hover "
                     "bandında bitince sıradaki.",
         start=dict(start="ground", heading=0.0, fuel=[0.0, 0.0]), duration=120.0,
         schedule=[TO(40), dict(kind="turn", dpsi=360.0), dict(kind="move", dx=0.0, dy=40.0), dict(kind="move", dx=0.0, dy=-40.0),
                   dict(kind="bob", dh=30.0), dict(kind="bob", dh=-30.0)]),
    dict(id="to_hedef", title="Hedef değişikliği: 800 ft'e tırmanırken 300 ft'te dur", policy="takeoff",
         description="Kalkış 800 ft; süre hedefinin %30'unda yeni hedef 300 ft (ölçülen duruma göre, önceki görev «kesildi»).",
         start=dict(start="ground", heading=0.0, fuel=[0.0, 0.0]), duration=60.0,
         schedule=[TO(800), dict(kind="climb_to", h=300.0, _frac=0.3)]),
    dict(id="to_inis", title="Kalkış → 40 ft ileri kayma → iniş", policy="takeoff",
         description="Kalkış 50 ft, 40 ft ileri, sonra o noktaya yumuşak iniş: temas ≤ 4 ft/s, collective indirilir.",
         start=dict(start="ground", heading=0.0, fuel=[0.0, 0.0]), duration=75.0,
         schedule=[TO(50), dict(kind="move", dx=40.0, dy=0.0), dict(kind="land")]),
    dict(id="to_agir_bozucu", title="Tam yakıt (10280 lbs): hover'da bozucular, iniş", policy="takeoff",
         description="Brüt ağırlık 10280 lbs (iki tank dolu). 60 ft hover'da yanal cyclic darbesi, 10 ft/s yana itki, +10° "
                     "yatış bozucusu; her birinden sonra hover'a dönüş, en sonda iniş.",
         start=dict(start="ground", heading=0.0, fuel=[890.0, 890.0]), duration=130.0,
         schedule=[TO(60), dict(kind="recover", dist="kick", axis=2, mag=0.3, dur=0.8),
                   dict(kind="recover", dist="push", dn=0.0, de=10.0, dw=0.0),
                   dict(kind="recover", dist="tilt", dphi=10.0, dtheta=0.0), dict(kind="land")]),
    dict(id="to_inis_300", title="300 ft hover'dan pad'e iniş", policy="takeoff",
         description="300 ft'te hover; iniş: ~5 ft/s alçalma, 20 ft'ten itibaren yavaşlama (profil), yumuşak temas, "
                     "collective tam aşağı, dört kızak noktası yerde ve ağırlık kızaklarda.",
         start=dict(start="hover", alt=300.0, heading=0.0, fuel=[0.0, 0.0]), duration=100.0,
         schedule=[dict(kind="hold"), dict(kind="land")]),
    dict(id="to_1000", title="Kalkış: yerden 1000 ft hover", policy="takeoff",
         description="Rotor warm-up'tan sonra 1000 ft'e (süre hedefi 12 ft/s tırmanışla) ve pad'in üstünde hover.",
         start=dict(start="ground", heading=0.0, fuel=[0.0, 0.0]), duration=100.0, schedule=[TO(1000)]),
    # ---- dayanıklılık (collective ±0.45 + S4 ince ayarlı model, README 29): manevra bitmeden gelen / ters komutlar,
    #      zarf sınırları ----
    dict(id="rob_slalom_ters", title="Dayanıklılık: slalomda ani ters dönüşler (60 kt)", policy="robust",
         description="Her dönüş bitmeden ters yöne: +90°, 3 s sonra −90°, … Önceki komut «kesildi» sayılır; yeni Δ ölçülen "
                     "heading'e göre, süre hedefi ters yöndeki dönüşü durdurma payıyla.",
         start=dict(alt=800.0, speed=101.3, heading=0.0), duration=42.0,
         schedule=[(5, {"heading": 90}), (8, {"heading": -90}), (11, {"heading": 90}), (14, {"heading": -90}),
                   (17, {"heading": 45})]),
    dict(id="rob_donus_tirmanis", title="Dayanıklılık: 180° dönüş ortasında ani tırmanış, sonra fren (60 kt)",
         policy="robust",
         description="+180° dönüş sürerken +150 ft (dönüş devam eder), ardından −30 ft/s. Verilmeyen eksen önceki hedefine devam eder.",
         start=dict(alt=800.0, speed=101.3, heading=0.0), duration=45.0,
         schedule=[(5, {"heading": 180}), (9, {"altitude": 150}), (13, {"speed": -30})]),
    dict(id="rob_testere", title="Dayanıklılık: ±180° testere ve dikey ters (60 kt)", policy="robust",
         description="+180°, 3 s sonra −180°, 3 s sonra +180°; ardından +150 ft tırmanırken 3 s sonra −200 ft.",
         start=dict(alt=800.0, speed=101.3, heading=0.0), duration=58.0,
         schedule=[(5, {"heading": 180}), (8, {"heading": -180}), (11, {"heading": 180}), (28, {"altitude": 150}),
                   (31, {"altitude": -200})]),
    dict(id="rob_sinir", title="Zarf sınırı: 100 kt'ta ters dönüş, hover'a kadar fren, hover'da pedal dönüşü",
         policy="robust",
         description="100 kt: +180°, 4 s sonra −180°; üç adımda −50 ft/s ile 100 kt → ~5 kt; hover'da +180° pedal dönüşü.",
         start=dict(alt=900.0, speed=168.0, heading=0.0), duration=108.0,
         schedule=[(5, {"heading": 180}), (9, {"heading": -180}), (34, {"speed": -50}), (52, {"speed": -50}),
                   (70, {"speed": -50}), (88, {"heading": 180})]),
    dict(id="rob_alcak", title="Zarf sınırı: alçak irtifada tırmanıp ani dalış, 250 ft'te dönüşler (30 kt)",
         policy="robust",
         description="400 ft: +100 ft, 3 s sonra −200 ft (≈ 250 ft taban); orada +180°, 4 s sonra −90°.",
         start=dict(alt=400.0, speed=50.6, heading=0.0), duration=52.0,
         schedule=[(5, {"altitude": 100}), (8, {"altitude": -200}), (26, {"heading": 180}), (30, {"heading": -90})]),
    dict(id="rob_slalom_ters_m5", title="Karşılaştırma — ince ayar öncesi (M5) model: aynı ani ters dönüşler",
         policy="maneuver",
         description="Dayanıklılık eğitiminden önceki model (collective ±0.25), aynı slalom komutlarıyla (ilk uçuşla karşılaştır).",
         start=dict(alt=800.0, speed=101.3, heading=0.0), duration=42.0,
         schedule=[(5, {"heading": 90}), (8, {"heading": -90}), (11, {"heading": 90}), (14, {"heading": -90}),
                   (17, {"heading": 45})]),
    dict(id="man_donusler60", title="Manevra: 60 kt'ta yatışlı dönüşler", policy="maneuver",
         description="800 ft, 60 kt: +90°, −180°, +90°, −45°. Süre hedefi M5 çevikliğinden (55° yatışa kadar).",
         start=dict(alt=800.0, speed=101.3, heading=0.0), duration=90.0,
         schedule=[(5, {"heading": 90}), (25, {"heading": -180}), (50, {"heading": 90}), (70, {"heading": -45})]),
    dict(id="man_slalom", title="Manevra: slalom (60 kt)", policy="maneuver",
         description="Ardışık kısa süreli dönüşler: +45°, −90°, +90°, −90°, +45° (her biri ≈ 4–6 s'lik süre hedefiyle).",
         start=dict(alt=800.0, speed=101.3, heading=0.0), duration=72.0,
         schedule=[(5, {"heading": 45}), (16, {"heading": -90}), (29, {"heading": 90}), (42, {"heading": -90}),
                   (55, {"heading": 45})]),
    dict(id="man_dusukhiz", title="Manevra: düşük hızda çeviklik (5 kt)", policy="maneuver",
         description="Neredeyse askıda: 180° pedal dönüşü, bob-up / bob-down (±100 ft), hızlanma ve ani duruş (±40 ft/s), −90°.",
         start=dict(alt=800.0, speed=8.4, heading=0.0), duration=120.0,
         schedule=[(5, {"heading": 180}), (25, {"altitude": 100}), (42, {"altitude": -100}), (60, {"speed": 40}),
                   (80, {"speed": -40}), (100, {"heading": -90})]),
    dict(id="man_hiz_irtifa", title="Manevra: hızlanma, yavaşlama, irtifa (60 kt)", policy="maneuver",
         description="+50 ft/s, −50 ft/s, +150 ft, −150 ft ve yavaşlarken tırmanma (−40 ft/s ile +100 ft birlikte).",
         start=dict(alt=800.0, speed=101.3, heading=0.0), duration=110.0,
         schedule=[(5, {"speed": 50}), (27, {"speed": -50}), (49, {"altitude": 150}), (69, {"altitude": -150}),
                   (89, {"speed": -40, "altitude": 100})]),
    dict(id="man_tirmanarak_donus", title="Manevra: tırmanarak dönüş, birleşik komutlar (80 kt)", policy="maneuver",
         description="700 ft, 80 kt: +180° ile +150 ft birlikte; ardından −90° / −40 ft/s / −100 ft; sonra +45° ile +30 ft/s.",
         start=dict(alt=700.0, speed=135.0, heading=90.0), duration=85.0,
         schedule=[(5, {"heading": 180, "altitude": 150}), (35, {"heading": -90, "speed": -40, "altitude": -100}),
                   (62, {"heading": 45, "speed": 30})]),
    dict(id="man_gorev6", title="Karşılaştırma — manevra modeli: 6 komutluk görev", policy="maneuver",
         description=("Eski komut modelinin görevinin aynısı (300 ft, 15 ft/s, aynı komutlar ve zamanlar), "
                      "süre hedefi M5'ten. Bir sonraki kayıtla karşılaştır: yatış, yunuslama ve oturma süreleri."),
         start=dict(alt=300.0, speed=15.0, heading=0.0), duration=222.0, schedule=GOREV6),
    dict(id="gorev6", title="Karşılaştırma — eski komut modeli (v2): aynı görev", policy="v2_R1_final.zip",
         description="Tek uçuşta dönüş, hız, irtifa ve birleşik komutlar (300 ft, 15 ft/s, kuzeye başlangıç); süre hedefi yok.",
         start=dict(alt=300.0, speed=15.0, heading=0.0), duration=222.0, schedule=GOREV6),
    dict(id="donusler", title="Komut modeli (v2): büyük dönüşler", policy="v2_R1_final.zip",
         description="+180°, −90°, +45° ve −10°; dönüşte hız ve irtifa korunuyor mu?",
         start=dict(alt=300.0, speed=15.0, heading=0.0), duration=150.0,
         schedule=[(5, {"heading": 180}), (50, {"heading": -90}), (90, {"heading": 45}), (120, {"heading": -10})]),
    dict(id="baslangic800", title="Komut modeli (v2): 800 ft, 22 ft/s başlangıç", policy="v2_R1_final.zip",
         description="Eğitimdeki standart başlangıç dışında: birleşik komut, hız ve büyük dönüş.",
         start=dict(alt=800.0, speed=22.0, heading=45.0), duration=165.0,
         schedule=[(5, {"heading": -60, "altitude": -80}), (50, {"speed": -8}), (85, {"heading": 120}),
                   (125, {"altitude": 100})]),
    dict(id="v1_gorev6", title="Komut modeli v1 (gevşek kriter): aynı görev", policy="v1_R1_final.zip",
         description=("Gevşek kriterle eğitilen ilk model, bugünkü (v2) kriterle değerlendirildi: dönüşlerde hız "
                      "4 ft/s kuplaj sınırını aşıyor, elevator titreşiyor."),
         start=dict(alt=300.0, speed=15.0, heading=0.0), duration=222.0, schedule=GOREV6),
]


def record_missions(out: Path, every: int = 2, only: list | None = None, maneuver_model: Path | None = None,
                    robust_model: Path | None = None, takeoff_model: Path | None = None,
                    flight_model: Path | None = None) -> dict:
    flights = []
    cache = {}
    for m in MISSIONS:
        if only and m["id"] not in only:
            continue
        if m["policy"] == "maneuver":
            path = Path(maneuver_model) if maneuver_model else MANEUVER_POLICY
        elif m["policy"] == "robust":
            path = Path(robust_model) if robust_model else ROBUST_POLICY
        elif m["policy"] == "takeoff":
            path = Path(takeoff_model) if takeoff_model else TAKEOFF_POLICY
        elif m["policy"] == "flight":
            path = Path(flight_model) if flight_model else FLIGHT_POLICY
            if not path.exists():
                print(f"{m['id']:14s} atlandı: {path} yok", flush=True)
                continue
        else:
            path = REPO_ROOT / "models_command_curriculum" / m["policy"]
        key = (str(path), m.get("env_config", "v2"))
        if key not in cache:
            cache[key] = LiveFlight(path, env_config=m.get("env_config", "v2"))
        fl = cache[key]
        t0 = time.time()
        rec = fl.run_mission(m["start"], m["schedule"], m["duration"])
        d = fl.export_dict(every=every, title=m["title"], description=m["description"])
        d["id"] = m["id"]
        ok = sum(1 for c in d["commands"] if c.get("verdict"))
        print(f"{m['id']:14s} {len(rec['data']['t']):5d} satır  {ok}/{len(d['commands'])} komut başarılı  "
              f"bitiş={d['termination'] or 'görev sonu'}  ({time.time() - t0:.1f} s)", flush=True)
        flights.append(d)
    bundle = dict(format=FORMAT + "+bundle", created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  flights=flights)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"kaydedildi: {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return bundle


# =====================================================================
# CLI
# =====================================================================

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "record":
        p = argparse.ArgumentParser(prog="command_viz.py record", description="Kayıtlı demo uçuşlarını üret.")
        p.add_argument("--out", type=Path, default=VIZ_DIR / "demo_flights.json")
        p.add_argument("--every", type=int, default=2, help="her N satırdan birini yaz (1 = hepsi)")
        p.add_argument("--only", nargs="*", help="yalnızca bu görev kimlikleri")
        p.add_argument("--maneuver-model", type=Path, default=None,
                       help=f"manevra görevleri için model (varsayılan {MANEUVER_POLICY.relative_to(REPO_ROOT)})")
        p.add_argument("--robust-model", type=Path, default=None,
                       help=f"dayanıklılık görevleri için model (varsayılan {ROBUST_POLICY.relative_to(REPO_ROOT)})")
        p.add_argument("--takeoff-model", type=Path, default=None,
                       help=f"kalkış görevleri için model (varsayılan {TAKEOFF_POLICY.relative_to(REPO_ROOT)})")
        p.add_argument("--flight-model", type=Path, default=None,
                       help=f"tek ajan görevleri için model (varsayılan {FLIGHT_POLICY.relative_to(REPO_ROOT)})")
        a = p.parse_args(argv[1:])
        record_missions(a.out, a.every, a.only, a.maneuver_model, a.robust_model, a.takeoff_model, a.flight_model)
        return
    p = live_parser()
    a = p.parse_args(argv)
    serve(live_from_args(a, p), a.host, a.port, a.open)


def live_parser() -> argparse.ArgumentParser:
    """Canlı uçuşun komut satırı (main ve masaüstü uygulamasının 3D penceresi, ah1s_app.sim_window, ortak)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_POLICY,
                   help="PPO modeli (.zip); görev (komut / manevra) modelin observation boyutundan anlaşılır")
    p.add_argument("--env-config", choices=["v2", "v1"], default="v2", help="yalnızca komut görevi")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = ağdaki diğer cihazlar da erişsin")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--start", choices=["ground", "hover", "cruise"], default=None,
                   help="kalkış / uçuş görevi: yerde, havada (hover) ya da ileri uçuşta (yalnızca uçuş) başla")
    p.add_argument("--start-alt", type=float, default=None, help="ft AGL (varsayılan: manevra 800, komut 300, kalkış hover 100)")
    p.add_argument("--start-speed", type=float, default=None, help="ft/s (varsayılan: manevra 101.3 = 60 kt, komut 15)")
    p.add_argument("--start-heading", type=float, default=None, help="° (varsayılan 0)")
    p.add_argument("--start-speed-kt", type=float, default=None, help="uçuş görevi, ileri uçuşta başlangıç: hava hızı kt")
    p.add_argument("--wind-kt", type=float, default=None, help="uçuş görevi: sabit rüzgâr (kt)")
    p.add_argument("--wind-dir", type=float, default=None, help="uçuş görevi: rüzgârın geldiği yön, başlangıç heading'ine göre °")
    p.add_argument("--turb", choices=["none", "light", "moderate", "severe"], default=None, help="uçuş görevi: türbülans")
    p.add_argument("--gusts", action="store_true", help="uçuş görevi: gust'lar (dakikada ~1, 5–12 kt)")
    p.add_argument("--fuel", type=float, nargs=2, default=None, help="kalkış / uçuş görevi: tank başına yakıt (lbs)")
    p.add_argument("--temp-dc", type=float, default=None, help="uçuş görevi: standart günden sıcaklık farkı (°C)")
    p.add_argument("--lat", type=float, default=None, help="başlangıç enlemi (°); --lon ile birlikte (yoksa reset00.xml)")
    p.add_argument("--lon", type=float, default=None, help="başlangıç boylamı (°)")
    p.add_argument("--location-name", default=None, help="konumun sayfada görünen adı")
    p.add_argument("--open", action="store_true", help="tarayıcıyı aç")
    p.add_argument("--nose-heading", action="store_true",
                   help="uçuş görevi: heading komutu burun yönü (eğitimdeki gibi); varsayılan: ileri uçuşta yer izi (rota), "
                        "burun rüzgâra göre düzeltilir")
    return p


def live_from_args(a, p: argparse.ArgumentParser | None = None) -> LiveFlight:
    if (a.lat is None) != (a.lon is None):
        if p is not None:
            p.error("--lat ve --lon birlikte verilmeli")
        raise ValueError("--lat ve --lon birlikte verilmeli")
    location = dict(name=a.location_name or f"{a.lat:.3f}, {a.lon:.3f}", lat=a.lat, lon=a.lon) if a.lat is not None else None
    return LiveFlight(a.model, env_config=a.env_config,
                      start=dict(alt=a.start_alt, speed=a.start_speed, heading=a.start_heading, start=a.start,
                                 speed_kt=a.start_speed_kt, wind_kt=a.wind_kt, wind_dir=a.wind_dir, turb=a.turb,
                                 gusts=True if a.gusts else None, fuel=list(a.fuel) if a.fuel else None,
                                 temp_dc=a.temp_dc, location=location), course_hold=not a.nose_heading)


if __name__ == "__main__":
    main()
