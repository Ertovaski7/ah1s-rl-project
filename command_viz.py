from __future__ import annotations

"""
COMMAND VIZ — komut / manevra ajanını 3D ve metriklerle canlı izle
==================================================================

Tek bir JSBSim uçuşu (eğitilmiş PPO + env) arka planda gerçek zamanlı akar.
Tarayıcıdaki sayfadan istediğin an Δheading / Δhız / Δirtifa komutu verirsin;
helikopteri (low-poly Bell modeli, `viz/heli_bell.glb`) 3D izler, komutun
metriklerini (yükselme, aşma, oturma, kuplaj, eğitimdeki başarı kararı) ve
zaman serilerini görürsün. Aynı sayfa kayıtlı uçuşları da oynatır.

İki görev (model observation boyutundan anlaşılır):
  manevra (varsayılan, models_maneuver/maneuver_M5_final.zip): helicopter_env_maneuver.py,
      0–100 kt, her komutun bir süre hedefi (T) var — boş bırakılırsa seçili çeviklikten
      (M3 rahat / M4 hızlı / M5 agresif). Sayfada attitude göstergesi ve ajanın yatış /
      yunuslama komutları görünür.
  komut (models_command_curriculum/v2_R1_final.zip): helicopter_env_command.py, 15 ft/s civarı.

Kullanım
--------
  Yerel (repo kökünde):
    python command_viz.py                       # manevra modeli → http://127.0.0.1:8765
    python command_viz.py --model models_command_curriculum/v2_R1_final.zip --port 8766   # eski komut modeli
  Colab (hücrede, satır içi; localhost / paylaşım linki yok):
    import command_viz; command_viz.colab()
  Kayıtlı demo uçuşları üret (sayfanın "kayıt" modu için):
    python command_viz.py record --out viz/demo_flights.json

Uçuş dışa aktarma: sayfadaki JSON / ACMI düğmeleri (ACMI = Tacview kaydı).
Komut uygulama yolu eğitim ve değerlendirmeyle aynıdır (env.queue_command →
env._issue_due_commands): güvenli aralığın dışına taşan Δ'nın işareti çevrilir.
"""

import argparse
import base64
import json
import math
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
from helicopter_env_maneuver import OBS_DIM_M, HelicopterEnvManeuver, ManeuverEnvConfig  # noqa: E402
from maneuver_curriculum import DEFAULT_MANEUVER_LEVELS  # noqa: E402

VIZ_DIR = REPO_ROOT / "viz"
PAGE_FILE = VIZ_DIR / "command_viz.html"
MODEL_FILE = VIZ_DIR / "heli_bell.glb"
COMMAND_POLICY = REPO_ROOT / "models_command_curriculum" / "v2_R1_final.zip"
MANEUVER_POLICY = REPO_ROOT / "models_maneuver" / "maneuver_M5_final.zip"
# varsayılan: manevra modeli (süre hedefli Δ komutları, 0–100 kt); yoksa eski komut modeli
DEFAULT_POLICY = MANEUVER_POLICY if MANEUVER_POLICY.exists() else COMMAND_POLICY
# görev başına varsayılan başlangıç (eğitim aralığının içinde)
DEFAULT_START = {"command": dict(alt=300.0, speed=15.0, heading=0.0, seed=0),
                 "maneuver": dict(alt=800.0, speed=101.3, heading=0.0, seed=0)}      # 800 ft, 60 kt
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
]
COLS = [c for c, _ in COLUMNS]
AXIS_ERR_COL = {"heading": "e_psi", "speed": "e_u", "altitude": "e_h"}


def _limits(cfg: CommandEnvConfig, task: str = "command") -> dict:
    if task == "maneuver":
        trained = {"heading": 180.0, "speed": 50.0, "altitude": 150.0,
                   "start_alt_ft": [600.0, 1500.0], "start_speed_fps": [0.0, 169.0]}
        timing = dict(grace_frac=cfg.deadline_grace_frac, grace_min_s=cfg.deadline_grace_min_s,
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
    from stable_baselines3 import PPO
    model = PPO.load(str(path), device="cpu")
    fn = lambda obs: model.predict(obs, deterministic=True)[0]          # noqa: E731
    return (fn, int(model.observation_space.shape[0])) if with_obs_dim else fn


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
                 policy=None, start: dict | None = None, task: str | None = None):
        self.policy_path = Path(policy_path)
        self.env_config = env_config
        obs_dim = None
        if policy is None:
            policy, obs_dim = load_policy(self.policy_path, with_obs_dim=True)
        # görev modelin observation boyutundan anlaşılır (komut 19, manevra 24)
        self.task = task or ("maneuver" if obs_dim == OBS_DIM_M else "command")
        if self.task == "maneuver":
            cfg = ManeuverEnvConfig()
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

    def request_reset(self, alt=None, speed=None, heading=None, seed=None, **_ignored) -> dict:
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
        options = dict(commands=[], episode_s=LIVE_EPISODE_S, start_alt_ft=float(opts["alt"]),
                       start_speed_fps=float(opts["speed"]))
        if opts.get("heading") is not None:
            options["start_heading_deg"] = float(opts["heading"])
        obs, info = self.env.reset(seed=int(opts.get("seed", 0)), options=options)
        f = self.env.fdm
        lat0, lon0 = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
        ground = float(f["position/h-sl-ft"]) - float(f["position/h-agl-ft"])
        with self.lock:
            self.start_opts = dict(opts)
            self._obs = obs
            self._lat0, self._lon0, self._ground_msl = lat0, lon0, ground
            self.flight_id += 1
            self.rows = [self._row(info, np.zeros(4), np.zeros(4), self.env.trim, 0.0)]
            self.commands, self.events, self._pending_cmds = [], [], []
            self.meta = dict(model=self.policy_path.name, env_config=self.env_config, start=dict(opts),
                             lat0_deg=math.degrees(lat0), lon0_deg=math.degrees(lon0), ground_msl_ft=ground,
                             setup=dict(mode=info["setup"].get("mode"), heading_deg=info["heading_deg"]),
                             created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            self.events.append(dict(t=0.0, type="start", message=(
                f"Başladı: {opts['alt']:.0f} ft, {opts['speed']:.0f} ft/s, heading {info['heading_deg']:.0f}°")))
            self.termination = None
            self._done = False
            self.state, self.message = "running", "Uçuyor — komut verebilirsin."

    def _step(self):
        env = self.env
        with self.lock:
            pending, self._pending_cmds = self._pending_cmds, []
        for d in pending:
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
            self._sync_windows()
            if term or trunc:
                self._done = True
                self.termination = info.get("termination") or ("time_limit" if trunc else "?")
                for k, r in enumerate(info.get("command_results") or []):
                    if k < len(self.commands):
                        self.commands[k].update(verdict=bool(r["success"]), coupling_ok=r.get("coupling_ok"),
                                                final_err=r.get("final_err"), max_abs_err=r.get("max_abs_err"),
                                                settle_s=r.get("settle_s"), on_time=r.get("on_time"))
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
                         T=w.get("T"), deadline=w.get("deadline"))
                c["ref"]["heading_wrapped"] = env.ref["heading"] % 360.0
                if any(abs(c["applied"][a] - c["requested"][a]) > 1e-9 for a in AXES):
                    self.events.append(dict(t=float(w["t"]), type="adjusted", command=c["id"], message=(
                        f"K{c['id']}: güvenli aralık için Δ değiştirildi → " + ", ".join(
                            f"{a} {c['applied'][a]:+g}" for a in AXES if c['applied'][a] != c['requested'][a]))))
            c["streak_s"] = float(w["streak"])
            c["max_abs_err"] = dict(w["max_abs_err"])
            if w["success"] is not None and c["verdict"] is None:   # pencere kapandı → env'in kararı
                c.update(verdict=bool(w["success"]), coupling_ok=w.get("coupling_ok"), final_err=w.get("final_err"),
                         settle_s=w.get("settle_s"), on_time=w.get("on_time"))

    def _row(self, info: dict, action, filt, controls, reward) -> list:
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
        return [round(float(v), nd) if _finite(v) else None for v, (_, nd) in zip(vals, COLUMNS)]

    def _status(self) -> dict:
        return dict(state=self.state, message=self.message, paused=self.paused, time_scale=self.time_scale,
                    t=self.rows[-1][0] if self.rows else 0.0, termination=self.termination, flight=self.flight_id,
                    sim_rate=round(self.sim_rate, 2), pending=len(self._pending_cmds))

    # ---------------- görev (kayıt) ----------------

    def run_mission(self, start: dict, schedule: list, duration_s: float) -> dict:
        """Gerçek zamansız: aynı _step yolu, komutlar zamanında kuyruğa girer."""
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
            for k, c in enumerate(self.commands):
                if c["applied"] is None:
                    continue
                t_end = self.commands[k + 1]["t"] if k + 1 < len(self.commands) else float(t[-1])
                last = k + 1 == len(self.commands)
                m = command_metrics(t, err, c["applied"], c["t"], t_end, self.limits,
                                    terminated=bool(last and self.termination and self.termination != "time_limit"),
                                    deadline=c.get("deadline"))
                cmds.append(dict(c, metrics=m))
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

def page_fragment(boot: dict | None = None) -> str:
    html = PAGE_FILE.read_text(encoding="utf-8")
    if boot:
        html = f"<script>window.AH1S_VIZ_BOOT = {json.dumps(boot)};</script>\n" + html
    return html


def page_document(boot: dict | None = None) -> str:
    return ("<!doctype html>\n<html lang=\"tr\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + page_fragment(boot) + "\n</body>\n</html>\n")


# =====================================================================
# YEREL HTTP SUNUCU (yalnızca standart kütüphane)
# =====================================================================

STATIC_TYPES = {".glb": "model/gltf-binary", ".json": "application/json", ".js": "text/javascript",
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
                        and target.suffix in STATIC_TYPES and "tools" not in target.parts):
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
    glb_b64 = base64.b64encode(MODEL_FILE.read_bytes()).decode("ascii")
    output.register_callback("ah1s_viz.hello", lambda: JSON(flight.hello()))
    output.register_callback("ah1s_viz.state", lambda fid=None, since=0: JSON(flight.state_since(fid, since)))
    output.register_callback("ah1s_viz.command", lambda d=None: JSON(flight.request_command(d or {})))
    output.register_callback("ah1s_viz.reset", lambda d=None: JSON(flight.request_reset(**(d or {}))))
    output.register_callback("ah1s_viz.control", lambda d=None: JSON(flight.set_control(**(d or {}))))
    output.register_callback("ah1s_viz.model", lambda: JSON(dict(b64=glb_b64)))
    output.register_callback("ah1s_viz.save", lambda: JSON(flight.save(save_dir)))
    flight.start()
    display(HTML(page_fragment(dict(transport="colab"))))
    return flight


# =====================================================================
# KAYITLI DEMO UÇUŞLARI
# =====================================================================

# policy="maneuver" → manevra modeli (MANEUVER_POLICY ya da `record --maneuver-model`); diğerleri
# models_command_curriculum/ içindeki komut modelleri. Manevra görevlerinde süre hedefi (T) verilmezse
# env onu M5 çevikliğinden hesaplar (55° yatış, 25°/s pedal, 6 ft/s² ivme, 20 ft/s tırmanış, +2 s tepki).
GOREV6 = [(5, {"heading": 90}), (40, {"speed": 8}), (75, {"altitude": 100}),
          (110, {"heading": -45, "altitude": -60}), (145, {"speed": -10}), (180, {"heading": 180})]
MISSIONS = [
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


def record_missions(out: Path, every: int = 2, only: list | None = None, maneuver_model: Path | None = None) -> dict:
    flights = []
    cache = {}
    for m in MISSIONS:
        if only and m["id"] not in only:
            continue
        if m["policy"] == "maneuver":
            path = Path(maneuver_model) if maneuver_model else MANEUVER_POLICY
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
        a = p.parse_args(argv[1:])
        record_missions(a.out, a.every, a.only, a.maneuver_model)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_POLICY,
                   help="PPO modeli (.zip); görev (komut / manevra) modelin observation boyutundan anlaşılır")
    p.add_argument("--env-config", choices=["v2", "v1"], default="v2", help="yalnızca komut görevi")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = ağdaki diğer cihazlar da erişsin")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--start-alt", type=float, default=None, help="ft AGL (varsayılan: manevra 800, komut 300)")
    p.add_argument("--start-speed", type=float, default=None, help="ft/s (varsayılan: manevra 101.3 = 60 kt, komut 15)")
    p.add_argument("--start-heading", type=float, default=None, help="° (varsayılan 0)")
    p.add_argument("--open", action="store_true", help="tarayıcıyı aç")
    a = p.parse_args(argv)
    flight = LiveFlight(a.model, env_config=a.env_config,
                        start=dict(alt=a.start_alt, speed=a.start_speed, heading=a.start_heading))
    serve(flight, a.host, a.port, a.open)


if __name__ == "__main__":
    main()
