"""Duman testleri: env'ler kurulup adım atabiliyor mu (2026-10-02 regresyonu sonrası). Çalıştır: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run(env, n=40):
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    for _ in range(n):
        obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            break
    return info


def test_flight_env_resets_and_steps():
    from helicopter_env_flight import HelicopterEnvFlight
    for lvl in ("F1", "F5", "F13"):
        info = _run(HelicopterEnvFlight(level=lvl))
        assert "altitude" in info


def test_flight_env_rotor_dt_sim():
    from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
    env = HelicopterEnvFlight(level="F2", config=FlightEnvConfig(rotor_dt_mode="sim"))
    _run(env)
    assert abs(float(env.fdm["ge/inflow-amplification"]) - 11.506) < 0.01


def test_takeoff_env_resets_and_steps():
    from helicopter_env_takeoff import HelicopterEnvTakeoff
    _run(HelicopterEnvTakeoff(level="K2"))


def test_maneuver_env_resets_and_steps():
    from helicopter_env_maneuver import HelicopterEnvManeuver
    _run(HelicopterEnvManeuver(level="M1"))


def test_regime_model_loads_and_predicts():
    """flight_v5 rejim uzmanlı model (regime_policy.py): yüklenir, iniş / ileri / hover rejimlerini ayırır."""
    from pathlib import Path
    from stable_baselines3 import PPO
    p = Path(__file__).resolve().parents[1] / "models_flight" / "flight_v5.zip"
    if not p.exists():
        return
    m = PPO.load(str(p), device="cpu")
    assert type(m.policy).__name__ == "RegimePolicy"
    import torch as th
    o = np.zeros((3, m.observation_space.shape[0]), dtype=np.float32)
    o[1, 34] = 1.0
    o[2, 24] = 1.0
    assert m.policy.regime_index(th.as_tensor(o)).tolist() == [0, 1, 2]
    a, _ = m.predict(o[0], deterministic=True)
    assert a.shape == (4,)


def test_flight_env_accel_flags_and_f16():
    """2026-10-03 hızlanma bayrakları (referans rampa, bant, paylar, alçak hız duvarı) ve F16 seviyesi adım atabiliyor."""
    from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
    cfg = FlightEnvConfig(ff_init_airspeed=True, ff_ratchet=True, accel_band_k_hi=0.6, accel_h_full_frac=0.5,
                          accel_vs_allow_fps=5.0, side_ramp_kt=(10.0, 25.0), accel_allow_hs_ft=(40.0, 140.0),
                          pen_low_speed=1.0)
    info = _run(HelicopterEnvFlight(level="F16", config=cfg), n=60)
    assert "altitude" in info


def test_live_weather_and_envelope():
    """Canlı hava değişikliği (rüzgâr rampası, türbülans, gust) ve zarf dışı komutun reddi (strict yönlendirici)."""
    from flight_commands import route_command
    from helicopter_env_flight import HelicopterEnvFlight
    env = HelicopterEnvFlight(level="F8")
    env.reset(seed=0, options=dict(level="F8", tasks=[dict(kind="hold")], start="hover", start_alt_ft=100.0,
                                   physics=dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none"), episode_s=60.0))
    p = env.ext.set_live(env.fdm, wind_kt=20.0, wind_from_deg=90.0, turb_level="light", gusts=True,
                         gust_now=dict(mag_kt=8.0, dir_deg=270.0))
    assert p["wind_kt"] == 20.0 and p["turb_level"] == "light" and p["gust_rate_per_min"] == 1.0
    for _ in range(100):                                       # 7.5 s: rüzgâr 4 kt/s ile 20 kt'a çıkar
        env.step(np.zeros(4))
    wn, we = env.ext.wind_ned
    assert abs(we + 20.0 * 1.6878) < 0.5 and abs(wn) < 0.5     # 090°'den → batıya eser
    assert not route_command(env, alt_ft=10000.0, strict=True).ok
    assert not route_command(env, speed_kt=1000.0, strict=True).ok
    assert route_command(env, alt_ft=10000.0).ok               # strict değilken eski davranış: kırpma
