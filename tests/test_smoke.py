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
