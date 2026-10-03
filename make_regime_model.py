"""Mevcut tek-ağlı uçuş modelini rejim uzmanlı modele dönüştür (regime_policy.py), davranış birebir aynı.

Üç uzmanın (hover / ileri uçuş / iniş) her biri kaynak modelin aksiyon ağının kopyası; değer ağı aynen. Doğrulama:
rastgele + gerçek gözlemlerde action ortalaması ve değer farkı ≈ 0.

Kullanım:
  python make_regime_model.py --src models_flight/flight_v4.zip --out runs/regime/flight_v4_regime.zip
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight  # noqa: E402
from regime_policy import REGIMES, RegimePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = PPO.load(a.src, device="cpu")
    ov = dict(getattr(src, "ah1s_env_overrides", None) or {})
    venv = DummyVecEnv([lambda: HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))])
    pk = dict(src.policy_kwargs)
    new = PPO(RegimePolicy, venv, policy_kwargs=pk, learning_rate=1e-4, n_steps=src.n_steps, batch_size=src.batch_size,
              n_epochs=src.n_epochs, gamma=src.gamma, gae_lambda=src.gae_lambda, clip_range=0.2, ent_coef=src.ent_coef,
              vf_coef=src.vf_coef, max_grad_norm=src.max_grad_norm, device="cpu", verbose=0)
    missing, unexpected = new.policy.load_state_dict(src.policy.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(k.split(".")[0] in ("pi_experts", "act_experts", "log_std_experts") for k in missing), missing
    p = new.policy
    for k in range(len(REGIMES)):
        p.pi_experts[k].load_state_dict(src.policy.mlp_extractor.policy_net.state_dict())
        p.act_experts[k].load_state_dict(src.policy.action_net.state_dict())
        with th.no_grad():
            p.log_std_experts[k].copy_(src.policy.log_std)
    setattr(new, "ah1s_env_overrides", ov)
    # doğrulama
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, size=(4000, src.observation_space.shape[0])).astype(np.float32)
    obs[:1000, 24], obs[1000:2000, 34] = 1.0, 1.0
    with th.no_grad():
        o = th.as_tensor(obs)
        d0, d1 = src.policy.get_distribution(o).distribution, p.get_distribution(o).distribution
        dmu = float((d0.mean - d1.mean).abs().max())
        dsd = float((d0.stddev - d1.stddev).abs().max())
        dv = float((src.policy.predict_values(o) - p.predict_values(o)).abs().max())
    print(f"doğrulama: en büyük fark action ortalaması {dmu:.2e}, std {dsd:.2e}, değer {dv:.2e}; "
          f"rejim sayıları {np.bincount(p.regime_index(o).numpy(), minlength=3).tolist()} (hover / ileri / iniş)")
    assert dmu < 1e-5 and dsd < 1e-6 and dv < 1e-5
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    new.save(a.out)
    back = PPO.load(a.out, device="cpu")
    with th.no_grad():
        dmu2 = float((back.policy.get_distribution(o).distribution.mean - d0.mean).abs().max())
    print(f"kaydedildi: {a.out}  (yeniden yüklemede fark {dmu2:.2e}; politika sınıfı {type(back.policy).__name__})")


if __name__ == "__main__":
    main()
