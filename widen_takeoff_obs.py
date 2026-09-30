from __future__ import annotations

"""
WIDEN TAKEOFF OBS — kalkış modeline yeni gözlem girdisi ekle (davranışı değiştirmeden)
=====================================================================================

2026-09-28: kalkış env'ine tork göstergesi gözlemi eklendi (`TakeoffEnvConfig.torque_obs`, obs 29 → 30). Eski model
bu girdiyi bilmiyor. Bu script modeli yeni gözlem boyutuna taşır: policy ve value ağlarının ilk katmanında yeni
sütunların ağırlıkları SIFIR → aynı durum için action / value birebir aynı (yeni girdi başta etkisiz); ince ayar
yeni girdiyi kullanmayı öğrenir. Env ayarları (`ah1s_env_overrides`) modelle kaydedilir; eğitim / değerlendirme /
canlı sayfa env'i bu ayarlarla kurar.

Kullanım
  python widen_takeoff_obs.py --model models_takeoff/takeoff_final.zip --out runs/tq/init_torque.zip \\
      --env '{"aircraft": "repo", "ground_effect": "calibrated", "power_cap_psi": 56, "torque_obs": true}'
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_OVERRIDES_ATTR = "ah1s_env_overrides"


def widen(model_path, out_path, env_overrides: dict, check: int = 200, seed: int = 0) -> dict:
    import torch
    from stable_baselines3 import PPO
    from helicopter_env_takeoff import HelicopterEnvTakeoff, TakeoffEnvConfig

    old = PPO.load(str(model_path), device="cpu")
    ov = dict(getattr(old, ENV_OVERRIDES_ATTR, None) or {})
    ov.update(env_overrides)
    env = HelicopterEnvTakeoff(level="K9", config=TakeoffEnvConfig(**ov))
    n_old, n_new = old.observation_space.shape[0], env.observation_space.shape[0]
    if n_new < n_old:
        raise SystemExit(f"yeni gözlem boyutu ({n_new}) eskisinden ({n_old}) küçük")
    new = PPO("MlpPolicy", env, learning_rate=old.learning_rate, n_steps=old.n_steps, batch_size=old.batch_size,
              n_epochs=old.n_epochs, gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=old.clip_range,
              ent_coef=old.ent_coef, vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm, target_kl=old.target_kl,
              policy_kwargs=old.policy_kwargs, device="cpu", verbose=0)
    sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
    widened = []
    with torch.no_grad():
        for k, v_new in sd_new.items():
            v_old = sd_old[k]
            if v_old.shape == v_new.shape:
                sd_new[k] = v_old.clone()
            elif v_old.dim() == 2 and v_old.shape[0] == v_new.shape[0] and v_old.shape[1] == n_old and v_new.shape[1] == n_new:
                w = torch.zeros_like(v_new)
                w[:, :n_old] = v_old
                sd_new[k] = w
                widened.append(k)
            else:
                raise SystemExit(f"beklenmeyen şekil: {k} {tuple(v_old.shape)} → {tuple(v_new.shape)}")
    new.policy.load_state_dict(sd_new)
    new.num_timesteps = old.num_timesteps
    setattr(new, ENV_OVERRIDES_ATTR, ov)
    # doğrulama: rastgele gözlemlerde (yeni girdiler rastgele) action ve value aynı mı
    rng = np.random.default_rng(seed)
    o_old = rng.uniform(-2, 2, size=(check, n_old)).astype(np.float32)
    o_new = np.concatenate([o_old, rng.uniform(-2, 2, size=(check, n_new - n_old)).astype(np.float32)], axis=1)
    a_old, _ = old.predict(o_old, deterministic=True)
    a_new, _ = new.predict(o_new, deterministic=True)
    with torch.no_grad():
        v_old = old.policy.predict_values(torch.as_tensor(o_old)).numpy()
        v_new = new.policy.predict_values(torch.as_tensor(o_new)).numpy()
    diff_a, diff_v = float(np.abs(a_old - a_new).max()), float(np.abs(v_old - v_new).max())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    new.save(str(out_path))
    return dict(obs=[n_old, n_new], widened=widened, max_action_diff=diff_a, max_value_diff=diff_v, env=ov)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="{}", help="env ayarları (JSON), modelinkilerin üstüne")
    args = ap.parse_args(argv)
    info = widen(args.model, args.out, json.loads(args.env))
    print(json.dumps(info, indent=1, ensure_ascii=False))
    if info["max_action_diff"] > 1e-5 or info["max_value_diff"] > 1e-4:
        raise SystemExit("[hata] genişletilmiş model eskisiyle aynı davranmıyor")
    print(f"kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
