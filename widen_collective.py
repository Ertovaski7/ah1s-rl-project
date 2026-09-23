from __future__ import annotations

"""
WIDEN COLLECTIVE — eğitilmiş manevra modelinin collective yetkisini davranışı koruyarak genişletir
=================================================================================================

Manevra env'inde collective = trim + coll_scale · filt[0] (filt: action'ın alçak geçiren filtresi).
M5 modeli coll_scale = 0.25 ile eğitildi. Ölçüm (bölüm 29): 60–100 kt'ta 55° yatışlı dönüşte
irtifayı tutmak için ~0.35–0.45 gerekiyor → yetki doyuyor, irtifa 55–145 ft düşüyor; ajan bu yüzden
~45° yatışta kalmayı öğrenmiş (büyük dönüşler süre hedefine geç kalıyor).

Yetkiyi 0.25 → 0.45 yapıp modeli sıfırdan eğitmek yerine ağırlıklar yeniden ölçeklenir (k = eski / yeni):

  * action_net'in collective satırı (ağırlık + bias) · k    → aynı fiziksel collective komutu
  * log_std[collective] + ln k                              → aynı fiziksel keşif gürültüsü
  * ilk katmanda (policy ve value ağları) obs[20] = filt[0] sütunu / k  → aynı iç temsil

Sonuç: ±0.25 içinde kalan her durumda policy fiziksel olarak birebir aynı kumandayı verir (değer ağı da aynı);
eski modelin doyduğu yerde (ör. yüksek hızda dik yatış) artık ±0.45'e kadar gidebilir. Model yeni ayarı
zip'inde taşır (`ah1s_env_overrides`); değerlendirme / görselleştirme / eğitim env'i buna göre kurar.

Kullanım
  python widen_collective.py models_maneuver/maneuver_M5_final.zip runs/rob/m5_coll045.zip --coll-scale 0.45
"""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def widen(src, dst, new_scale: float) -> dict:
    import torch
    from stable_baselines3 import PPO

    from helicopter_env_maneuver import ENV_OVERRIDES_ATTR, OBS_DIM_M, OBS_FILT_INDEX, ManeuverEnvConfig, model_env_overrides

    model = PPO.load(str(src), device="cpu")
    if int(model.observation_space.shape[0]) != OBS_DIM_M:
        raise SystemExit(f"{src} bir manevra modeli değil (obs {model.observation_space.shape})")
    ov = model_env_overrides(model)
    old_scale = float(ov.get("coll_scale", ManeuverEnvConfig().coll_scale))
    k = old_scale / float(new_scale)
    pol = model.policy
    with torch.no_grad():
        pol.mlp_extractor.policy_net[0].weight[:, OBS_FILT_INDEX] /= k
        pol.mlp_extractor.value_net[0].weight[:, OBS_FILT_INDEX] /= k
        pol.action_net.weight[0, :] *= k
        pol.action_net.bias[0] *= k
        pol.log_std[0] += math.log(k)
    ov["coll_scale"] = float(new_scale)
    setattr(model, ENV_OVERRIDES_ATTR, ov)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dst))
    return dict(old_scale=old_scale, new_scale=float(new_scale), k=k, overrides=ov)


def check(src, dst, n_obs: int = 2000, seed: int = 0) -> float:
    """Rastgele observation'larda fiziksel collective komutunun ve değerin aynı kaldığını doğrular.
    Döndürür: en büyük fark (±eski yetki içindeki durumlarda ~1e-6)."""
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from helicopter_env_maneuver import OBS_FILT_INDEX, model_env_overrides

    a, b = PPO.load(str(src), device="cpu"), PPO.load(str(dst), device="cpu")
    sa = model_env_overrides(a).get("coll_scale", 0.25)
    sb = model_env_overrides(b).get("coll_scale", 0.25)
    rng = np.random.default_rng(seed)
    obs_a = rng.uniform(-1.0, 1.0, size=(n_obs, a.observation_space.shape[0])).astype(np.float32)
    obs_b = obs_a.copy()
    obs_b[:, OBS_FILT_INDEX] *= sa / sb                    # aynı fiziksel filtre durumu, yeni birimde
    with torch.no_grad():
        ma = a.policy.get_distribution(torch.as_tensor(obs_a)).distribution.mean.numpy()
        mb = b.policy.get_distribution(torch.as_tensor(obs_b)).distribution.mean.numpy()
        va = a.policy.predict_values(torch.as_tensor(obs_a)).numpy()
        vb = b.policy.predict_values(torch.as_tensor(obs_b)).numpy()
    phys_a, phys_b = sa * ma[:, 0], sb * mb[:, 0]          # fiziksel collective (clip öncesi)
    return float(max(np.abs(phys_a - phys_b).max(), np.abs(ma[:, 1:] - mb[:, 1:]).max(), np.abs(va - vb).max()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--coll-scale", type=float, default=0.45)
    args = ap.parse_args(argv)
    info = widen(args.src, args.dst, args.coll_scale)
    diff = check(args.src, args.dst)
    print(f"collective yetkisi ±{info['old_scale']:g} → ±{info['new_scale']:g} (k = {info['k']:.4f}); "
          f"kaydedildi: {args.dst}  env ayarları {info['overrides']}")
    print(f"kontrol: eski yetki içindeki durumlarda en büyük fark {diff:.2e} (fiziksel collective / diğer action / değer)")


if __name__ == "__main__":
    main()
