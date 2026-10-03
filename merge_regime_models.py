"""İki rejim uzmanlı modeli birleştir (2026-10-03): uzmanlar ayrı eğitildiği için (diğerleri kilitliyken) birleştirilebilir.

Örnek: ileri uçuş uzmanı A'dan (hızlanma eğitimi, hover + iniş kilitli), hover ve iniş uzmanları B'den (AFCS heading hold
eğitimi, ileri uçuş kilitli). Değer ağı (critic) A'nınki kalır (yalnızca eğitimde kullanılır; deterministik uçuş aksiyon
ağlarıyla). Env ayarları: A'nınki + B'nin üstüne yazdıkları (ör. afcs_hdg_hold_to_land).

Kullanım:
  python merge_regime_models.py --base A.zip --take B.zip --regimes hover,land --out C.zip
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO  # noqa: E402

from regime_policy import REGIMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--regimes", default="hover,land")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, take = PPO.load(a.base, device="cpu"), PPO.load(a.take, device="cpu")
    pb, pt = base.policy, take.policy
    regs = [r for r in a.regimes.split(",") if r]
    for r in regs:
        k = REGIMES.index(r)
        pb.pi_experts[k].load_state_dict(pt.pi_experts[k].state_dict())
        pb.act_experts[k].load_state_dict(pt.act_experts[k].state_dict())
        with th.no_grad():
            pb.log_std_experts[k].copy_(pt.log_std_experts[k])
    ov = dict(getattr(base, "ah1s_env_overrides", None) or {})
    ov_t = dict(getattr(take, "ah1s_env_overrides", None) or {})
    for key, v in ov_t.items():
        if key not in ov:
            ov[key] = v
        elif ov[key] != v:
            print(f"uyarı: {key} farklı (base {ov[key]!r}, take {v!r}) → take'inki")
            ov[key] = v
    setattr(base, "ah1s_env_overrides", ov)
    # doğrulama: her rejimde aksiyon ortalaması kaynağıyla aynı
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, size=(3000, base.observation_space.shape[0])).astype(np.float32)
    obs[:, 24] = 0.0
    obs[:, 34] = 0.0
    obs[1000:2000, 34] = 1.0
    obs[2000:, 24] = 1.0
    o = th.as_tensor(obs)
    with th.no_grad():
        mu = pb.get_distribution(o).distribution.mean
        for r, sl in (("hover", slice(0, 1000)), ("cruise", slice(1000, 2000)), ("land", slice(2000, 3000))):
            src = pt if r in regs else PPO.load(a.base, device="cpu").policy
            d = float((mu[sl] - src.get_distribution(o[sl]).distribution.mean).abs().max())
            print(f"{r:7s} ← {'take' if r in regs else 'base'}: en büyük fark {d:.2e}")
            assert d < 1e-5
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    base.save(a.out)
    print(f"kaydedildi: {a.out}; env ayarları: {ov}")


if __name__ == "__main__":
    main()
