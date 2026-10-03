from __future__ import annotations

"""
REGIME POLICY — uçuş rejimine göre ayrılmış aksiyon ağları (parametre izolasyonu), unutmaya karşı (2026-10-03)
=============================================================================================================

Sorun: tek bir aksiyon ağı bütün becerileri taşıyor. İleri uçuş eğitimi ağırlıkları değiştirince iniş de değişiyor
(ve tersi) — "catastrophic forgetting". Üç seed'li F14 deneyi: aynı eğitim, üç farklı bozulma.

Çözüm (öğretmen / öğrenci YOK, taklit kaybı YOK — mentor kuralı): ajan yine TEK bir ağ ve TEK bir PPO; ama aksiyonu
üreten kısım üç uzmana bölünür. Hangi uzmanın çalışacağını gözlemdeki bayraklar belirler (öğrenilmiş bir seçici değil):
    iniş     : gözlem[24] (iniş bayrağı) = 1                 → uzman 2
    ileri    : gözlem[34] (ileri uçuş bayrağı) = 1, iniş değil → uzman 1
    hover    : diğer her şey (yerde kalkış, hover, manevralar, duruş) → uzman 0
Her uzman = aksiyon MLP'si (256×256) + aksiyon katmanı + log_std. Değer ağı (critic) ortak. Başlangıçta üç uzman da
mevcut modelin aksiyon ağının birebir kopyası → davranış aynı (make_regime_model.py doğrular).
Eğitimde bir rejim DONDURULABİLİR (`frozen_regimes`): o uzmanın ağırlıkları değişmez → o rejimdeki davranış (ör. iniş)
başka bir rejim eğitilirken bozulamaz. Literatürdeki karşılığı: parametre izolasyonu / görev başına ayrı çıkış
(PackNet, Progressive Networks ailesi) — burada görev kimliği uçuş rejimi.

Kayıtlı model sıradan bir SB3 PPO zip'i; PPO.load bu modülü içe aktarır (repo kökü sys.path'te olmalı).
"""

import copy

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

REGIMES = ("hover", "cruise", "land")


class RegimePolicy(ActorCriticPolicy):
    def __init__(self, *args, land_idx: int = 24, cruise_idx: int = 34, frozen_regimes=(), **kwargs):
        self.land_idx, self.cruise_idx = int(land_idx), int(cruise_idx)
        self.frozen_regimes = tuple(frozen_regimes)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.pi_experts = nn.ModuleList([copy.deepcopy(self.mlp_extractor.policy_net) for _ in REGIMES])
        self.act_experts = nn.ModuleList([copy.deepcopy(self.action_net) for _ in REGIMES])
        self.log_std_experts = nn.ParameterList([nn.Parameter(self.log_std.detach().clone()) for _ in REGIMES])
        # temel sınıfın tek aksiyon ağı kullanılmıyor (yalnızca dönüştürmede kaynak): eğitilmesin
        for p in list(self.mlp_extractor.policy_net.parameters()) + list(self.action_net.parameters()) + [self.log_std]:
            p.requires_grad_(False)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.set_frozen(self.frozen_regimes)

    # ------------------------------------------------------------------
    def set_frozen(self, regimes) -> None:
        self.frozen_regimes = tuple(regimes)
        for k, name in enumerate(REGIMES):
            train = name not in self.frozen_regimes
            for p in list(self.pi_experts[k].parameters()) + list(self.act_experts[k].parameters()):
                p.requires_grad_(train)
            self.log_std_experts[k].requires_grad_(train)

    def regime_index(self, obs: th.Tensor) -> th.Tensor:
        land = obs[:, self.land_idx] > 0.5
        cruise = (obs[:, self.cruise_idx] > 0.5) & ~land
        idx = th.zeros(obs.shape[0], dtype=th.long, device=obs.device)
        idx[cruise] = 1
        idx[land] = 2
        return idx

    def _dist(self, features: th.Tensor):
        idx = self.regime_index(features)
        means = th.stack([a(p(features)) for p, a in zip(self.pi_experts, self.act_experts)], dim=1)   # B × 3 × A
        mean = means.gather(1, idx[:, None, None].expand(-1, 1, means.shape[-1])).squeeze(1)
        log_std = th.stack(list(self.log_std_experts), dim=0)[idx]                                     # B × A
        return self.action_dist.proba_distribution(mean, log_std)

    # ------------------------------------------------------------------
    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        values = self.value_net(self.mlp_extractor.forward_critic(features))
        distribution = self._dist(features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        distribution = self._dist(features)
        values = self.value_net(self.mlp_extractor.forward_critic(features))
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        return self._dist(super().extract_features(obs, self.pi_features_extractor))

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d.update(land_idx=self.land_idx, cruise_idx=self.cruise_idx, frozen_regimes=self.frozen_regimes)
        return d
