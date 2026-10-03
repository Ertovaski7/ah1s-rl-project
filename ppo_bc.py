from __future__ import annotations

"""
PPO + davranış klonlama (BC) buffer'ı — unutmaya karşı (2026-10-03)
=================================================================

Sorun: ajan tek bir sinir ağı. Bir beceriyi (ör. ileri uçuşta irtifa tutma) eğitirken ağırlıklar değişir; başka bir
beceri (ör. iniş) için kullanılan ağırlıklar da değişir → o beceri bozulur ("catastrophic forgetting"). Tekrar
(rehearsal: eski seviyelerden episode) bunu azaltır ama sıfırlamaz: PPO on-policy'dir, eski deneyimi yeniden kullanamaz;
tekrar episode'larının gradyanı ödüle ve şansa bağlıdır.

Çare (literatür: Wołczyk vd. 2024 "Fine-tuning RL models is secretly a forgetting mitigation problem", ICML — BC
knowledge retention; Rolnick vd. 2019 CLEAR — replay + behavioral cloning; Rusu vd. 2015 policy distillation):
  1. Eğitim ÖNCESİ, dondurulmuş referans modelin (öğretmen) korunacak becerilerdeki durumları bir buffer'a yazılır
     (gözlem + öğretmenin action ortalaması). `docs/flight/collect_bc_buffer.py`.
  2. Eğitim SIRASINDA, her PPO gradyan adımında buffer'dan bir mini-batch alınır; öğrencinin action ortalaması
     öğretmeninkinden uzaklaştıkça ceza: L_bc = mean_batch Σ_eksen (μ_öğrenci − μ_öğretmen)² / (2 σ_öğretmen²)
     (öğretmenin sabit σ'sıyla Gauss KL'sinin ortalama terimi). Toplam kayıp = PPO kaybı + bc_coef · L_bc.
Buffer yalnızca KORUNACAK görevlerin durumlarını içerir (eğitilen görevin durumları hariç) → yeni beceri öğrenilir,
eskiler o durumlarda öğretmenin davranışına bağlı kalır.

Kullanım: train_command_curriculum.py --bc-buffer <npz> --bc-coef 1.0 (bkz. orada).
Kaydedilen model sıradan bir SB3 PPO zip'idir (buffer zip'e girmez); PPO.load ile açılır.
"""

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class PPOBC(PPO):
    """SB3 PPO; train() SB3 2.x'inkinin aynısı + isteğe bağlı BC (policy distillation) kaybı."""

    bc_coef: float = 0.0
    bc_batch: int = 512

    def set_bc_buffer(self, path: str, coef: float = 1.0, batch: int = 512, seed: int = 0):
        d = np.load(path)
        self._bc_obs = th.as_tensor(d["obs"], dtype=th.float32, device=self.device)
        self._bc_mu = th.as_tensor(d["mu"], dtype=th.float32, device=self.device)
        self._bc_inv2var = th.as_tensor(1.0 / (2.0 * np.asarray(d["sigma"], dtype=np.float64) ** 2), dtype=th.float32,
                                        device=self.device)
        self._bc_rng = np.random.default_rng(seed)
        self.bc_coef, self.bc_batch = float(coef), int(batch)
        self.bc_path = str(path)
        if self._bc_obs.shape[1] != self.observation_space.shape[0]:
            raise ValueError(f"BC buffer gözlem boyutu {self._bc_obs.shape[1]} ≠ model {self.observation_space.shape[0]}")
        return int(self._bc_obs.shape[0])

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_bc_obs", "_bc_mu", "_bc_inv2var", "_bc_rng"]

    def _bc_loss(self) -> th.Tensor | None:
        if self.bc_coef <= 0.0 or getattr(self, "_bc_obs", None) is None:
            return None
        idx = th.as_tensor(self._bc_rng.integers(0, self._bc_obs.shape[0], size=self.bc_batch), device=self.device)
        mu_s = self.policy.get_distribution(self._bc_obs[idx]).distribution.mean
        return (((mu_s - self._bc_mu[idx]) ** 2) * self._bc_inv2var).sum(dim=-1).mean()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions, bc_losses = [], [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf,
                                                                     clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                bc = self._bc_loss()
                if bc is not None:
                    loss = loss + self.bc_coef * bc
                    bc_losses.append(bc.item())
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if bc_losses:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.last_bc_loss = float(np.mean(bc_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
