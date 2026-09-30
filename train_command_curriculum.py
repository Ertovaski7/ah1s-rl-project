from __future__ import annotations

"""
TRAIN COMMAND CURRICULUM — PPO (sıfırdan) + otomatik seviye atlama
=================================================================

Mentorun planı (2026-09-22): teacher/student yok. PPO rastgele ağırlıklarla
başlar; helikopter 300 ft / 15 ft/s'de başlar; önce ±5° heading, öğrenince ±10°,
... sonra hız. Seviyeler: command_curriculum.py (DEFAULT_LEVELS).

Seviye atlama (promotion)
-------------------------
Son `--window` (100) episode'un başarı oranı ≥ `--threshold` (0.8) olunca:
  1) model `models/level_XX_<ad>.zip` olarak kaydedilir (o seviyeyi geçen ağ),
  2) tüm env'ler bir sonraki seviyeye alınır (ağ AYNEN devam eder),
  3) sayaçlar sıfırlanır.
Seviye değişince yarıda kalan episode'lar eski seviyeden sayılmaz.

Çıktılar (--out klasörü; Colab'da Google Drive önerilir)
-------------------------------------------------------
  models/latest.zip              düzenli kayıt (devam etmek için)
  models/level_XX_<ad>.zip        her geçilen seviye
  curriculum_state.json           seviye, geçmiş, adım sayısı (devam için)
  progress.csv                    her rollout sonunda seviye / başarı / hata özeti
  tb/                             TensorBoard (kuruluysa)

Kullanım (Colab, repo kökünde)
------------------------------
  # Drive'a kaydetmek için önce: from google.colab import drive; drive.mount('/content/drive')
  %run train_command_curriculum.py --out /content/drive/MyDrive/ah1s_runs/cmd_v1 --total-steps 6000000
  # kopan oturumu devam ettir:
  %run train_command_curriculum.py --out /content/drive/MyDrive/ah1s_runs/cmd_v1 --total-steps 6000000 --resume
  # yalnızca bir seviyede kal (atlama yok):
  %run train_command_curriculum.py --out runs/h1_only --level H1 --no-promote --total-steps 500000
  # hızlı duman testi (~1 dk):
  %run train_command_curriculum.py --out runs/smoke --smoke

Manevra curriculum'u (Δ komut + süre hedefi, 0–100 kt; helicopter_env_maneuver.py,
maneuver_curriculum.py — seviyeler M1…M5):
  %run train_command_curriculum.py --task maneuver --out runs/man --total-steps 8000000

Kalkış / hover / iniş curriculum'u (dört kumanda doğrudan; helicopter_env_takeoff.py,
takeoff_curriculum.py — seviyeler K1…K9, K5'ten itibaren ince ayar modu):
  %run train_command_curriculum.py --task takeoff --out runs/to --total-steps 12000000
  Seviyelerin `rehearse` / `p_rehearse` ayarı: eğitim env'lerinde episode'ların bir kısmı eski seviyelerden
  (unutmaya karşı); bu episode'lar seviye atlama istatistiğine girmez, log'da "tekrar:" diye ayrıca görünür.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from command_curriculum import DEFAULT_LEVELS, command_axis, find_level  # noqa: E402

INFO_KEYS = ("episode_success", "level_index", "commands_ok", "commands_total")


def make_config(name: str, task: str = "command", overrides: dict | None = None):
    if task == "takeoff":
        from helicopter_env_takeoff import TakeoffEnvConfig
        return TakeoffEnvConfig(**(overrides or {}))
    if task == "maneuver":
        from helicopter_env_maneuver import ManeuverEnvConfig
        return ManeuverEnvConfig(**(overrides or {}))
    from helicopter_env_command import CommandEnvConfig
    return CommandEnvConfig.v1() if name == "v1" else CommandEnvConfig()


def task_levels(task: str):
    if task == "takeoff":
        from takeoff_curriculum import DEFAULT_TAKEOFF_LEVELS
        return DEFAULT_TAKEOFF_LEVELS
    if task == "maneuver":
        from maneuver_curriculum import DEFAULT_MANEUVER_LEVELS
        return DEFAULT_MANEUVER_LEVELS
    return DEFAULT_LEVELS


def make_env_fn(rank: int, level_index: int, config_name: str, task: str = "command", overrides: dict | None = None,
                train: bool = False):
    def _init():
        # SubprocVecEnv işçisinde de repo kökü import yolunda olsun
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        if task == "takeoff":
            from helicopter_env_takeoff import HelicopterEnvTakeoff
            # train=True: seviyenin rehearse / p_rehearse ayarıyla eski seviyelerden de episode (unutmaya karşı)
            return HelicopterEnvTakeoff(level=level_index, config=make_config(config_name, task, overrides), rehearsal=train)
        if task == "maneuver":
            from helicopter_env_maneuver import HelicopterEnvManeuver
            return HelicopterEnvManeuver(level=level_index, config=make_config(config_name, task, overrides))
        from helicopter_env_command import HelicopterEnvCommand
        return HelicopterEnvCommand(level=level_index, config=make_config(config_name))
    return _init


def build_vec_env(n_envs: int, level_index: int, vec: str, config_name: str = "v2", task: str = "command",
                  overrides: dict | None = None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    fns = [make_env_fn(i, level_index, config_name, task, overrides, train=True) for i in range(n_envs)]
    if vec == "subproc" and n_envs > 1:
        method = "fork" if sys.platform.startswith("linux") else None
        venv = SubprocVecEnv(fns, start_method=method)
    else:
        venv = DummyVecEnv(fns)
    return VecMonitor(venv, info_keywords=INFO_KEYS)


# =====================================================================
# CURRICULUM CALLBACK
# =====================================================================

def make_callback(out: Path, levels, start_level: int, args, history: list, env_config: dict):
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=1)
            self.level = start_level
            self.recent = deque(maxlen=args.window)
            self.n_eps = 0
            self.level_start_steps = 0
            self.level_start_time = time.time()
            self.term = Counter()
            self.cmd_err = deque(maxlen=args.window)     # komut penceresi sonundaki |e_ψ| (seviyeye göre)
            # eksen türüne göre komut başarısı (heading / speed / altitude / combined) — unutmayı yakalar
            self.axis_ok = {}
            self.rehearse_ok = {}                        # eski seviyelerden tekrar episode'ları (seviye adı → başarılar)
            self.history = list(history)
            self.last_save = 0
            self.last_snap = 0
            self.last_eval = 0
            self.best_eval = -1.0
            self.eval_env = None
            self.t0 = time.time()
            self.csv_path = out / "progress.csv"
            self.state_path = out / "curriculum_state.json"

        # -------------------------------------------------------------
        def _apply_fine_tuning(self):
            """--fine-from seviyesinden itibaren daha küçük öğrenme hızı + KL sınırı.
            (lr 3e-4 ile hız seviyelerinde ajan büyük dönüşleri hızla unuttu; 1e-4 + KL 0.02 ile korudu.)"""
            if args.fine_from is None or self.level < find_level(args.fine_from, levels):
                return
            if self.model.learning_rate != args.fine_lr or self.model.target_kl != args.fine_kl:
                self.model.learning_rate = args.fine_lr
                self.model.lr_schedule = lambda _: args.fine_lr
                self.model.target_kl = args.fine_kl
                print(f"[curriculum] ince ayar modu: lr={args.fine_lr:g}, target_kl={args.fine_kl}")

        def _on_training_start(self):
            self.training_env.env_method("set_level", self.level)
            self.level_start_steps = self.num_timesteps
            self.last_save = self.num_timesteps
            self.last_snap = self.num_timesteps
            if args.eval_freq:
                self._evaluate()                        # başlangıç (ince ayar öncesi) değeri
            self._apply_fine_tuning()
            print(f"[curriculum] başlangıç seviyesi: {levels[self.level].name} — {levels[self.level].description}")

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if not done:
                    continue
                if info.get("level_index") != self.level:
                    # tekrar (rehearsal) episode'u: eski seviyenin başarısını ayrıca izle (unutma göstergesi)
                    if info.get("level") in getattr(levels[self.level], "rehearse", ()):
                        self.rehearse_ok.setdefault(info["level"], deque(maxlen=args.window)).append(
                            bool(info.get("episode_success", False)))
                    continue
                self.recent.append(bool(info.get("episode_success", False)))
                self.n_eps += 1
                self.term[info.get("termination", "?")] += 1
                for res in info.get("command_results") or []:
                    self.cmd_err.append(abs(float(res["final_err"]["heading"])))
                    ax = res.get("category") or command_axis(res["cmd"])      # dayanıklılık: interrupt / interrupted
                    self.axis_ok.setdefault(ax, deque(maxlen=args.window)).append(bool(res["success"]))
            if self.num_timesteps - self.last_save >= args.save_freq:
                self._save("latest")
                self.last_save = self.num_timesteps
            if args.snapshot_freq and self.num_timesteps - self.last_snap >= args.snapshot_freq:
                (out / "models").mkdir(parents=True, exist_ok=True)
                self.model.save(out / "models" / f"snap_{self.num_timesteps // 1000:05d}k.zip")
                self.last_snap = self.num_timesteps
            if args.eval_freq and self.num_timesteps - self.last_eval >= args.eval_freq:
                cur = self._evaluate()
                # --promote-on-eval: mevcut seviyenin DETERMİNİSTİK başarısı eşiği geçince de seviye atla
                # (inişte stokastik policy'nin gürültüsü temas hızını / oturmayı bozuyor; eğitim başarısı düşük kalıyor)
                if args.promote and args.promote_on_eval and cur is not None and cur >= self._threshold():
                    print(f"[curriculum] deterministik değerlendirme {cur:.0%} ≥ eşik {self._threshold():.0%} → seviye atlama")
                    return self._promote()
            if (args.promote and len(self.recent) >= args.window and self.n_eps >= args.min_episodes
                    and float(np.mean(self.recent)) >= self._threshold() and self._axes_ok()):
                return self._promote()
            return True

        def _threshold(self) -> float:
            """Episode başarısı eşiği: seviye kendi eşiğini taşıyabilir (promote_threshold; ör. çok komutlu
            dayanıklılık episode'ları), yoksa --threshold. Komut türü kapıları her zaman --threshold."""
            return float(getattr(levels[self.level], "promote_threshold", None) or args.threshold)

        def _axes_ok(self) -> bool:
            """--axis-gate: seviyedeki HER komut türü (tekrar edilenler dahil) ayrı ayrı eşiği geçmeli."""
            if not args.axis_gate:
                return True
            for ax, dq in self.axis_ok.items():
                if ax == "none":
                    continue
                if len(dq) < args.axis_min_commands or float(np.mean(dq)) < args.threshold:
                    return False
            return True

        def _axis_summary(self) -> str:
            short = {"interrupt": "kesen", "interrupted": "kesilen"}
            return " ".join(f"{short.get(ax, ax[:3])}={np.mean(dq):.0%}({len(dq)})" for ax, dq in sorted(self.axis_ok.items())
                            if ax != "none")

        def _on_rollout_end(self):
            rate = float(np.mean(self.recent)) if self.recent else 0.0
            n_term = sum(self.term.values())
            fails = {k: v / n_term for k, v in self.term.items() if k != "time_limit"} if n_term else {}
            self.logger.record("curriculum/level", self.level)
            self.logger.record("curriculum/success_rate", rate)
            self.logger.record("curriculum/episodes_at_level", self.n_eps)
            self.logger.record("curriculum/fail_rate", sum(fails.values()))
            if self.cmd_err:
                self.logger.record("curriculum/final_abs_heading_err", float(np.mean(self.cmd_err)))
            row = dict(timesteps=self.num_timesteps, wall_min=(time.time() - self.t0) / 60.0,
                       level=levels[self.level].name, episodes_at_level=self.n_eps, success_rate=rate,
                       fail_rate=sum(fails.values()), top_failure=max(fails, key=fails.get) if fails else "",
                       final_abs_heading_err=float(np.mean(self.cmd_err)) if self.cmd_err else float("nan"),
                       ep_rew_mean=float(np.mean([e["r"] for e in self.model.ep_info_buffer]))
                       if self.model.ep_info_buffer else float("nan"))
            new = not self.csv_path.exists()
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"[curriculum] {row['timesteps']:>9d} adım | {row['wall_min']:6.1f} dk | seviye {row['level']} | "
                  f"episode {self.n_eps:4d} | başarı(son {len(self.recent)}) {rate:5.1%} | "
                  f"düşme {row['fail_rate']:5.1%} {row['top_failure']} | ödül {row['ep_rew_mean']:.1f} | "
                  f"eksen: {self._axis_summary()}" + (f" | tekrar: " + " ".join(
                      f"{k}={np.mean(v):.0%}({len(v)})" for k, v in sorted(self.rehearse_ok.items())) if self.rehearse_ok else ""),
                  flush=True)

        # -------------------------------------------------------------
        def _evaluate(self):
            """Deterministik değerlendirme: --eval-levels'ın her birinden --eval-episodes episode (sabit seed'ler,
            her seferinde aynı). Ortalama episode başarısı en iyiyse models/best.zip. (Eğitimdeki başarı oranı
            stokastik policy'nindir; ince ayarda deterministik performans ondan farklı gidebilir.)"""
            self.last_eval = self.num_timesteps
            if self.eval_env is None:
                self.eval_env = make_env_fn(0, 0, args.env_config, args.task, args.env_overrides)()
            t0, rates = time.time(), {}
            names = [x.strip() for x in args.eval_levels.split(",") if x.strip()]
            cur_name = levels[self.level].name
            if args.promote_on_eval and cur_name not in names:
                names.append(cur_name)
            for lv in names:
                ok = 0
                for k in range(args.eval_episodes):
                    obs, _ = self.eval_env.reset(seed=90_000 + k, options=dict(level=lv))
                    done, info = False, {}
                    while not done:
                        obs, _, term, trunc, info = self.eval_env.step(self.model.predict(obs, deterministic=True)[0])
                        done = term or trunc
                    ok += bool(info.get("episode_success", False))
                rates[lv] = ok / max(1, args.eval_episodes)
            fixed = [x.strip() for x in args.eval_levels.split(",") if x.strip()]
            base = [rates[k] for k in fixed if k in rates] or list(rates.values())
            score = float(np.mean(base)) if base else 0.0          # best.zip: yalnızca --eval-levels ortalaması
            best = score > self.best_eval
            if best:
                self.best_eval = score
                (out / "models").mkdir(parents=True, exist_ok=True)
                self.model.save(out / "models" / "best.zip")
            row = dict(timesteps=self.num_timesteps, score=score, **{f"success_{k}": rates.get(k) for k in fixed})
            if args.promote_on_eval:                                # sabit sütunlar (seviye değişse de CSV bozulmasın)
                row.update(current_level=cur_name, success_current=rates.get(cur_name))
            path = out / "eval.csv"
            new = not path.exists()
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"[eval] {self.num_timesteps:>9d} adım | deterministik başarı " +
                  " ".join(f"{k}={v:.0%}" for k, v in rates.items()) +
                  f" | ortalama {score:.1%}{'  ← en iyi, best.zip' if best else ''} ({time.time() - t0:.0f} s)", flush=True)
            return rates.get(cur_name)

        def _save(self, tag: str):
            (out / "models").mkdir(parents=True, exist_ok=True)
            self.model.save(out / "models" / f"{tag}.zip")
            state = dict(level_index=self.level, level=levels[self.level].name,
                         timesteps=int(self.num_timesteps), history=self.history,
                         args=vars(args), env_config=env_config,
                         saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)

        def _promote(self) -> bool:
            lv = levels[self.level]
            rate = float(np.mean(self.recent))
            rec = dict(level=lv.name, level_index=self.level, success_rate=rate, episodes=self.n_eps,
                       axis_success={ax: float(np.mean(dq)) for ax, dq in self.axis_ok.items()},
                       steps_at_level=int(self.num_timesteps - self.level_start_steps),
                       total_steps=int(self.num_timesteps),
                       wall_min_at_level=(time.time() - self.level_start_time) / 60.0)
            self.history.append(rec)
            self._save(f"level_{self.level:02d}_{lv.name}")
            print(f"\n[curriculum] ✔ SEVİYE GEÇİLDİ: {lv.name} ({lv.description}) başarı {rate:.1%}, "
                  f"{rec['steps_at_level']} adım, {rec['wall_min_at_level']:.1f} dk\n", flush=True)
            last = len(levels) - 1 if args.stop_after is None else find_level(args.stop_after, levels)
            if self.level >= last:
                print("[curriculum] son seviye geçildi — eğitim bitiyor.")
                self._save("latest")
                return False
            self.level += 1
            self.training_env.env_method("set_level", self.level)
            self._apply_fine_tuning()
            self.recent.clear()
            self.cmd_err.clear()
            self.term.clear()
            self.axis_ok = {}
            self.rehearse_ok = {}
            self.n_eps = 0
            self.level_start_steps = self.num_timesteps
            self.level_start_time = time.time()
            self._save("latest")
            print(f"[curriculum] yeni seviye: {levels[self.level].name} — {levels[self.level].description}")
            return True

        def _on_training_end(self):
            self._save("latest")

    return CurriculumCallback()


# =====================================================================
# MAIN
# =====================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["command", "maneuver", "takeoff"], default="command",
                   help="command: Δ komut curriculum'u (H1…R1); maneuver: süre hedefli manevralar (M1…M5, S1…S4); "
                        "takeoff: yerden kalkış / hover / iniş, 4 kumanda doğrudan (K1…K9)")
    p.add_argument("--out", default="runs/command_curriculum", help="çıktı klasörü (Colab'da Drive önerilir)")
    p.add_argument("--total-steps", type=int, default=6_000_000,
                   help="üst sınır; son seviye geçilince eğitim kendiliğinden biter")
    p.add_argument("--level", default=None, help="başlangıç seviyesi (ad ya da indeks; varsayılan H1 / M1)")
    p.add_argument("--stop-after", default=None, help="bu seviye geçilince dur (ör. H2)")
    p.add_argument("--no-promote", dest="promote", action="store_false", help="seviye atlama kapalı")
    p.add_argument("--window", type=int, default=100, help="başarı oranı penceresi (episode)")
    p.add_argument("--min-episodes", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--no-axis-gate", dest="axis_gate", action="store_false",
                   help="seviye atlamada eksen başına başarı şartını kapat (v1 davranışı)")
    p.add_argument("--axis-min-commands", type=int, default=30,
                   help="eksen başına en az bu kadar komut sonucu olmadan seviye atlanmaz")
    p.add_argument("--resume", action="store_true", help="out/models/latest.zip + curriculum_state.json'dan devam")
    p.add_argument("--init-model", default=None, help="başka bir koşunun modelinden başla (ör. level_00_H1.zip)")
    p.add_argument("--n-envs", type=int, default=os.cpu_count() or 2)
    p.add_argument("--vec", choices=["subproc", "dummy"], default="subproc")
    p.add_argument("--env-config", choices=["v2", "v1"], default="v2",
                   help="reward/başarı ayarı: v2 (varsayılan, yumuşak kumanda + kuplaj sınırı) ya da v1 (ilk koşu)")
    p.add_argument("--coll-scale", type=float, default=None,
                   help="manevra: collective yetkisi trim ± bu (varsayılan: başlangıç modelininki, yoksa 0.25). "
                        "Model bu ayarı zip'inde taşır; mevcut bir modelin yetkisini genişletmek için widen_collective.py")
    p.add_argument("--seed", type=int, default=42)
    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048, help="env başına rollout uzunluğu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=None,
                   help="indirim çarpanı (varsayılan: komut 0.99, manevra 0.995 — manevra episode'ları daha uzun)")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--log-std-init", type=float, default=None,
                   help="başlangıç keşif gürültüsü: σ = e^x (varsayılan −1 → 0.37; kalkış −1.2 → 0.30)")
    p.add_argument("--fine-from", default=None,
                   help="bu seviyeden itibaren ince ayar (küçük lr + KL sınırı; varsayılan V1 / M3); 'none' → kapalı")
    p.add_argument("--fine-lr", type=float, default=1e-4)
    p.add_argument("--fine-kl", type=float, default=0.02)
    p.add_argument("--promote-on-eval", action="store_true",
                   help="mevcut seviyenin deterministik değerlendirme başarısı eşiği geçince de seviye atla "
                        "(değerlendirmeye mevcut seviye eklenir)")
    p.add_argument("--net", default="128,128", help="gizli katmanlar (pi ve vf için ayrı ağ)")
    p.add_argument("--save-freq", type=int, default=100_000, help="latest.zip kayıt aralığı (adım)")
    p.add_argument("--snapshot-freq", type=int, default=0,
                   help="bu kadar adımda bir models/snap_<adım>k.zip (0 = kapalı; en iyi ara modeli seçmek için)")
    p.add_argument("--eval-freq", type=int, default=0,
                   help="bu kadar adımda bir deterministik değerlendirme (sabit seed'ler); en iyisi models/best.zip")
    p.add_argument("--eval-levels", default=None,
                   help="değerlendirme seviyeleri (virgülle; varsayılan manevra M5,S2,S3 · kalkış K2,K4,K9)")
    p.add_argument("--eval-episodes", type=int, default=30, help="seviye başına değerlendirme episode'u")
    p.add_argument("--env-overrides", dest="env_overrides_cli", default=None,
                   help="env ayarları (JSON; kalkış / manevra), init / resume modelinin taşıdıklarının üstüne yazılır ve "
                        "modelle kaydedilir. Ör. kalkış: '{\"aircraft\": \"repo\", \"power_cap_psi\": 56, \"torque_obs\": true}'")
    p.add_argument("--smoke", action="store_true", help="çok kısa deneme koşusu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.level is None:
        args.level = {"maneuver": "M1", "takeoff": "K1"}.get(args.task, "H1")
    if args.fine_from is None:
        args.fine_from = {"maneuver": "M3", "takeoff": "K5"}.get(args.task, "V1")
    if args.gamma is None:
        args.gamma = 0.995 if args.task in ("maneuver", "takeoff") else 0.99
    if args.log_std_init is None:
        args.log_std_init = -1.2 if args.task == "takeoff" else -1.0     # kalkış: dört kumanda doğrudan, daha az gürültü
    if args.eval_levels is None:
        args.eval_levels = {"takeoff": "K2,K4,K9"}.get(args.task, "M5,S2,S3")
    if str(args.fine_from).lower() == "none":
        args.fine_from = None
    if args.smoke:
        args.total_steps, args.n_steps, args.batch_size, args.save_freq = 4096, 512, 128, 2048
        args.n_envs = min(args.n_envs, 2)

    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)          # işçi süreçleriyle çekirdek kavgası olmasın
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = task_levels(args.task)

    level_index = find_level(args.level, levels)
    history, model_path, state = [], None, {}
    if args.resume:
        state_file = out / "curriculum_state.json"
        latest = out / "models" / "latest.zip"
        if not (state_file.exists() and latest.exists()):
            raise FileNotFoundError(f"--resume için {state_file} ve {latest} gerekli")
        state = json.loads(state_file.read_text())
        saved_task = state.get("args", {}).get("task", "command")
        if saved_task != args.task:
            raise SystemExit(f"[resume] bu koşu --task {saved_task} ile başlamıştı; devam etmek için aynı --task'ı ver.")
        level_index, history = int(state["level_index"]), state.get("history", [])
        # Adım sayacı modelin içinde saklı (num_timesteps); reset_num_timesteps=False ile devam eder.
        model_path = latest
        print(f"[resume] {latest} — seviye {levels[level_index].name}, {state['timesteps']} adım")
    elif args.init_model:
        model_path = Path(args.init_model)

    if args.resume and state.get("args", {}).get("env_config") and state["args"]["env_config"] != args.env_config:
        print(f"[uyarı] bu koşu env_config={state['args']['env_config']} ile başlamıştı; şimdi {args.env_config}")
    # manevra env ayarları: modelin taşıdığı (ah1s_env_overrides) + komut satırı. Model bu ayarlarla kaydedilir.
    args.env_overrides = {}
    if args.task == "maneuver":
        from helicopter_env_maneuver import read_env_overrides
        inherited = read_env_overrides(model_path) if model_path is not None else {}
        args.env_overrides = dict(inherited)
        if args.coll_scale is not None:
            old = inherited.get("coll_scale", 0.25)
            if model_path is not None and abs(old - args.coll_scale) > 1e-9:
                print(f"[uyarı] model collective yetkisi ±{old:g} ile eğitilmiş; şimdi ±{args.coll_scale:g} → action ölçeği "
                      f"değişir. Davranışı koruyarak genişletmek için: python widen_collective.py ...")
            args.env_overrides["coll_scale"] = float(args.coll_scale)
        if args.env_overrides:
            print(f"[env] ayarlar: {args.env_overrides} (modelle birlikte kaydedilir)")
    elif args.coll_scale is not None:
        raise SystemExit("--coll-scale yalnızca --task maneuver için")
    if args.task == "takeoff":
        # kalkış env ayarları (2026-09-28: repo uçağı / kalibre yer etkisi / güç tavanı / tork gözlemi ve cezası):
        # modelin taşıdığı + --env-overrides. Model bu ayarlarla kaydedilir (evaluate_takeoff, command_viz okur).
        from helicopter_env_maneuver import read_env_overrides
        from helicopter_env_takeoff import TakeoffEnvConfig
        args.env_overrides = dict(read_env_overrides(model_path)) if model_path is not None else {}
    cli_ov = json.loads(args.env_overrides_cli) if args.env_overrides_cli else {}
    if cli_ov:
        if args.task not in ("takeoff", "maneuver"):
            raise SystemExit("--env-overrides yalnızca --task takeoff / maneuver için")
        args.env_overrides.update(cli_ov)
    if args.task == "takeoff":
        unknown = [k for k in args.env_overrides if k not in TakeoffEnvConfig.__dataclass_fields__]
        if unknown:
            raise SystemExit(f"bilinmeyen kalkış env ayarı: {unknown}")
        if args.env_overrides:
            print(f"[env] kalkış ayarları: {args.env_overrides} (modelle birlikte kaydedilir)")
    venv = build_vec_env(args.n_envs, level_index, args.vec, args.env_config, args.task, args.env_overrides)
    from dataclasses import asdict
    env_config = asdict(make_config(args.env_config, args.task, args.env_overrides))
    tb = None
    try:
        import tensorboard  # noqa: F401
        tb = str(out / "tb")
    except Exception:            # noqa: BLE001
        pass

    net = [int(x) for x in args.net.split(",") if x]
    if model_path is not None:
        model = PPO.load(str(model_path), env=venv, device="cpu", tensorboard_log=tb)
        # PPO.load hiperparametreleri dosyadan alır; bu koşu için komut satırındakileri uygula
        model.learning_rate = args.lr
        model.lr_schedule = lambda _: args.lr
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl
        # rollout / minibatch ayarları da komut satırından (ince ayarda daha büyük, daha sakin güncellemeler için)
        if model.n_steps != args.n_steps:
            from stable_baselines3.common.buffers import RolloutBuffer
            model.n_steps = args.n_steps
            model.rollout_buffer = RolloutBuffer(args.n_steps, model.observation_space, model.action_space,
                                                 device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
                                                 n_envs=model.n_envs)
        model.batch_size = args.batch_size
        model.n_epochs = args.n_epochs
        print(f"[model] yüklendi: {model_path}  (n_steps {model.n_steps}, batch {model.batch_size}, epoch {model.n_epochs})")
    else:
        model = PPO(
            "MlpPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5, target_kl=args.target_kl,
            policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=torch.nn.Tanh,
                               log_std_init=args.log_std_init),
            tensorboard_log=tb, seed=args.seed, verbose=0, device="cpu")
        print(f"[model] yeni PPO (rastgele ağırlıklar) — ağ {net}, σ0={np.exp(args.log_std_init):.2f}")
    if args.task in ("maneuver", "takeoff"):
        from helicopter_env_maneuver import ENV_OVERRIDES_ATTR
        setattr(model, ENV_OVERRIDES_ATTR, dict(args.env_overrides))      # her kayıtta zip'e girer

    print(f"[train] out={out}  n_envs={args.n_envs} ({args.vec})  toplam {args.total_steps} adım  "
          f"seviye atlama={'açık' if args.promote else 'KAPALI'} (eşik {args.threshold:.0%}, pencere {args.window})")
    cb = make_callback(out, levels, level_index, args, history, env_config)
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_steps, callback=cb, reset_num_timesteps=not args.resume,
                    tb_log_name={"maneuver": "ppo_man", "takeoff": "ppo_takeoff"}.get(args.task, "ppo_cmd"),
                    progress_bar=False)
    except KeyboardInterrupt:
        print("\n[train] durduruldu (Ctrl+C / Colab stop) — kaydediliyor...")
        cb._save("latest")
    finally:
        venv.close()
    print(f"[train] bitti: {(time.time() - t0) / 60:.1f} dk. Son seviye: {levels[cb.level].name}. "
          f"Geçilen seviyeler: {[h['level'] for h in cb.history]}")
    return model, cb


if __name__ == "__main__":
    main()
