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


def make_config(name: str):
    from helicopter_env_command import CommandEnvConfig
    return CommandEnvConfig.v1() if name == "v1" else CommandEnvConfig()


def make_env_fn(rank: int, level_index: int, config_name: str):
    def _init():
        # SubprocVecEnv işçisinde de repo kökü import yolunda olsun
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from helicopter_env_command import HelicopterEnvCommand
        return HelicopterEnvCommand(level=level_index, config=make_config(config_name))
    return _init


def build_vec_env(n_envs: int, level_index: int, vec: str, config_name: str = "v2"):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    fns = [make_env_fn(i, level_index, config_name) for i in range(n_envs)]
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
            self.history = list(history)
            self.last_save = 0
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
            self._apply_fine_tuning()
            print(f"[curriculum] başlangıç seviyesi: {levels[self.level].name} — {levels[self.level].description}")

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if not done or info.get("level_index") != self.level:
                    continue
                self.recent.append(bool(info.get("episode_success", False)))
                self.n_eps += 1
                self.term[info.get("termination", "?")] += 1
                for res in info.get("command_results") or []:
                    self.cmd_err.append(abs(float(res["final_err"]["heading"])))
                    ax = command_axis(res["cmd"])
                    self.axis_ok.setdefault(ax, deque(maxlen=args.window)).append(bool(res["success"]))
            if self.num_timesteps - self.last_save >= args.save_freq:
                self._save("latest")
                self.last_save = self.num_timesteps
            if (args.promote and len(self.recent) >= args.window and self.n_eps >= args.min_episodes
                    and float(np.mean(self.recent)) >= args.threshold and self._axes_ok()):
                return self._promote()
            return True

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
            return " ".join(f"{ax[:3]}={np.mean(dq):.0%}({len(dq)})" for ax, dq in sorted(self.axis_ok.items())
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
                  f"eksen: {self._axis_summary()}", flush=True)

        # -------------------------------------------------------------
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
    p.add_argument("--out", default="runs/command_curriculum", help="çıktı klasörü (Colab'da Drive önerilir)")
    p.add_argument("--total-steps", type=int, default=6_000_000,
                   help="üst sınır; son seviye geçilince eğitim kendiliğinden biter")
    p.add_argument("--level", default="H1", help="başlangıç seviyesi (ad ya da indeks)")
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
    p.add_argument("--seed", type=int, default=42)
    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048, help="env başına rollout uzunluğu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--log-std-init", type=float, default=-1.0, help="başlangıç keşif gürültüsü: σ = e^x (−1 → 0.37)")
    p.add_argument("--fine-from", default="V1",
                   help="bu seviyeden itibaren ince ayar (küçük lr + KL sınırı); 'none' → kapalı")
    p.add_argument("--fine-lr", type=float, default=1e-4)
    p.add_argument("--fine-kl", type=float, default=0.02)
    p.add_argument("--net", default="128,128", help="gizli katmanlar (pi ve vf için ayrı ağ)")
    p.add_argument("--save-freq", type=int, default=100_000, help="latest.zip kayıt aralığı (adım)")
    p.add_argument("--smoke", action="store_true", help="çok kısa deneme koşusu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
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
    levels = DEFAULT_LEVELS

    level_index = find_level(args.level, levels)
    history, model_path, state = [], None, {}
    if args.resume:
        state_file = out / "curriculum_state.json"
        latest = out / "models" / "latest.zip"
        if not (state_file.exists() and latest.exists()):
            raise FileNotFoundError(f"--resume için {state_file} ve {latest} gerekli")
        state = json.loads(state_file.read_text())
        level_index, history = int(state["level_index"]), state.get("history", [])
        # Adım sayacı modelin içinde saklı (num_timesteps); reset_num_timesteps=False ile devam eder.
        model_path = latest
        print(f"[resume] {latest} — seviye {levels[level_index].name}, {state['timesteps']} adım")
    elif args.init_model:
        model_path = Path(args.init_model)

    if args.resume and state.get("args", {}).get("env_config") and state["args"]["env_config"] != args.env_config:
        print(f"[uyarı] bu koşu env_config={state['args']['env_config']} ile başlamıştı; şimdi {args.env_config}")
    venv = build_vec_env(args.n_envs, level_index, args.vec, args.env_config)
    from dataclasses import asdict
    env_config = asdict(make_config(args.env_config))
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
        print(f"[model] yüklendi: {model_path}")
    else:
        model = PPO(
            "MlpPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5, target_kl=args.target_kl,
            policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=torch.nn.Tanh,
                               log_std_init=args.log_std_init),
            tensorboard_log=tb, seed=args.seed, verbose=0, device="cpu")
        print(f"[model] yeni PPO (rastgele ağırlıklar) — ağ {net}, σ0={np.exp(args.log_std_init):.2f}")

    print(f"[train] out={out}  n_envs={args.n_envs} ({args.vec})  toplam {args.total_steps} adım  "
          f"seviye atlama={'açık' if args.promote else 'KAPALI'} (eşik {args.threshold:.0%}, pencere {args.window})")
    cb = make_callback(out, levels, level_index, args, history, env_config)
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_steps, callback=cb, reset_num_timesteps=not args.resume,
                    tb_log_name="ppo_cmd", progress_bar=False)
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
