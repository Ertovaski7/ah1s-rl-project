"""BC buffer'ı topla (2026-10-03; ppo_bc.py): dondurulmuş öğretmen modelin korunacak görevlerdeki durumları.

Öğretmen seviyelerden örneklenen episode'ları uçar (action: öğretmenin kendi keşif gürültüsüyle → öğretmenin
yörüngesinin çevresindeki durumlar da gelir; etiket her zaman öğretmenin action ORTALAMASI). Yalnızca korunacak görev
türlerinin durumları kaydedilir; görev türü başına üst sınır (dengeli buffer).

Varsayılan korunacaklar (mükemmellik eğitimi ileri uçuş irtifa tutma + tırmanışı eğitir, bunlar hariç):
  land (iniş, yerdeki oturma dahil), hold, turn, move, pirouette, stop, recover, depart ve kalkışın kızak < 15 ft kısmı.
Hariç: cruise (ileri uçuş Δ'ları, hızlanma), climb_to, bob, kalkışın 15 ft üstü.

Kullanım:
  python docs/flight/collect_bc_buffer.py --model models_flight/flight_v4.zip --out runs/bc/buffer_v4.npz --n 150000
"""
import argparse, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch as th
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight

PROTECT = ("land", "hold", "turn", "move", "pirouette", "stop", "recover", "depart")
LEVELS = ("F6a", "F6", "F14", "F11", "F13", "F10", "F8", "F5", "F3", "F7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150000); ap.add_argument("--cap-frac", type=float, default=0.3,
                    help="görev türü başına en çok bu kesir")
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--protect", default=",".join(PROTECT)); ap.add_argument("--takeoff-hs", type=float, default=15.0)
    a = ap.parse_args()
    protect = set(a.protect.split(","))
    m = PPO.load(a.model, device="cpu")
    ov = dict(getattr(m, "ah1s_env_overrides", None) or {})
    env = HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))
    cap = int(a.cap_frac * a.n)
    obs_l, mu_l, kind_l, kinds = [], [], [], Counter()
    t0, ep = time.time(), 0
    while sum(kinds.values()) < a.n:
        lvl = LEVELS[ep % len(LEVELS)]
        try:
            obs, info = env.reset(seed=a.seed0 + ep, options=dict(level=lvl))
        except RuntimeError:
            ep += 1; continue
        done = False
        while not done:
            with th.no_grad():
                dist = m.policy.get_distribution(th.as_tensor(obs[None], dtype=th.float32))
                mu = dist.distribution.mean.numpy()[0]
                act = dist.sample().numpy()[0]
            w = env.windows[-1] if env.windows else None
            kind = (w["kind"] if w is not None and not w["closed"] else "hold")
            hs = float(info.get("skid_height", 0.0))
            keep = kind in protect or (kind == "takeoff" and hs < a.takeoff_hs) or (w is None and info.get("wow", 0) > 0)
            if keep and kinds[kind] < cap:
                obs_l.append(obs.copy()); mu_l.append(mu); kind_l.append(kind); kinds[kind] += 1
            obs, r, te, tr, info = env.step(np.clip(act, -1.0, 1.0))
            done = te or tr
        ep += 1
        if ep % 10 == 0:
            print(f"{ep} episode, {sum(kinds.values())} durum, {dict(kinds)} ({time.time() - t0:.0f} s)", flush=True)
    sigma = np.exp(m.policy.log_std.detach().numpy())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, obs=np.asarray(obs_l, dtype=np.float32), mu=np.asarray(mu_l, dtype=np.float32), sigma=sigma,
                        kinds=np.array(kind_l), teacher=str(a.model))
    print(f"kaydedildi: {a.out}  {len(obs_l)} durum, σ_öğretmen {np.round(sigma, 3)}, türler {dict(kinds)}")


if __name__ == "__main__":
    main()
