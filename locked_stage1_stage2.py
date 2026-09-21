from __future__ import annotations

"""
LOCKED STAGE 1 / STAGE 2 DEFINITIONS
====================================

Kilitli (değiştirilmemesi gereken) Stage 1 ve Stage 2 modelleri ile,
Stage 1 -> Stage 2 handoff'u için gereken küçük yardımcılar.

Önceden bu tanımlar, `deneme/diagnose_stage4_entry_margin_v3.py` adlı
1200 satırlık bir Stage-4 iniş diagnostic dosyasının ilk kısmı `exec` ile
çalıştırılarak alınıyordu. Aşağıdaki kod o dosyadan birebir (verbatim)
kopyalanmıştır; yalnızca final sistemin kullandığı isimler tutulmuştur.
Kullanılmayan Stage 3 modeli artık yüklenmez ve diagnostic sonuç klasörleri
artık oluşturulmaz.

Kullananlar:
    validate_live_multiturn_same_fdm_v1.py
    training/train_turn_live_entry_v12_residual_all_targets.py
"""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine_mapped import HelicopterEnvStage2RefineMapped


# =====================================================================
# LOCKED INPUTS — DO NOT MODIFY
# =====================================================================

STAGE1_MODEL_PATH = Path(
    "models_stage1_early_distilled/AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = Path(
    "models_stage2_hybrid_final/AH1S_STAGE2_HYBRID_FINAL.zip"
)

for required in [
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
]:
    if not required.exists():
        raise FileNotFoundError(f"Required locked input missing: {required}")


# =====================================================================
# REPRODUCIBILITY / MISSION CONSTANTS
# =====================================================================

SEED = 42
np.random.seed(SEED)

STAGE1_MAX_TIME = 120.0

HANDOFF_STABLE_TIME = 5.0
DEFAULT_CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040


# =====================================================================
# MODELS
# =====================================================================

stage1_model = PPO.load(str(STAGE1_MODEL_PATH))
stage2_model = PPO.load(str(STAGE2_MODEL_PATH))


# =====================================================================
# SMALL HELPERS
# =====================================================================

def info_float(info, key, default=float("nan")) -> float:
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def get_fdm(env):
    if getattr(env, "fdm", None) is not None:
        return env.fdm
    base = getattr(env, "base_env", None)
    if base is not None and getattr(base, "fdm", None) is not None:
        return base.fdm
    raise RuntimeError("Active FDM not found.")


def env_control_dt(env) -> float:
    dt = float(getattr(env, "dt", DEFAULT_CONTROL_DT) or DEFAULT_CONTROL_DT)
    if not np.isfinite(dt) or dt <= 0.0:
        dt = DEFAULT_CONTROL_DT
    return dt
