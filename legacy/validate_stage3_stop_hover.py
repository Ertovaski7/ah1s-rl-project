from __future__ import annotations

"""
STAGE 3 — ENDPOINT STOP + HOVER VALIDATOR
=========================================

Görev zinciri (hepsi AYNI canlı JSBSim FDM üzerinde, reset / teleport yok,
runtime teacher OFF):

    Stage 1  kalkış -> ~300 ft stabil hover (5 s)
    Stage 2  ileri uçuş, 80 ft ileriye kadar
    Stage 3  kilitli Stage 3 PPO: 300 ft ilerideki hedef noktada frenleyip
             durma + 5 s stabil hover

Stage 3 başarı ölçütü (5 s boyunca aynı anda):
    |hedefe konum hatası| <= 5 ft ve |ileri hız| <= 0.6 ft/s
    |cross-track| <= 5 ft ve |lateral hız| <= 0.6 ft/s
    295 <= irtifa <= 305 ft, |vertical speed| <= 0.75 ft/s, |heading hatası| <= 1°

Kaynak: Kilitli Stage 3'ü tekrar üreten kod, Stage 4 çalışmalarında
deneme/diagnose_stage4_entry_margin_v3.py içinde duruyordu (bu dosya
temizlikte kaldırıldı). Aşağıdaki yardımcılar ve build_stage4_handoff()
oradan BİREBİR alınmıştır; yalnızca fonksiyon adı
build_stage3_hover_handoff() olarak değiştirildi ve Stage 4'e atıf yapan
birkaç hata mesajı sadeleştirildi (kontrol mantığı aynı). Stage 1/2 modelleri
ve ortak yardımcılar locked_stage1_stage2.py'den gelir.

build_stage3_hover_handoff() canlı FDM'i (env1, env2, fdm) hedef noktada
stabil hover'da geri döndürür; yeni bir curriculum (ör. iniş, irtifa
değişimi) bu state'ten başlatılabilir. İş bitince close_handoff(start)
çağrılmalıdır.

Kullanım:
    python validate_stage3_stop_hover.py
"""

import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from locked_stage1_stage2 import (
    AILERON_SCALE,
    HANDOFF_STABLE_TIME,
    RUDDER_SCALE,
    STAGE1_MAX_TIME,
    HelicopterEnvStage1Distill,
    HelicopterEnvStage2RefineMapped,
    env_control_dt,
    get_fdm,
    info_float,
    stage1_model,
    stage2_model,
)


# =====================================================================
# LOCKED STAGE 3 MODEL
# =====================================================================

STAGE3_MODEL_PATH = Path(
    "models_stage3_hybrid_final/AH1S_STAGE3_HYBRID_FINAL.zip"
)

if not STAGE3_MODEL_PATH.exists():
    raise FileNotFoundError(f"Required locked input missing: {STAGE3_MODEL_PATH}")

stage3_model = PPO.load(str(STAGE3_MODEL_PATH))


# =====================================================================
# MISSION CONSTANTS (kilitli Stage 3 değerleri)
# =====================================================================

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 90.0

STAGE3_TAKEOVER_FORWARD_FT = 80.0

EARTH_RADIUS_FT = 20_902_231.0

# Endpoint stop / hover qualification.
STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 0.60
STOP_HOLD_SECONDS = 5.0
CROSS_TOL_FT = 5.0
LAT_SPEED_TOL_FPS = 0.60
ALT_MIN = 295.0
ALT_MAX = 305.0
VS_TOL_FPS = 0.75
HEADING_TOL_DEG = 1.0

# Stage-3 safety envelope.
STAGE3_ALT_SAFE_MIN = 288.0
STAGE3_ALT_SAFE_MAX = 312.0
STAGE3_MAX_ABS_PITCH_DEG = 8.0
STAGE3_MAX_ABS_ROLL_DEG = 10.0
STAGE3_MAX_CROSS_SAFE_FT = 15.0


# =====================================================================
# SMALL HELPERS
# =====================================================================

def fdm_float(fdm, key, default=float("nan")) -> float:
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def first_finite(fdm, keys, default=float("nan")) -> float:
    for key in keys:
        value = fdm_float(fdm, key)
        if np.isfinite(value):
            return value
    return float(default)


def latitude_deg(fdm) -> float:
    return first_finite(fdm, ["position/lat-gc-deg", "position/lat-geod-deg"])


def longitude_deg(fdm) -> float:
    return first_finite(fdm, ["position/long-gc-deg", "position/long-geod-deg"])


def heading_rad(fdm) -> float:
    return first_finite(fdm, ["attitude/heading-true-rad", "attitude/psi-rad"])


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def local_ne_ft(lat, lon, lat0, lon0):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = EARTH_RADIUS_FT * dlat
    east = EARTH_RADIUS_FT * math.cos(math.radians(lat0)) * dlon
    return float(north), float(east)


def mission_axes(north, east, heading):
    c = math.cos(heading)
    s = math.sin(heading)
    forward = north * c + east * s
    cross = -north * s + east * c
    return float(forward), float(cross)


def geometry(fdm, lat0, lon0, mission_heading):
    north, east = local_ne_ft(
        latitude_deg(fdm),
        longitude_deg(fdm),
        lat0,
        lon0,
    )
    return mission_axes(north, east, mission_heading)


def mission_ground_velocity(fdm, mission_heading):
    vn = first_finite(fdm, ["velocities/v-north-fps"])
    ve = first_finite(fdm, ["velocities/v-east-fps"])
    if not np.isfinite(vn) or not np.isfinite(ve):
        return float("nan"), float("nan")
    return mission_axes(vn, ve, mission_heading)


def physical_commands(fdm):
    return {
        "physical_collective_cmd": fdm_float(fdm, "fcs/collective-cmd-norm"),
        "physical_elevator_cmd": fdm_float(fdm, "fcs/elevator-cmd-norm"),
        "physical_aileron_cmd": fdm_float(fdm, "fcs/aileron-cmd-norm"),
        "physical_rudder_cmd": fdm_float(fdm, "fcs/rudder-cmd-norm"),
    }


def snapshot(fdm, lat0, lon0, mission_heading):
    forward, cross = geometry(fdm, lat0, lon0, mission_heading)
    heading_error = wrap_angle(heading_rad(fdm) - mission_heading)
    ground_fwd, ground_cross = mission_ground_velocity(fdm, mission_heading)

    state = {
        "forward_ft": float(forward),
        "position_error_ft": float(TARGET_FORWARD_FT - forward),
        "cross_track_ft": float(cross),
        "altitude_ft": fdm_float(fdm, "position/h-agl-ft"),
        "forward_speed_fps": fdm_float(fdm, "velocities/u-aero-fps", 0.0),
        "lateral_speed_fps": fdm_float(fdm, "velocities/v-aero-fps", 0.0),
        "mission_ground_forward_speed_fps": float(ground_fwd),
        "mission_ground_cross_speed_fps": float(ground_cross),
        "vertical_speed_fps": fdm_float(fdm, "velocities/h-dot-fps", 0.0),
        "heading_error_rad": float(heading_error),
        "heading_error_deg": float(math.degrees(heading_error)),
        "pitch_rad": fdm_float(fdm, "attitude/pitch-rad", 0.0),
        "roll_rad": fdm_float(fdm, "attitude/roll-rad", 0.0),
        "roll_rate_rad_s": fdm_float(fdm, "velocities/p-rad_sec", 0.0),
        "pitch_rate_rad_s": fdm_float(fdm, "velocities/q-rad_sec", 0.0),
        "yaw_rate_rad_s": fdm_float(fdm, "velocities/r-rad_sec", 0.0),
        "rotor_rpm": fdm_float(fdm, "propulsion/engine/rotor-rpm", 323.0),
    }
    state.update(physical_commands(fdm))
    return state


def stage3_observation(state):
    obs = np.array(
        [
            (state["altitude_ft"] - TARGET_ALT_FT) / 10.0,
            state["vertical_speed_fps"] / 5.0,
            state["position_error_ft"] / 250.0,
            state["forward_speed_fps"] / 12.0,
            state["cross_track_ft"] / 10.0,
            state["lateral_speed_fps"] / 5.0,
            math.sin(state["heading_error_rad"]),
            math.cos(state["heading_error_rad"]),
            state["pitch_rad"] / 0.20,
            state["roll_rad"] / 0.20,
            state["roll_rate_rad_s"] / 1.0,
            state["pitch_rate_rad_s"] / 1.0,
            state["yaw_rate_rad_s"] / 1.0,
            state["rotor_rpm"] / 400.0,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -10.0, +10.0).astype(np.float32)


def endpoint_stop_now(state) -> bool:
    return bool(
        abs(state["position_error_ft"]) <= STOP_POS_TOL_FT
        and abs(state["forward_speed_fps"]) <= STOP_SPEED_TOL_FPS
    )


def endpoint_lateral_now(state) -> bool:
    return bool(
        abs(state["cross_track_ft"]) <= CROSS_TOL_FT
        and abs(state["lateral_speed_fps"]) <= LAT_SPEED_TOL_FPS
    )


def endpoint_hover_now(state) -> bool:
    return bool(
        endpoint_stop_now(state)
        and endpoint_lateral_now(state)
        and ALT_MIN <= state["altitude_ft"] <= ALT_MAX
        and abs(state["vertical_speed_fps"]) <= VS_TOL_FPS
        and abs(state["heading_error_deg"]) <= HEADING_TOL_DEG
    )


def stage3_safety_reason(state) -> str:
    if state["altitude_ft"] < STAGE3_ALT_SAFE_MIN:
        return "altitude_below_stage3_safe"
    if state["altitude_ft"] > STAGE3_ALT_SAFE_MAX:
        return "altitude_above_stage3_safe"
    if abs(math.degrees(state["pitch_rad"])) > STAGE3_MAX_ABS_PITCH_DEG:
        return "pitch_stage3_limit"
    if abs(math.degrees(state["roll_rad"])) > STAGE3_MAX_ABS_ROLL_DEG:
        return "roll_stage3_limit"
    if abs(state["cross_track_ft"]) > STAGE3_MAX_CROSS_SAFE_FT:
        return "cross_stage3_limit"
    return ""


def physics_steps(env) -> int:
    try:
        value = int(getattr(env, "PHYSICS_STEPS", 10))
    except Exception:
        value = 10
    return max(1, value)


def close_handoff(start) -> None:
    env2 = start.get("env2")
    env1 = start.get("env1")
    if env2 is not None:
        try:
            env2.fdm = None
        except Exception:
            pass
    if env1 is not None:
        try:
            env1.close()
        except Exception:
            pass


# =====================================================================
# RAW STAGE-3 PHYSICS CYCLE
# =====================================================================

def raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    # Mapped Stage-2 actuator path: this is the already validated wiring.
    env2._apply_action(action)

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError("JSBSim stopped during Stage-3 flight.")

    if hasattr(env2, "previous_action"):
        try:
            env2.previous_action = action.copy()
        except Exception:
            pass

    # Match the validated Stage-3 raw-cycle bookkeeping order.
    state = snapshot(fdm, lat0, lon0, mission_heading)

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(state["forward_ft"])

    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass

    # Refresh Stage-2 bookkeeping even though Stage 3/4 uses its own obs.
    try:
        env2._get_obs()
    except Exception:
        pass

    return state, action


# =====================================================================
# TRUE CONTINUOUS STAGE1 -> STAGE2 -> LOCKED STAGE3 HANDOFF
# =====================================================================

def build_stage3_hover_handoff(
    detailed=False,
    require_full_hold=True,
    custom_entry_forward_ft=None,
    custom_entry_max_speed_fps=1.0,
):
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )
    obs1, info1 = env1.reset()

    fdm = get_fdm(env1)
    active_fdm_id = id(fdm)
    mission_heading = heading_rad(fdm)
    dt1 = env_control_dt(env1)

    sim_time_at_stage1_start = first_finite(
        fdm,
        ["simulation/sim-time-sec"],
    )

    stable_time = 0.0
    stage1_elapsed = 0.0

    for _ in range(int(STAGE1_MAX_TIME / dt1)):
        action1, _ = stage1_model.predict(obs1, deterministic=True)
        obs1, _, terminated, truncated, info1 = env1.step(action1)
        stage1_elapsed += dt1

        altitude = info_float(info1, "altitude")
        vertical_speed = info_float(info1, "vertical_speed")
        vn = info_float(info1, "vn", 0.0)
        ve = info_float(info1, "ve", 0.0)
        horizontal_speed = float(np.hypot(vn, ve))
        drift = info_float(info1, "drift", 999.0)

        stable = bool(
            295.0 <= altitude <= 305.0
            and abs(vertical_speed) <= 0.50
            and horizontal_speed <= 1.0
            and drift <= 3.0
        )
        stable_time = stable_time + dt1 if stable else 0.0

        if stable_time >= HANDOFF_STABLE_TIME:
            break

        if terminated and not bool(info1.get("success", False)):
            env1.close()
            raise RuntimeError("Stage 1 failed before stable handoff.")
        if truncated:
            env1.close()
            raise RuntimeError("Stage 1 truncated before stable handoff.")

    if stable_time < HANDOFF_STABLE_TIME:
        env1.close()
        raise RuntimeError("Stable Stage-1 handoff was not reached.")

    # Mission geometry origin = true Stage-1 hover handoff.
    lat0 = latitude_deg(fdm)
    lon0 = longitude_deg(fdm)

    sim_time_before_attach = first_finite(fdm, ["simulation/sim-time-sec"])

    env2 = HelicopterEnvStage2RefineMapped(
        aileron_scale=AILERON_SCALE,
        rudder_scale=RUDDER_SCALE,
    )

    # Disposable reset initializes Python-side bookkeeping only.
    env2.reset()
    env2.fdm = fdm

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = 0.0
    if hasattr(env2, "target_heading"):
        env2.target_heading = float(mission_heading)

    for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
        if hasattr(env2, attr):
            setattr(env2, attr, 0)

    if id(get_fdm(env2)) != active_fdm_id:
        env2.fdm = None
        env1.close()
        raise RuntimeError("FDM continuity failed during Stage-2 attach.")

    sim_time_after_attach = first_finite(fdm, ["simulation/sim-time-sec"])
    clock_reset_on_attach = bool(
        np.isfinite(sim_time_before_attach)
        and np.isfinite(sim_time_after_attach)
        and abs(sim_time_after_attach - sim_time_before_attach) > 1e-9
    )
    if clock_reset_on_attach:
        env2.fdm = None
        env1.close()
        raise RuntimeError("Simulation clock changed during Stage-2 attach.")

    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
    dt2 = env_control_dt(env2)
    stage2_elapsed = 0.0

    reached_stage3_takeover = False

    for _ in range(int(STAGE2_MAX_TIME / dt2)):
        action2, _ = stage2_model.predict(obs2, deterministic=True)
        action2 = np.asarray(action2, dtype=np.float32).reshape(-1)

        obs2, _, terminated, truncated, info2 = env2.step(action2)
        obs2 = np.asarray(obs2, dtype=np.float32)
        stage2_elapsed += dt2

        forward, _cross = geometry(fdm, lat0, lon0, mission_heading)
        if forward >= STAGE3_TAKEOVER_FORWARD_FT:
            if hasattr(env2, "forward_distance"):
                env2.forward_distance = float(forward)
            reached_stage3_takeover = True
            break

        if terminated and not bool(info2.get("success", False)):
            env2.fdm = None
            env1.close()
            raise RuntimeError("Stage 2 failed before Stage-3 takeover.")
        if truncated:
            env2.fdm = None
            env1.close()
            raise RuntimeError("Stage 2 truncated before Stage-3 takeover.")

    if not reached_stage3_takeover:
        env2.fdm = None
        env1.close()
        raise RuntimeError("Stage-3 takeover distance was not reached.")

    # Refresh bookkeeping after the true forward-distance correction.
    try:
        env2._get_obs()
    except Exception:
        pass

    dt3 = dt2
    hover_hold = 0.0
    stage3_elapsed = 0.0

    min_alt = +999.0
    max_alt = -999.0
    max_cross = 0.0
    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    max_abs_heading = 0.0

    next_print = 0.0
    state = snapshot(fdm, lat0, lon0, mission_heading)

    for step in range(int(STAGE3_MAX_TIME / dt3)):
        obs3 = stage3_observation(state)
        action3, _ = stage3_model.predict(obs3, deterministic=True)
        action3 = np.asarray(action3, dtype=np.float32).reshape(-1)

        state, _used = raw_policy_cycle(
            env2,
            fdm,
            action3,
            lat0,
            lon0,
            mission_heading,
        )

        stage3_elapsed = (step + 1) * dt3

        min_alt = min(min_alt, state["altitude_ft"])
        max_alt = max(max_alt, state["altitude_ft"])
        max_cross = max(max_cross, abs(state["cross_track_ft"]))
        max_abs_pitch = max(max_abs_pitch, abs(math.degrees(state["pitch_rad"])))
        max_abs_roll = max(max_abs_roll, abs(math.degrees(state["roll_rad"])))
        max_abs_heading = max(max_abs_heading, abs(state["heading_error_deg"]))

        reason = stage3_safety_reason(state)
        if reason:
            env2.fdm = None
            env1.close()
            raise RuntimeError(f"Locked Stage 3 violated safety before handoff: {reason}")

        hover_hold = hover_hold + dt3 if endpoint_hover_now(state) else 0.0

        if detailed and stage3_elapsed + 1e-9 >= next_print:
            print(
                f"Stage3 t={stage3_elapsed:6.2f}s | "
                f"FWD={state['forward_ft']:7.2f} | "
                f"V={state['forward_speed_fps']:+6.2f} | "
                f"X={state['cross_track_ft']:+6.2f} | "
                f"LAT={state['lateral_speed_fps']:+6.2f} | "
                f"ALT={state['altitude_ft']:7.2f} | "
                f"VS={state['vertical_speed_fps']:+6.2f} | "
                f"HOLD={hover_hold:4.2f}s"
            )
            next_print += 10.0

        if require_full_hold:
            if hover_hold >= STOP_HOLD_SECONDS:
                break
        else:
            # require_full_hold=False: early handoff option used by the Stage-4
            # (landing) experiments.  Without a custom trigger, stop at the first
            # state inside the locked endpoint envelope; with one, hand over
            # slightly earlier during the braking approach.
            if custom_entry_forward_ft is None:
                entry_ok = endpoint_hover_now(state)
            else:
                entry_ok = bool(
                    state["forward_ft"] >= float(custom_entry_forward_ft)
                    and state["forward_ft"] <= TARGET_FORWARD_FT + STOP_POS_TOL_FT
                    and abs(state["forward_speed_fps"]) <= float(custom_entry_max_speed_fps)
                    and abs(state["cross_track_ft"]) <= CROSS_TOL_FT
                    and abs(state["lateral_speed_fps"]) <= LAT_SPEED_TOL_FPS
                    and ALT_MIN <= state["altitude_ft"] <= ALT_MAX
                    and abs(state["vertical_speed_fps"]) <= VS_TOL_FPS
                    and abs(state["heading_error_deg"]) <= HEADING_TOL_DEG
                )
            if entry_ok:
                break

    if require_full_hold:
        handoff_pass = bool(hover_hold >= STOP_HOLD_SECONDS and endpoint_hover_now(state))
    elif custom_entry_forward_ft is None:
        handoff_pass = bool(endpoint_hover_now(state))
    else:
        handoff_pass = bool(
            state["forward_ft"] >= float(custom_entry_forward_ft)
            and state["forward_ft"] <= TARGET_FORWARD_FT + STOP_POS_TOL_FT
            and abs(state["forward_speed_fps"]) <= float(custom_entry_max_speed_fps)
            and abs(state["cross_track_ft"]) <= CROSS_TOL_FT
            and abs(state["lateral_speed_fps"]) <= LAT_SPEED_TOL_FPS
            and ALT_MIN <= state["altitude_ft"] <= ALT_MAX
            and abs(state["vertical_speed_fps"]) <= VS_TOL_FPS
            and abs(state["heading_error_deg"]) <= HEADING_TOL_DEG
        )

    if not handoff_pass:
        env2.fdm = None
        env1.close()
        if require_full_hold:
            raise RuntimeError(
                "Locked Stage-3 final PPO did not reproduce the 5 s endpoint hover."
            )
        raise RuntimeError(
            "Locked Stage-3 PPO never reached the requested early-handoff state."
        )

    sim_time_handoff = first_finite(fdm, ["simulation/sim-time-sec"])

    return {
        "env1": env1,
        "env2": env2,
        "fdm": fdm,
        "lat0": float(lat0),
        "lon0": float(lon0),
        "mission_heading": float(mission_heading),
        "active_fdm_id": int(active_fdm_id),
        "same_fdm": bool(id(fdm) == active_fdm_id and id(get_fdm(env2)) == active_fdm_id),
        "clock_reset_on_attach": bool(clock_reset_on_attach),
        "sim_time_at_stage1_start": float(sim_time_at_stage1_start),
        "sim_time_handoff": float(sim_time_handoff),
        "stage1_elapsed_s": float(stage1_elapsed),
        "stage2_elapsed_s": float(stage2_elapsed),
        "stage3_elapsed_s": float(stage3_elapsed),
        "stage3_hover_hold_s": float(hover_hold),
        "stage3_min_alt_ft": float(min_alt),
        "stage3_max_alt_ft": float(max_alt),
        "stage3_max_cross_ft": float(max_cross),
        "stage3_max_abs_pitch_deg": float(max_abs_pitch),
        "stage3_max_abs_roll_deg": float(max_abs_roll),
        "stage3_max_abs_heading_deg": float(max_abs_heading),
        "state": state.copy(),
        "handoff_pass": bool(handoff_pass),
    }


# =====================================================================
# COMMAND-LINE VALIDATION
# =====================================================================

def main() -> int:
    print("=" * 120)
    print("STAGE 3 — ENDPOINT STOP + HOVER")
    print("Stage1 -> Stage2 -> Stage3 (locked PPO) | SAME live FDM | runtime teacher OFF")
    print(f"Target: stop {TARGET_FORWARD_FT:.0f} ft ahead at {TARGET_ALT_FT:.0f} ft AGL, hold hover {STOP_HOLD_SECONDS:.0f} s")
    print("=" * 120)

    try:
        start = build_stage3_hover_handoff(detailed=True)
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}")
        print("STAGE 3 STOP + HOVER: FAIL")
        return 1

    try:
        s = start["state"]
        ok = bool(
            start["handoff_pass"]
            and start["same_fdm"]
            and not start["clock_reset_on_attach"]
        )

        print("\n" + "=" * 120)
        print("STAGE 3 SUMMARY")
        print("=" * 120)
        print(
            f"phase times        : stage1={start['stage1_elapsed_s']:.2f}s | "
            f"stage2={start['stage2_elapsed_s']:.2f}s | stage3={start['stage3_elapsed_s']:.2f}s"
        )
        print(
            f"final position     : forward={s['forward_ft']:.2f} ft | "
            f"error={s['position_error_ft']:+.2f} ft | cross={s['cross_track_ft']:+.2f} ft"
        )
        print(
            f"final speeds       : forward={s['forward_speed_fps']:+.3f} ft/s | "
            f"lateral={s['lateral_speed_fps']:+.3f} ft/s | vertical={s['vertical_speed_fps']:+.3f} ft/s"
        )
        print(
            f"final attitude     : alt={s['altitude_ft']:.2f} ft | "
            f"heading_error={s['heading_error_deg']:+.3f} deg"
        )
        print(
            f"stage3 envelope    : alt=[{start['stage3_min_alt_ft']:.2f}, {start['stage3_max_alt_ft']:.2f}] ft | "
            f"max|cross|={start['stage3_max_cross_ft']:.2f} ft | "
            f"max|pitch|={start['stage3_max_abs_pitch_deg']:.2f} deg | "
            f"max|roll|={start['stage3_max_abs_roll_deg']:.2f} deg"
        )
        print(f"endpoint hover hold: {start['stage3_hover_hold_s']:.2f} s")
        print(f"same FDM           : {start['same_fdm']} (id={start['active_fdm_id']})")
        print("STAGE 3 STOP + HOVER:", "PASS" if ok else "FAIL")
        print("=" * 120)
        return 0 if ok else 1
    finally:
        close_handoff(start)


if __name__ == "__main__":
    sys.exit(main())
