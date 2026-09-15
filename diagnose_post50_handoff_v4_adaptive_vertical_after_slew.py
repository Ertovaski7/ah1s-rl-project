from __future__ import annotations

"""
POST +50 HANDOFF DIAGNOSTIC V4 — TWO-PHASE BUMPLESS RECOVERY
==============================================================

Evidence from V2/V3 diagnostics:
- Immediate jump from the +50 endpoint collective (~0.5374) toward 0.574
  caused a strong climb to the 325 ft safety ceiling.
- Holding the endpoint collective while slewing E/A/R prevented the climb,
  but after ~5 s the aircraft began an accelerating descent.

V4 therefore tests a two-phase transition on the SAME live FDM:
1) Bumpless phase: keep the exact endpoint collective while E/A/R slew toward
   recovery trims.
2) Adaptive vertical phase: once the cyclic/pedal slew is complete, keep the
   new heading with strong AFCS and adjust collective incrementally from its
   current value using LIVE altitude + vertical-speed feedback.

No reset, no teleport, no retraining, runtime teacher OFF.
This is a diagnostic/validator, not a final controller.
"""

import math
import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEED = 42
TARGET = 50.0
MAX_RECOVERY_S = 30.0
LOG_DT_S = 0.5

TARGET_ALT = 300.0
SLEW_AILERON_PER_STEP = 0.0030
SLEW_RUDDER_PER_STEP = 0.0015
SLEW_ELEVATOR_PER_STEP = 0.0006

TARGET_ELEVATOR = -0.1450
TARGET_AILERON = 0.19095
TARGET_RUDDER = 0.3900

# Outer-loop altitude -> desired vertical-speed reference.
ALT_TO_VS = 0.10
DESIRED_VS_LIMIT = 1.0

# Incremental collective controller after the bumpless cyclic transition.
# It starts from the REAL endpoint collective; there is no jump to a fixed trim.
COLLECTIVE_STEP_GAIN = 0.0015
COLLECTIVE_STEP_LIMIT = 0.00045
COLLECTIVE_MIN = 0.515
COLLECTIVE_MAX_HIGH_LAT = 0.565
COLLECTIVE_MAX_LOW_LAT = 0.595


def clip_slew(x: float, target: float, step: float) -> float:
    return float(x + np.clip(target - x, -step, +step))


def fdm_float(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def tel(env, fdm, t, phase, collective):
    s = env._raw_state()
    return {
        "t": float(t),
        "phase": str(phase),
        "hdg": float(v7.heading_deg(fdm)),
        "alt": float(s["altitude"]),
        "vs": float(s["vertical_speed"]),
        "v": float(s["forward_velocity"]),
        "lat": float(s["lateral_velocity"]),
        "roll": math.degrees(float(s["roll"])),
        "pitch": math.degrees(float(s["pitch"])),
        "yaw": math.degrees(float(s["r_rate"])),
        "collective": float(collective),
        "elevator": fdm_float(fdm, "fcs/elevator-cmd-norm"),
        "aileron": fdm_float(fdm, "fcs/aileron-cmd-norm"),
        "rudder": fdm_float(fdm, "fcs/rudder-cmd-norm"),
    }


def print_tel(q):
    print(
        f"{q['phase']:10s} t={q['t']:5.2f}s | hdg={q['hdg']:7.2f} | "
        f"alt={q['alt']:7.2f} vs={q['vs']:+6.2f} v={q['v']:6.2f} lat={q['lat']:+6.2f} | "
        f"roll={q['roll']:+6.2f} pitch={q['pitch']:+6.2f} yaw={q['yaw']:+6.2f} | "
        f"c={q['collective']:.4f} e={q['elevator']:+.4f} a={q['aileron']:+.4f} r={q['rudder']:+.4f}"
    )


def main():
    stack = arb.load_stack()
    bootstrap = v16.RandomizedEntryEnv(target_turn_deg=TARGET)
    _, reset_info = bootstrap.reset(seed=SEED)
    fdm = bootstrap.fdm
    fdm_id = id(fdm)

    print("=" * 176)
    print("POST +50 HANDOFF DIAGNOSTIC V4 — TWO-PHASE BUMPLESS + ADAPTIVE VERTICAL RECOVERY")
    print("verified +50 -> endpoint collective hold while E/A/R slew -> incremental altitude/VS collective feedback | SAME FDM | teacher OFF")
    print("=" * 176)
    print(f"entry extra_steps={int(reset_info.get('extra_entry_steps', 0))} fdm_id={fdm_id}")

    try:
        r = v7.run_rl_primitive(stack, fdm, TARGET, fdm_id)
        print(
            f"+50 primitive: PASS={r['success']} safety={r['safe']} done={r['done']:+.2f} "
            f"rem={r['rem']:+.2f} alt={r['alt']:.2f} | {r['capture']}"
        )
        if not r["success"] or r["safe"]:
            raise RuntimeError("+50 primitive did not pass")

        env, _ = v7.make_env_on_fdm(fdm, 1.0)
        env.turn_active = False
        heading_ref = v7.heading_deg(fdm)

        collective = fdm_float(fdm, "fcs/collective-cmd-norm")
        elevator = fdm_float(fdm, "fcs/elevator-cmd-norm")
        aileron = fdm_float(fdm, "fcs/aileron-cmd-norm")
        rudder = fdm_float(fdm, "fcs/rudder-cmd-norm")
        endpoint_collective = float(collective)

        print(
            f"endpoint controls c/e/a/r={collective:.4f}/{elevator:+.4f}/{aileron:+.4f}/{rudder:+.4f}"
        )

        fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        dt = env.CONTROL_DT
        max_steps = int(MAX_RECOVERY_S / dt)
        log_every = max(1, int(round(LOG_DT_S / dt)))

        adaptive_started = False
        stable_s = 0.0
        max_alt = -1e9
        min_alt = +1e9
        max_abs_vs = 0.0
        min_abs_lat = 1e9
        safety = False

        for i in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("FDM identity changed")

            # Phase A: bumpless E/A/R transition with collective frozen exactly
            # at the specialist endpoint value.
            elevator = clip_slew(elevator, TARGET_ELEVATOR, SLEW_ELEVATOR_PER_STEP)
            aileron = clip_slew(aileron, TARGET_AILERON, SLEW_AILERON_PER_STEP)
            rudder = clip_slew(rudder, TARGET_RUDDER, SLEW_RUDDER_PER_STEP)

            slew_done = bool(
                abs(elevator - TARGET_ELEVATOR) < 1e-6
                and abs(aileron - TARGET_AILERON) < 1e-6
                and abs(rudder - TARGET_RUDDER) < 1e-6
            )

            s_pre = env._raw_state()
            if slew_done:
                adaptive_started = True
                # Outer altitude loop asks only for a modest vertical-speed
                # correction.  Inner VS loop moves collective incrementally.
                desired_vs = float(np.clip(
                    ALT_TO_VS * (TARGET_ALT - float(s_pre["altitude"])),
                    -DESIRED_VS_LIMIT,
                    +DESIRED_VS_LIMIT,
                ))
                vs_error = desired_vs - float(s_pre["vertical_speed"])
                dc = float(np.clip(
                    COLLECTIVE_STEP_GAIN * vs_error,
                    -COLLECTIVE_STEP_LIMIT,
                    +COLLECTIVE_STEP_LIMIT,
                ))

                lat_abs = abs(float(s_pre["lateral_velocity"]))
                cmax = COLLECTIVE_MAX_HIGH_LAT if lat_abs >= 12.0 else COLLECTIVE_MAX_LOW_LAT
                collective = float(np.clip(collective + dc, COLLECTIVE_MIN, cmax))
            else:
                collective = endpoint_collective

            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
            fdm["fcs/collective-cmd-norm"] = collective
            fdm["fcs/elevator-cmd-norm"] = elevator
            fdm["fcs/aileron-cmd-norm"] = aileron
            fdm["fcs/rudder-cmd-norm"] = rudder

            js_ok = True
            for _ in range(env.PHYSICS_STEPS):
                if not fdm.run():
                    js_ok = False
                    break

            t = (i + 1) * dt
            s = env._raw_state()
            phase = "ADAPT" if adaptive_started else "SLEW"
            q = tel(env, fdm, t, phase, collective)

            max_alt = max(max_alt, q["alt"])
            min_alt = min(min_alt, q["alt"])
            max_abs_vs = max(max_abs_vs, abs(q["vs"]))
            min_abs_lat = min(min_abs_lat, abs(q["lat"]))

            if i % log_every == 0 or not js_ok:
                print_tel(q)

            safety = bool(
                (not js_ok)
                or q["alt"] < 275.0
                or q["alt"] > 325.0
                or abs(q["roll"]) > 18.0
                or abs(q["pitch"]) > 15.0
                or abs(q["yaw"]) > 30.0
                or q["v"] < 1.0
            )
            if safety:
                print("SAFETY STOP")
                print_tel(q)
                break

            stable = bool(
                295.0 <= q["alt"] <= 305.0
                and abs(q["vs"]) <= 0.75
                and abs(q["roll"]) <= 5.0
                and abs(q["yaw"]) <= 3.0
                and abs(q["lat"]) <= 8.0
                and q["v"] >= 8.0
            )
            stable_s = stable_s + dt if stable else 0.0
            if stable_s >= 2.0:
                print("RECOVERY PASS")
                print_tel(q)
                break

        final = tel(env, fdm, (i + 1) * dt, "FINAL", collective)
        print(
            f"SUMMARY TWO_PHASE: pass={stable_s >= 2.0} safety={safety} "
            f"max_alt={max_alt:.2f} min_alt={min_alt:.2f} max_abs_vs={max_abs_vs:.2f} "
            f"min_abs_lat={min_abs_lat:.2f} | final alt={final['alt']:.2f} vs={final['vs']:+.2f} "
            f"lat={final['lat']:+.2f} roll={final['roll']:+.2f} hdg={final['hdg']:.2f} c={collective:.4f}"
        )

        v7.release_wrapper(env)
    finally:
        try:
            bootstrap.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
