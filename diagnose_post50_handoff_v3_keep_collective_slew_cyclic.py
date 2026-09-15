from __future__ import annotations

"""
POST +50 HANDOFF DIAGNOSTIC V3 — KEEP ENDPOINT COLLECTIVE, SLEW CYCLICS
========================================================================

V2 showed:
- HOLD_LAST keeps altitude nearly flat but leaves very large lateral velocity.
- SLEW_TO_TRIM reduces lateral velocity somewhat, but altitude climbs to the
  325-ft safety ceiling.

The key confound in SLEW_TO_TRIM is that collective rises from the +50 endpoint
(~0.537) toward 0.574 at the same time as elevator/aileron/rudder are slewed.
This diagnostic isolates the hypothesis that the vertical runaway is caused
primarily by that collective increase.

Procedure:
1) Build the same randomized full-entry state (seed 42).
2) Execute one verified +50 V22 specialist primitive.
3) Strong AFCS holds the newly established heading.
4) KEEP collective fixed at the exact endpoint physical value.
5) Slew only elevator/aileron/rudder gradually toward the recovery trim.
6) Log live telemetry every ~0.5 s for 15 s.

Same live FDM, no reset inside the run, no model changes, runtime teacher OFF.
"""

import math
import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEED = 42
TARGET = 50.0
MAX_S = 15.0
LOG_DT_S = 0.5

TARGET_E = -0.145
TARGET_A = 0.19095
TARGET_R = 0.390

# Same approximate per-control-step slew observed in the prior V2 diagnostic.
E_STEP = 0.0006
A_STEP = 0.0030
R_STEP = 0.0015


def fdm_float(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def slew(x, target, step):
    return float(x + np.clip(float(target) - float(x), -float(step), +float(step)))


def tel(env, fdm, t):
    s = env._raw_state()
    return {
        "t": float(t),
        "hdg": float(v7.heading_deg(fdm)),
        "alt": float(s["altitude"]),
        "vs": float(s["vertical_speed"]),
        "v": float(s["forward_velocity"]),
        "lat": float(s["lateral_velocity"]),
        "roll": math.degrees(float(s["roll"])),
        "pitch": math.degrees(float(s["pitch"])),
        "yaw": math.degrees(float(s["r_rate"])),
        "c": fdm_float(fdm, "fcs/collective-cmd-norm"),
        "e": fdm_float(fdm, "fcs/elevator-cmd-norm"),
        "a": fdm_float(fdm, "fcs/aileron-cmd-norm"),
        "r": fdm_float(fdm, "fcs/rudder-cmd-norm"),
        "psi": math.degrees(fdm_float(fdm, "ap/afcs/psi-trim-rad", 0.0)) % 360.0,
    }


def print_tel(label, q):
    print(
        f"{label:16s} t={q['t']:5.2f}s | hdg={q['hdg']:7.2f} psiTrim={q['psi']:7.2f} | "
        f"alt={q['alt']:7.2f} vs={q['vs']:+6.2f} v={q['v']:6.2f} lat={q['lat']:+6.2f} | "
        f"roll={q['roll']:+6.2f} pitch={q['pitch']:+6.2f} yaw={q['yaw']:+6.2f} | "
        f"c={q['c']:.4f} e={q['e']:+.4f} a={q['a']:+.4f} r={q['r']:+.4f}"
    )


def main():
    stack = arb.load_stack()
    bootstrap = v16.RandomizedEntryEnv(target_turn_deg=TARGET)
    _, reset_info = bootstrap.reset(seed=SEED)
    fdm = bootstrap.fdm
    fdm_id = id(fdm)

    print("=" * 176)
    print("POST +50 HANDOFF DIAGNOSTIC V3 — KEEP ENDPOINT COLLECTIVE, SLEW CYCLICS")
    print("One verified +50 specialist -> fixed endpoint collective + bumpless E/A/R slew | SAME FDM | teacher OFF")
    print("=" * 176)
    print(f"entry extra_steps={int(reset_info.get('extra_entry_steps', 0))} fdm_id={fdm_id}")

    env = None
    try:
        r = v7.run_rl_primitive(stack, fdm, TARGET, fdm_id)
        print(
            f"+50 primitive: PASS={r['success']} safety={r['safe']} done={r['done']:+.2f} "
            f"rem={r['rem']:+.2f} alt={r['alt']:.2f} | {r['capture']}"
        )
        if not r["success"] or r["safe"]:
            raise RuntimeError("+50 primitive failed; aborting diagnostic")

        env, _ = v7.make_env_on_fdm(fdm, 1.0)
        env.turn_active = False
        heading_ref = v7.heading_deg(fdm)

        c_hold = fdm_float(fdm, "fcs/collective-cmd-norm")
        e = fdm_float(fdm, "fcs/elevator-cmd-norm")
        a = fdm_float(fdm, "fcs/aileron-cmd-norm")
        rr = fdm_float(fdm, "fcs/rudder-cmd-norm")

        print(
            f"endpoint controls: c/e/a/r={c_hold:.4f}/{e:+.4f}/{a:+.4f}/{rr:+.4f} | "
            f"holding collective EXACTLY at endpoint value"
        )

        fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        dt = env.CONTROL_DT
        max_steps = int(MAX_S / dt)
        log_every = max(1, int(round(LOG_DT_S / dt)))

        max_alt = -1e9
        min_alt = 1e9
        max_abs_vs = 0.0
        max_abs_yaw = 0.0
        min_abs_lat = 1e9
        safety = False

        q = tel(env, fdm, 0.0)
        print_tel("KEEP_C+SLEW", q)

        for i in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("FDM identity changed")

            e = slew(e, TARGET_E, E_STEP)
            a = slew(a, TARGET_A, A_STEP)
            rr = slew(rr, TARGET_R, R_STEP)

            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

            fdm["fcs/collective-cmd-norm"] = c_hold
            fdm["fcs/elevator-cmd-norm"] = e
            fdm["fcs/aileron-cmd-norm"] = a
            fdm["fcs/rudder-cmd-norm"] = rr

            js_ok = True
            for _ in range(env.PHYSICS_STEPS):
                if not fdm.run():
                    js_ok = False
                    break

            t = (i + 1) * dt
            q = tel(env, fdm, t)
            max_alt = max(max_alt, q["alt"])
            min_alt = min(min_alt, q["alt"])
            max_abs_vs = max(max_abs_vs, abs(q["vs"]))
            max_abs_yaw = max(max_abs_yaw, abs(q["yaw"]))
            min_abs_lat = min(min_abs_lat, abs(q["lat"]))

            if i % log_every == 0 or not js_ok:
                print_tel("KEEP_C+SLEW", q)

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
                break

        print_tel("FINAL", q)
        print(
            "SUMMARY KEEP_C_SLEW: "
            f"safety={safety} max_alt={max_alt:.2f} min_alt={min_alt:.2f} "
            f"max_abs_vs={max_abs_vs:.2f} max_abs_yaw={max_abs_yaw:.2f} "
            f"min_abs_lat={min_abs_lat:.2f} | final alt={q['alt']:.2f} vs={q['vs']:+.2f} "
            f"lat={q['lat']:+.2f} roll={q['roll']:+.2f} hdg={q['hdg']:.2f}"
        )

    finally:
        if env is not None:
            v7.release_wrapper(env)
        try:
            bootstrap.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
