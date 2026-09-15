from __future__ import annotations

"""
POST +50 SAME-FDM RECOVERY DIAGNOSTIC V1
========================================

Purpose
-------
V8/V9/V10 all failed AFTER a successful +50 primitive, but for different
recovery laws.  V10 reached the 325 ft safety ceiling even while collective
was already clamped low (~0.574).  Therefore the next step is diagnosis, not
another blind gain change.

This script:
- builds one randomized full-entry state (seed 42),
- executes ONE verified +50 V22 specialist primitive on the same FDM,
- then holds the newly established heading with strong AFCS,
- applies the same low physical recovery trim used by V10,
- prints live telemetry every 0.5 s for up to 12 s,
- never resets or teleports the JSBSim state,
- never modifies model weights, runtime teacher OFF.

The log is meant to reveal whether the altitude overshoot is driven mainly by
pre-existing vertical momentum, lateral/roll transient, heading/AFCs transient,
or a control-command application issue.
"""

import math
import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEED = 42
TARGET = 50.0
LOG_DT_S = 0.5
MAX_RECOVERY_S = 12.0


def fdm_float(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


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
        "collective": fdm_float(fdm, "fcs/collective-cmd-norm"),
        "elevator": fdm_float(fdm, "fcs/elevator-cmd-norm"),
        "aileron": fdm_float(fdm, "fcs/aileron-cmd-norm"),
        "rudder": fdm_float(fdm, "fcs/rudder-cmd-norm"),
        "psi_trim": math.degrees(fdm_float(fdm, "ap/afcs/psi-trim-rad", 0.0)) % 360.0,
        "roll_afcs": fdm_float(fdm, "ap/afcs/roll-channel-active-norm"),
        "yaw_afcs": fdm_float(fdm, "ap/afcs/yaw-channel-active-norm"),
    }


def p(label, q):
    print(
        f"{label:14s} t={q['t']:5.2f}s | hdg={q['hdg']:7.2f} psiTrim={q['psi_trim']:7.2f} | "
        f"alt={q['alt']:7.2f} vs={q['vs']:+6.2f} v={q['v']:6.2f} lat={q['lat']:+6.2f} | "
        f"roll={q['roll']:+6.2f} pitch={q['pitch']:+6.2f} yaw={q['yaw']:+6.2f} | "
        f"c={q['collective']:.4f} e={q['elevator']:+.4f} a={q['aileron']:+.4f} r={q['rudder']:+.4f} | "
        f"AFCS(r/y)={q['roll_afcs']:.2f}/{q['yaw_afcs']:.2f}"
    )


def main():
    stack = arb.load_stack()
    bootstrap = v16.RandomizedEntryEnv(target_turn_deg=TARGET)
    _, reset_info = bootstrap.reset(seed=SEED)
    fdm = bootstrap.fdm
    fdm_id = id(fdm)

    print("=" * 170)
    print("POST +50 SAME-FDM RECOVERY DIAGNOSTIC V1")
    print("One verified +50 specialist -> low-trim recovery telemetry | SAME FDM | teacher OFF")
    print("=" * 170)
    print(f"entry extra_steps={int(reset_info.get('extra_entry_steps', 0))} fdm_id={fdm_id}")

    try:
        r = v7.run_rl_primitive(stack, fdm, TARGET, fdm_id)
        print(
            f"+50 primitive: PASS={r['success']} safety={r['safe']} done={r['done']:+.2f} "
            f"rem={r['rem']:+.2f} alt={r['alt']:.2f} | {r['capture']}"
        )
        if not r["success"] or r["safe"]:
            raise RuntimeError("The +50 primitive itself did not pass; aborting recovery diagnostic")

        env, _ = v7.make_env_on_fdm(fdm, 1.0)
        env.turn_active = False
        heading_ref = v7.heading_deg(fdm)

        # Intentionally simple: keep the exact new heading and a clearly low
        # physical collective.  We are observing the transient, not tuning it.
        fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        dt = env.CONTROL_DT
        log_every = max(1, int(round(LOG_DT_S / dt)))
        max_steps = int(MAX_RECOVERY_S / dt)

        p("REC START", tel(env, fdm, 0.0))

        for i in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("FDM identity changed")

            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

            fdm["fcs/collective-cmd-norm"] = 0.574
            fdm["fcs/elevator-cmd-norm"] = -0.145
            fdm["fcs/aileron-cmd-norm"] = 0.19095
            fdm["fcs/rudder-cmd-norm"] = 0.390

            js_ok = True
            for _ in range(env.PHYSICS_STEPS):
                if not fdm.run():
                    js_ok = False
                    break

            t = (i + 1) * dt
            if i % log_every == 0 or not js_ok:
                q = tel(env, fdm, t)
                p("REC", q)

            q = tel(env, fdm, t)
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
                p("REC FINAL", q)
                break
        else:
            p("REC FINAL", tel(env, fdm, MAX_RECOVERY_S))

        v7.release_wrapper(env)
    finally:
        try:
            bootstrap.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
