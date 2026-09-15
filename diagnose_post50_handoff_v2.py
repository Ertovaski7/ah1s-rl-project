from __future__ import annotations

"""
POST +50 HANDOFF DIAGNOSTIC V2
==============================

V1 showed a clear command discontinuity immediately after a successful +50
specialist turn:
    collective 0.5374 -> 0.5740
    elevator   -0.1316 -> -0.1450
    aileron    +0.3949 -> +0.1910
    rudder     +0.4401 -> +0.3900
while lateral velocity was still ~+31 ft/s.

V2 tests whether the post-turn failure is primarily a bumpless-transfer issue.
Two fresh runs are compared:

1) HOLD_LAST
   Keep the exact physical actuator commands present at turn success while the
   new heading reference / strong AFCS remain active.

2) SLEW_TO_TRIM
   Start from those exact physical commands and slew gradually toward the same
   recovery trim used by V10 instead of stepping there in one control tick.

Each mode rebuilds a fresh seed-42 entry, executes the verified +50 primitive,
then runs 12 s on the SAME FDM.  No reset inside a mode, no model changes,
runtime teacher OFF.
"""

import math
import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEED = 42
TARGET = 50.0
MAX_S = 12.0
LOG_DT_S = 0.5

TRIM_C = 0.5740
TRIM_E = -0.1450
TRIM_A = 0.19095
TRIM_R = 0.3900

# Per-control-step physical slew limits (CONTROL_DT ~= 0.075 s).
SLEW_C = 0.00075
SLEW_E = 0.00060
SLEW_A = 0.00300
SLEW_R = 0.00150


def fdm_float(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def snapshot(env, fdm, t):
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
    }


def print_snapshot(mode, q):
    print(
        f"{mode:12s} t={q['t']:5.2f}s | hdg={q['hdg']:7.2f} | "
        f"alt={q['alt']:7.2f} vs={q['vs']:+6.2f} v={q['v']:6.2f} lat={q['lat']:+6.2f} | "
        f"roll={q['roll']:+6.2f} pitch={q['pitch']:+6.2f} yaw={q['yaw']:+6.2f} | "
        f"c={q['c']:.4f} e={q['e']:+.4f} a={q['a']:+.4f} r={q['r']:+.4f}"
    )


def slew(x, target, limit):
    return float(x + np.clip(float(target) - float(x), -float(limit), +float(limit)))


def safety(q):
    return bool(
        q["alt"] < 275.0
        or q["alt"] > 325.0
        or abs(q["roll"]) > 18.0
        or abs(q["pitch"]) > 15.0
        or abs(q["yaw"]) > 30.0
        or q["v"] < 1.0
    )


def run_mode(stack, mode):
    bootstrap = v16.RandomizedEntryEnv(target_turn_deg=TARGET)
    _, reset_info = bootstrap.reset(seed=SEED)
    fdm = bootstrap.fdm
    fdm_id = id(fdm)

    try:
        r = v7.run_rl_primitive(stack, fdm, TARGET, fdm_id)
        if not r["success"] or r["safe"]:
            raise RuntimeError(f"+50 primitive failed before {mode}: {r}")

        env, _ = v7.make_env_on_fdm(fdm, 1.0)
        env.turn_active = False
        heading_ref = v7.heading_deg(fdm)

        # Capture the exact actuator state left by the successful PPO turn.
        c = fdm_float(fdm, "fcs/collective-cmd-norm")
        e = fdm_float(fdm, "fcs/elevator-cmd-norm")
        a = fdm_float(fdm, "fcs/aileron-cmd-norm")
        rud = fdm_float(fdm, "fcs/rudder-cmd-norm")

        fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        dt = env.CONTROL_DT
        n = int(MAX_S / dt)
        log_every = max(1, int(round(LOG_DT_S / dt)))

        q0 = snapshot(env, fdm, 0.0)
        print("\n" + "=" * 170)
        print(f"MODE {mode} | extra_steps={int(reset_info.get('extra_entry_steps', 0))} | fdm_id={fdm_id}")
        print(
            f"+50 PASS | done={r['done']:+.2f} rem={r['rem']:+.2f} alt={r['alt']:.2f} | "
            f"start controls c/e/a/r={c:.4f}/{e:+.4f}/{a:+.4f}/{rud:+.4f}"
        )
        print_snapshot(mode, q0)

        max_alt = q0["alt"]
        max_vs = abs(q0["vs"])
        max_yaw = abs(q0["yaw"])
        stopped = False

        for i in range(n):
            if id(fdm) != fdm_id:
                raise RuntimeError("FDM identity changed")

            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

            if mode == "HOLD_LAST":
                pass
            elif mode == "SLEW_TO_TRIM":
                c = slew(c, TRIM_C, SLEW_C)
                e = slew(e, TRIM_E, SLEW_E)
                a = slew(a, TRIM_A, SLEW_A)
                rud = slew(rud, TRIM_R, SLEW_R)
            else:
                raise ValueError(mode)

            fdm["fcs/collective-cmd-norm"] = c
            fdm["fcs/elevator-cmd-norm"] = e
            fdm["fcs/aileron-cmd-norm"] = a
            fdm["fcs/rudder-cmd-norm"] = rud

            js_ok = True
            for _ in range(env.PHYSICS_STEPS):
                if not fdm.run():
                    js_ok = False
                    break

            t = (i + 1) * dt
            q = snapshot(env, fdm, t)
            max_alt = max(max_alt, q["alt"])
            max_vs = max(max_vs, abs(q["vs"]))
            max_yaw = max(max_yaw, abs(q["yaw"]))

            if i % log_every == 0 or not js_ok:
                print_snapshot(mode, q)

            if (not js_ok) or safety(q):
                print(f"{mode}: SAFETY STOP")
                print_snapshot(mode, q)
                stopped = True
                break

        qf = snapshot(env, fdm, min(MAX_S, (i + 1) * dt))
        print(
            f"SUMMARY {mode}: safety={stopped} max_alt={max_alt:.2f} "
            f"max_abs_vs={max_vs:.2f} max_abs_yaw={max_yaw:.2f} | "
            f"final alt={qf['alt']:.2f} vs={qf['vs']:+.2f} lat={qf['lat']:+.2f} "
            f"roll={qf['roll']:+.2f} hdg={qf['hdg']:.2f}"
        )
        v7.release_wrapper(env)
        return qf, stopped
    finally:
        try:
            bootstrap.close()
        except Exception:
            pass


def main():
    stack = arb.load_stack()
    print("=" * 170)
    print("POST +50 HANDOFF DIAGNOSTIC V2 — HOLD-LAST vs BUMPLESS SLEW")
    print("Fresh +50 specialist for each mode | SAME FDM inside each run | teacher OFF")
    print("=" * 170)

    for mode in ("HOLD_LAST", "SLEW_TO_TRIM"):
        run_mode(stack, mode)


if __name__ == "__main__":
    main()
