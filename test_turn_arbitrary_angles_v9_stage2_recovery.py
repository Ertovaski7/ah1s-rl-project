from __future__ import annotations

"""
ARBITRARY TURN V9 — STAGE2-PPO SAME-FDM RECOVERY
================================================

V8 proved that a recovery phase is needed between back-to-back turn
primitives, but its fixed cyclic/collective recovery settled around ~291 ft
instead of returning to the ~300 ft entry envelope.

V9 keeps the V7 supervisory planner and turn primitives, but changes only the
between-primitive recovery:

    successful primitive
      -> hold the newly established heading with strong AFCS
      -> run the already-validated Stage2 PPO on the SAME live FDM
      -> wait for a stable ~300 ft forward-flight entry state
      -> launch the next primitive

No FDM reset, no state teleport, no model retraining, runtime teacher OFF.
"""

import argparse
import math

import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_full_entry_v16_randomized_entry_robustness as v16

DEFAULT_TARGETS = [75.0, -90.0, 120.0, 150.0]
DEFAULT_SEEDS = [42]

RECOVERY_MAX_S = 25.0
RECOVERY_HOLD_S = 2.0
REC_ALT_MIN = 296.0
REC_ALT_MAX = 304.0
REC_MAX_ABS_VS = 0.80
REC_MAX_ABS_ROLL_DEG = 4.5
REC_MAX_ABS_YAW_DEG_S = 2.0
REC_MAX_ABS_LAT_V = 4.5
REC_MIN_FORWARD_V = 8.0
REC_MAX_FORWARD_V = 18.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args()


def recover_with_stage2(fdm, fdm_id: int):
    """Recover the current live state using the frozen Stage2 PPO, same FDM."""
    env, _ = v7.make_env_on_fdm(fdm, 1.0)
    heading_ref = v7.heading_deg(fdm)
    elapsed = 0.0
    stable_s = 0.0
    success = False
    safety = False

    try:
        # Straight-flight semantics only; do NOT reset JSBSim.
        env.turn_active = False
        env.forward_distance = 0.0
        env.previous_action = np.zeros((4,), dtype=np.float32)
        env.steps = 0

        fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        max_steps = int(RECOVERY_MAX_S / env.CONTROL_DT)

        for _ in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("Shared FDM identity changed during Stage2 recovery")

            # Preserve the new absolute heading reference while Stage2 PPO
            # controls the forward/vertical state.
            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

            base_obs = np.asarray(env._base_obs(), dtype=np.float32)
            action, _ = env.stage2_model.predict(base_obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            # turn_active=False dispatches to the exact Stage2 mapped action path.
            env._apply_action(action)

            js_ok = True
            for _ in range(env.PHYSICS_STEPS):
                if not fdm.run():
                    js_ok = False
                    break

            elapsed += env.CONTROL_DT
            s = env._raw_state()
            env.forward_distance += float(s["forward_velocity"]) * env.CONTROL_DT

            roll_deg = abs(math.degrees(float(s["roll"])))
            pitch_deg = abs(math.degrees(float(s["pitch"])))
            yaw_deg_s = abs(math.degrees(float(s["r_rate"])))

            safety = bool(
                (not js_ok)
                or float(s["altitude"]) < 275.0
                or float(s["altitude"]) > 325.0
                or roll_deg > 18.0
                or pitch_deg > 15.0
                or yaw_deg_s > 30.0
                or float(s["forward_velocity"]) < 1.0
            )
            if safety:
                break

            stable = bool(
                REC_ALT_MIN <= float(s["altitude"]) <= REC_ALT_MAX
                and abs(float(s["vertical_speed"])) <= REC_MAX_ABS_VS
                and roll_deg <= REC_MAX_ABS_ROLL_DEG
                and yaw_deg_s <= REC_MAX_ABS_YAW_DEG_S
                and abs(float(s["lateral_velocity"])) <= REC_MAX_ABS_LAT_V
                and REC_MIN_FORWARD_V <= float(s["forward_velocity"]) <= REC_MAX_FORWARD_V
            )
            stable_s = stable_s + env.CONTROL_DT if stable else 0.0
            if stable_s >= RECOVERY_HOLD_S:
                success = True
                break

        s = env._raw_state()
        return {
            "success": bool(success),
            "safe": bool(safety),
            "elapsed": float(elapsed),
            "alt": float(s["altitude"]),
            "vs": float(s["vertical_speed"]),
            "v": float(s["forward_velocity"]),
            "lat": float(s["lateral_velocity"]),
            "roll": float(math.degrees(float(s["roll"]))),
            "yaw": float(math.degrees(float(s["r_rate"]))),
            "heading": float(v7.heading_deg(fdm)),
            "collective": float(fdm["fcs/collective-cmd-norm"]),
        }
    finally:
        v7.release_wrapper(env)


def run_command(stack, target: float, seed: int):
    plan = v7.plan_command(float(target))
    bootstrap_target = float(plan[0][1]) if plan else float(target)

    bootstrap = v16.RandomizedEntryEnv(target_turn_deg=bootstrap_target)
    _, reset_info = bootstrap.reset(seed=int(seed))
    fdm = bootstrap.fdm
    fdm_id = id(fdm)
    start_hdg = v7.heading_deg(fdm)
    start_sim = float(fdm["simulation/sim-time-sec"])

    rows = []
    recoveries = []
    ok = True

    try:
        for idx, (kind, angle) in enumerate(plan, start=1):
            if kind == "rl":
                r = v7.run_rl_primitive(stack, fdm, float(angle), fdm_id)
            else:
                r = v7.fine_afcs_tail(fdm, float(angle), fdm_id)
            rows.append(r)

            print(
                f"    primitive {idx}/{len(plan)} {kind.upper():4s} {angle:+6.1f} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} "
                f"done={r['done']:+7.2f} rem={r['rem']:+6.2f} alt={r['alt']:7.2f} | {r['capture']}"
            )

            if not r["success"] or r["safe"]:
                ok = False
                break

            if idx < len(plan):
                rec = recover_with_stage2(fdm, fdm_id)
                recoveries.append(rec)
                print(
                    f"      STAGE2 RECOVERY | PASS={str(rec['success']):5s} safety={rec['safe']} "
                    f"t={rec['elapsed']:5.1f}s alt={rec['alt']:7.2f} vs={rec['vs']:+5.2f} "
                    f"v={rec['v']:5.2f} lat={rec['lat']:+5.2f} roll={rec['roll']:+5.2f} "
                    f"yaw={rec['yaw']:+5.2f} c={rec['collective']:.4f}"
                )
                if not rec["success"] or rec["safe"]:
                    ok = False
                    break

        end_hdg = v7.heading_deg(fdm)
        end_sim = float(fdm["simulation/sim-time-sec"])
        total_done = float(sum(r["done"] for r in rows))
        final_rem = float(target - total_done)

        return {
            "seed": int(seed),
            "target": float(target),
            "plan": plan,
            "success": bool(ok and abs(final_rem) <= 2.0 and id(fdm) == fdm_id),
            "safe": any(bool(r["safe"]) for r in rows) or any(bool(x["safe"]) for x in recoveries),
            "done": total_done,
            "rem": final_rem,
            "start_hdg": start_hdg,
            "end_hdg": end_hdg,
            "sim_dt": end_sim - start_sim,
            "fdm_id": fdm_id,
            "extra": int(reset_info.get("extra_entry_steps", 0)),
        }
    finally:
        try:
            bootstrap.close()
        except Exception:
            pass


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    seeds = args.seeds or DEFAULT_SEEDS

    if any(abs(t) < 1.0 or abs(t) > 360.0 for t in targets):
        raise ValueError("Targets must satisfy 1 <= |target| <= 360 deg")

    stack = arb.load_stack()
    rows = []

    print("=" * 168)
    print("ARBITRARY TURN V9 — STAGE2-PPO SAME-FDM RECOVERY")
    print("primitive -> Stage2 PPO recovery -> next primitive | SAME FDM | teacher OFF | no retraining")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 168)

    for target in targets:
        for seed in seeds:
            plan = v7.plan_command(float(target))
            print("\n" + "-" * 160)
            print(f"USER COMMAND {target:+.1f} deg | plan={plan}")
            print("-" * 160)

            r = run_command(stack, float(target), int(seed))
            rows.append(r)
            print(
                f"seed={seed:3d} target={target:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} | "
                f"done={r['done']:+8.2f} rem={r['rem']:+7.2f} | "
                f"heading={r['start_hdg']:.2f}->{r['end_hdg']:.2f} | "
                f"sim_dt={r['sim_dt']:.2f}s | fdm={r['fdm_id']}"
            )

    print("\n" + "=" * 168)
    total_pass = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safe"]) for r in rows)
    print(f"TOTAL PASS={total_pass}/{len(rows)} | SAFETY_FAILURES={safety}/{len(rows)}")
    for target in targets:
        q = [r for r in rows if r["target"] == float(target)]
        print(
            f"target={target:+7.1f} | pass={sum(int(r['success']) for r in q)}/{len(q)} | "
            f"safety_fail={sum(int(r['safe']) for r in q)}/{len(q)} | "
            f"max_abs_remaining={max(abs(r['rem']) for r in q):.2f}"
        )
    print("ARBITRARY V9 STAGE2 RECOVERY:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 168)


if __name__ == "__main__":
    main()
