from __future__ import annotations

"""
ARBITRARY RELATIVE TURN V5 — DIRECT-FDM TERMINAL CAPTURE
========================================================

Why V4 still failed for +20
---------------------------
V4 asked for proven post-turn collective values below 0.590 (for example
0.574), but after capture it routed those commands through the Stage-2 action
mapping. Stage-2 clamps physical collective to [0.590, 0.650], so the intended
low collective was never actually applied. That is why +20 still climbed
through the 325 ft turn-safety ceiling.

V5 keeps the successful V3 absolute-heading capture, but during the short
terminal capture only it bypasses the Stage-2 normalized collective mapping and
writes the bounded physical terminal commands directly into the SAME live
JSBSim FDM. No reset, no teleport, no retraining, runtime teacher OFF.

The turn itself is still V22 PPO. Direct-FDM trim is only the post-turn capture /
stabilization layer, analogous to the already-proven full-mission transition.
"""

import argparse
import math
import types

import numpy as np

import test_turn_arbitrary_angles_v3_heading_capture as v3
import test_turn_arbitrary_angles_v1 as arb
import test_turn_full_entry_v16_randomized_entry_robustness as v16
import test_turn_full_entry_v22_v21_runtime as v22

DEFAULT_TARGETS = [20.0, -30.0]
DEFAULT_SEEDS = [42]
ROBUST_SEEDS = [7, 21, 42, 84, 123]

TARGET_ALT_FT = 300.0
TRANSITION_ELEVATOR = -0.145


def adaptive_collective_target(env) -> float:
    s = env._raw_state()
    alt = float(s["altitude"])
    vs = float(s["vertical_speed"])
    lat_abs = abs(float(s["lateral_velocity"]))

    # Same proven shape as the earlier successful post-turn transition.
    if lat_abs >= 7.0:
        feedback_target = 0.580 + 0.0003 * (TARGET_ALT_FT - alt) - 0.0040 * vs

        if lat_abs >= 18.0:
            collective_cap = 0.574
        elif lat_abs >= 14.0:
            collective_cap = 0.576
        elif lat_abs >= 10.0:
            collective_cap = 0.580
        else:
            collective_cap = 0.585

        return float(np.clip(feedback_target, 0.570, collective_cap))

    # When lateral motion is modest, use altitude + VS feedback.  Unlike the
    # Stage-2 action map, this physical command is allowed below 0.590.
    return float(np.clip(
        0.586 + 0.0010 * (TARGET_ALT_FT - alt) - 0.0080 * vs,
        0.570,
        0.605,
    ))


def direct_capture_apply(self, action):
    """Interpret action as PHYSICAL [collective,elevator,aileron,rudder]."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 4:
        raise ValueError("direct capture requires four physical controls")

    collective = float(np.clip(a[0], 0.560, 0.610))
    elevator = float(np.clip(a[1], -0.170, -0.120))
    aileron = float(np.clip(a[2], -1.0, 1.0))
    rudder = float(np.clip(a[3], -1.0, 1.0))

    # Keep the NEW heading reference and strong terminal stabilization active.
    self.fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    self.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    self.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

    self.fdm["fcs/collective-cmd-norm"] = collective
    self.fdm["fcs/elevator-cmd-norm"] = elevator
    self.fdm["fcs/aileron-cmd-norm"] = aileron
    self.fdm["fcs/rudder-cmd-norm"] = rudder

    return collective, elevator, aileron, rudder


def terminal_physical_action(env) -> np.ndarray:
    target = adaptive_collective_target(env)

    if not hasattr(env, "_v5_collective"):
        # Start with the proven low post-turn trim.  This value can now really
        # reach JSBSim because V5 bypasses the Stage-2 0.590 clamp.
        env._v5_collective = 0.574
        env._v5_collective_min = 1e9
        env._v5_collective_max = -1e9

    # Smooth physical collective to avoid a step at the handoff.
    step = float(np.clip(target - float(env._v5_collective), -0.00075, +0.00075))
    env._v5_collective = float(np.clip(float(env._v5_collective) + step, 0.560, 0.610))
    env._v5_collective_min = min(float(env._v5_collective_min), env._v5_collective)
    env._v5_collective_max = max(float(env._v5_collective_max), env._v5_collective)

    return np.asarray([
        env._v5_collective,
        TRANSITION_ELEVATOR,
        0.19095,
        0.390,
    ], dtype=np.float32)


def in_existing_specialist_gate(target: float) -> bool:
    tn = float(target) / 360.0
    return (
        abs(tn - v22.T50) <= v16.GATE_TOL
        or abs(tn - v22.T200) <= v16.GATE_TOL
    )


def run_one(stack, target: float, seed: int):
    bm, ba, p200, p50, p21 = stack
    env = v16.RandomizedEntryEnv(target_turn_deg=float(target))
    obs, reset_info = env.reset(seed=int(seed))
    info = reset_info

    specialist = in_existing_specialist_gate(target)
    capture = False
    capture_at = None
    capture_remaining = None
    capture_yaw = None
    capture_lead = None
    capture_heading = None
    target_heading = None
    success = False

    max_steps = int(max(100.0, abs(target) + 80.0) / env.CONTROL_DT)

    try:
        for _ in range(max_steps):
            remaining = float(target) - float(env.cumulative_turn_deg)
            yr = v3.yaw_rate_deg(env)

            if not specialist and not capture:
                lead = v3.capture_threshold(yr)
                moving_toward = (remaining * yr) > 0.0
                progress_needed = min(5.0, 0.25 * abs(float(target)))
                meaningful_progress = abs(float(env.cumulative_turn_deg)) >= progress_needed

                if moving_toward and meaningful_progress and abs(remaining) <= lead:
                    capture = True
                    capture_at = float(env.turn_steps * env.CONTROL_DT)
                    capture_remaining = float(remaining)
                    capture_yaw = float(yr)
                    capture_lead = float(lead)
                    capture_heading, target_heading = v3.enter_heading_capture(env, remaining)

                    # Structural V5 fix: terminal actions are physical controls,
                    # not Stage-2 normalized actions.
                    env._apply_action = types.MethodType(direct_capture_apply, env)

            if capture:
                env.fdm["ap/afcs/psi-trim-rad"] = math.radians(float(target_heading))
                env.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
                env.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
                action = terminal_physical_action(env)
            else:
                action = v22.act(bm, ba, p200, p50, p21, obs)

            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)

            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    return {
        "seed": int(seed),
        "target": float(target),
        "extra": int(reset_info.get("extra_entry_steps", 0)),
        "success": bool(success),
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", np.nan)),
        "vs": float(info.get("vertical_speed", np.nan)),
        "v": float(info.get("forward_velocity", np.nan)),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safe": bool(info.get("safety_failure", False)),
        "capture": bool(capture),
        "capture_at": capture_at,
        "capture_remaining": capture_remaining,
        "capture_yaw": capture_yaw,
        "capture_lead": capture_lead,
        "capture_heading": capture_heading,
        "target_heading": target_heading,
        "specialist": specialist,
        "cmin": float(getattr(env, "_v5_collective_min", np.nan)),
        "cmax": float(getattr(env, "_v5_collective_max", np.nan)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--robust", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    if args.robust and args.seeds is not None:
        raise ValueError("Use either --robust or --seeds, not both")
    seeds = ROBUST_SEEDS if args.robust else (args.seeds or DEFAULT_SEEDS)

    if any(abs(t) < 1.0 or abs(t) > 360.0 for t in targets):
        raise ValueError("Targets must satisfy 1 <= |target| <= 360 deg")

    stack = arb.load_stack()
    rows = []

    print("=" * 156)
    print("ARBITRARY TURN V5 — DIRECT-FDM TERMINAL CAPTURE")
    print("V22 PPO -> new-heading AFCS capture -> direct physical terminal trim; SAME FDM; teacher OFF")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 156)

    for target in targets:
        for seed in seeds:
            r = run_one(stack, float(target), int(seed))
            rows.append(r)

            if r["specialist"]:
                cap = "specialist-V22"
            elif r["capture"]:
                cap = (
                    f"cap@{r['capture_at']:.1f}s rem={r['capture_remaining']:+.2f} "
                    f"yaw={r['capture_yaw']:+.2f} lead={r['capture_lead']:.2f} "
                    f"hdg={r['capture_heading']:.1f}->{r['target_heading']:.1f} "
                    f"collective=[{r['cmin']:.4f},{r['cmax']:.4f}]"
                )
            else:
                cap = "NO_CAPTURE"

            print(
                f"seed={r['seed']:3d} target={r['target']:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} done={r['done']:+8.2f} rem={r['rem']:+7.2f} "
                f"alt={r['alt']:7.2f} vs={r['vs']:+5.2f} v={r['v']:5.2f} "
                f"hold={r['hold']:3.1f} safety={r['safe']} | {cap}"
            )

    print("\n" + "=" * 156)
    total_pass = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safe"]) for r in rows)
    print(f"TOTAL PASS={total_pass}/{len(rows)} | SAFETY_FAILURES={safety}/{len(rows)}")
    for target in targets:
        q = [r for r in rows if r["target"] == float(target)]
        print(
            f"target={target:+7.1f}: pass={sum(int(r['success']) for r in q)}/{len(q)} "
            f"safety_fail={sum(int(r['safe']) for r in q)}/{len(q)} "
            f"max_abs_rem={max(abs(r['rem']) for r in q):.2f}"
        )
    print("ARBITRARY CAPTURE V5:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 156)


if __name__ == "__main__":
    main()
