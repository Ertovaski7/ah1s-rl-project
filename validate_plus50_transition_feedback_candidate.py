from pathlib import Path

SOURCE = Path("validate_plus50_v22_adaptive_transition.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old_block = '''        if plus50_adaptive:
            pre_ms = physical_metrics(fdm)
            # Stage A: first dissipate the very large +50 lateral velocity with
            # low collective. Once sideslip has dropped, raise collective using
            # altitude + vertical-speed feedback. Slew limiting avoids the
            # 0.574 -> 0.576 bifurcation observed in the fixed-collective sweep.
            if abs(pre_ms["v_lat"]) > 10.0:
                collective_target = 0.574
            else:
                collective_target = float(np.clip(
                    0.580
                    + 0.0005 * (300.0 - pre_ms["alt"])
                    - 0.0015 * pre_ms["vs"],
                    0.574,
                    0.590,
                ))
            collective_step = float(np.clip(
                collective_target - transition_collective,
                -0.0005,
                +0.0005,
            ))
            transition_collective += collective_step
'''

new_block = '''        if plus50_adaptive:
            pre_ms = physical_metrics(fdm)
            lat_abs = abs(pre_ms["v_lat"])

            # Continuous +50 transition control:
            # 1) use altitude/vertical-speed feedback as soon as descent starts,
            #    instead of waiting for a hard v_lat threshold;
            # 2) limit the maximum collective while sideslip is still large so
            #    the early translational-lift overshoot cannot recur.
            feedback_target = (
                0.580
                + 0.0003 * (300.0 - pre_ms["alt"])
                - 0.0040 * pre_ms["vs"]
            )

            if lat_abs >= 18.0:
                collective_cap = 0.574
            elif lat_abs >= 14.0:
                collective_cap = 0.576
            elif lat_abs >= 10.0:
                collective_cap = 0.580
            elif lat_abs >= 7.0:
                collective_cap = 0.585
            else:
                collective_cap = 0.590

            collective_target = float(np.clip(
                feedback_target,
                0.574,
                collective_cap,
            ))
            collective_step = float(np.clip(
                collective_target - transition_collective,
                -0.0005,
                +0.0005,
            ))
            transition_collective += collective_step
'''

if old_block not in text:
    raise RuntimeError("feedback candidate: V22 adaptive block not found")
text = text.replace(old_block, new_block, 1)

text = text.replace(
    "V22 +50 ADAPTIVE TRANSITION -> SMOOTH STAGE2 HANDOFF",
    "PLUS50 FEEDBACK CANDIDATE -> SMOOTH STAGE2 HANDOFF",
)
text = text.replace(
    "+50 adaptive transition: 0.574 sideslip-damping then altitude/VS feedback; +200/+360 remain 0.590.",
    "+50 transition: continuous altitude/VS feedback with sideslip-dependent collective cap; +200/+360 unchanged.",
)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
