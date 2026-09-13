from pathlib import Path

SOURCE = Path("validate_full_mission_v19_360_transition_trim_0590.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

# Enable the transition block for positive turns, while keeping the proven
# -50 branch untouched.
old_gate = "if abs(requested_turn) >= 300.0:"
new_gate = "if requested_turn > 0.0:"
if old_gate not in text:
    raise RuntimeError("V22: transition gate not found")
text = text.replace(old_gate, new_gate, 1)

# +50 uses an adaptive two-stage collective schedule. +200/+360 remain at the
# proven fixed 0.590 collective.
old_init = """    transition_collective = 0.590
    transition_elevator = -0.145
    transition_aileron = 0.19095
    transition_rudder = 0.390

    while settle_elapsed < 20.0:
        fdm[\"fcs/collective-cmd-norm\"] = transition_collective
"""
new_init = """    plus50_adaptive = 45.0 <= requested_turn <= 55.0
    transition_collective = 0.574 if plus50_adaptive else 0.590
    transition_elevator = -0.145
    transition_aileron = 0.19095
    transition_rudder = 0.390
    transition_max_s = 30.0 if plus50_adaptive else 20.0

    while settle_elapsed < transition_max_s:
        if plus50_adaptive:
            pre_ms = physical_metrics(fdm)
            # Stage A: first dissipate the very large +50 lateral velocity with
            # low collective. Once sideslip has dropped, raise collective using
            # altitude + vertical-speed feedback. Slew limiting avoids the
            # 0.574 -> 0.576 bifurcation observed in the fixed-collective sweep.
            if abs(pre_ms[\"v_lat\"]) > 10.0:
                collective_target = 0.574
            else:
                collective_target = float(np.clip(
                    0.580
                    + 0.0005 * (300.0 - pre_ms[\"alt\"])
                    - 0.0015 * pre_ms[\"vs\"],
                    0.574,
                    0.590,
                ))
            collective_step = float(np.clip(
                collective_target - transition_collective,
                -0.0005,
                +0.0005,
            ))
            transition_collective += collective_step

        fdm[\"fcs/collective-cmd-norm\"] = transition_collective
"""
if old_init not in text:
    raise RuntimeError("V22: transition initialization block not found")
text = text.replace(old_init, new_init, 1)

old_print = """        f\"heading_error={heading_error_s:+.2f}deg | safety_failure={settle_safety}\"
    )
"""
new_print = """        f\"heading_error={heading_error_s:+.2f}deg | collective={transition_collective:.4f} | \"
        f\"adaptive_plus50={plus50_adaptive} | safety_failure={settle_safety}\"
    )
"""
if old_print not in text:
    raise RuntimeError("V22: transition print block not found")
text = text.replace(old_print, new_print, 1)

text = text.replace(
    "V19 DIRECT - 360 TRANSITION TRIM 0.590 -> SMOOTH STAGE2 HANDOFF",
    "V22 +50 ADAPTIVE TRANSITION -> SMOOTH STAGE2 HANDOFF",
)
text = text.replace(
    "360 transition collective=0.590; max settle=20.0s.",
    "+50 adaptive transition: 0.574 sideslip-damping then altitude/VS feedback; +200/+360 remain 0.590.",
)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
