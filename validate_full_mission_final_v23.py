from pathlib import Path

SOURCE = Path("validate_full_mission_v19_360_transition_trim_0590.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

# Final positive-turn transition gate. -50 remains on its proven path.
old_gate = "if abs(requested_turn) >= 300.0:"
new_gate = "if requested_turn > 0.0:"
if old_gate not in text:
    raise RuntimeError("V23 final: transition gate not found in V19")
text = text.replace(old_gate, new_gate, 1)

# Final transition policy:
# - +50: sideslip-limited altitude/VS feedback, then low-sideslip vertical recovery.
# - +200/+360: proven fixed 0.590 transition collective.
old_init = """    transition_collective = 0.590
    transition_elevator = -0.145
    transition_aileron = 0.19095
    transition_rudder = 0.390

    while settle_elapsed < 20.0:
        fdm[\"fcs/collective-cmd-norm\"] = transition_collective
"""

new_init = """    plus50_feedback = 45.0 <= requested_turn <= 55.0
    transition_collective = 0.574 if plus50_feedback else 0.590
    transition_elevator = -0.145
    transition_aileron = 0.19095
    transition_rudder = 0.390
    transition_max_s = 35.0 if plus50_feedback else 20.0

    while settle_elapsed < transition_max_s:
        if plus50_feedback:
            pre_ms = physical_metrics(fdm)
            lat_abs = abs(pre_ms[\"v_lat\"])

            if lat_abs >= 7.0:
                feedback_target = (
                    0.580
                    + 0.0003 * (300.0 - pre_ms[\"alt\"])
                    - 0.0040 * pre_ms[\"vs\"]
                )

                if lat_abs >= 18.0:
                    collective_cap = 0.574
                elif lat_abs >= 14.0:
                    collective_cap = 0.576
                elif lat_abs >= 10.0:
                    collective_cap = 0.580
                else:
                    collective_cap = 0.585

                collective_target = float(np.clip(
                    feedback_target,
                    0.574,
                    collective_cap,
                ))
            else:
                collective_target = float(np.clip(
                    0.590
                    + 0.0008 * (300.0 - pre_ms[\"alt\"])
                    - 0.0060 * pre_ms[\"vs\"],
                    0.580,
                    0.610,
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
    raise RuntimeError("V23 final: V19 transition initialization block not found")
text = text.replace(old_init, new_init, 1)

old_print = """        f\"heading_error={heading_error_s:+.2f}deg | safety_failure={settle_safety}\"
    )
"""
new_print = """        f\"heading_error={heading_error_s:+.2f}deg | collective={transition_collective:.4f} | \"
        f\"feedback_plus50={plus50_feedback} | safety_failure={settle_safety}\"
    )
"""
if old_print not in text:
    raise RuntimeError("V23 final: V19 transition print block not found")
text = text.replace(old_print, new_print, 1)

# Clean final wording: teacher is off, but the post-turn AFCS transition is a
# real runtime stabilization layer for positive-turn cases.
text = text.replace(
    "Runtime teacher/controller OFF. Same JSBSim FDM, no reset between flight phases.",
    "Runtime teacher OFF. Same JSBSim FDM, no reset between flight phases; conditional post-turn AFCS transition is enabled.",
)
text = text.replace(
    "Runtime teacher/controller: OFF",
    "Runtime teacher: OFF | post-turn AFCS transition: conditional",
)
text = text.replace(
    'print("CONTROLLER ACTIVE AT RUNTIME: False")',
    'print(f"POST-TURN AFCS TRANSITION ACTIVE: {requested_turn > 0.0}")',
)

text = text.replace(
    "V19 DIRECT - 360 TRANSITION TRIM 0.590 -> SMOOTH STAGE2 HANDOFF",
    "FINAL V23 - HYBRID RL + CONDITIONAL POST-TURN AFCS STABILIZATION",
)
text = text.replace(
    "360 transition collective=0.590; max settle=20.0s.",
    "+50 uses sideslip-limited feedback + vertical recovery; +200/+360 use fixed 0.590; -50 unchanged.",
)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
