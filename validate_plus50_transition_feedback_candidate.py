from pathlib import Path

SOURCE = Path("validate_full_mission_v19_360_transition_trim_0590.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

# Enable transition for positive turns only; -50 remains untouched.
old_gate = "if abs(requested_turn) >= 300.0:"
new_gate = "if requested_turn > 0.0:"
if old_gate not in text:
    raise RuntimeError("feedback candidate: transition gate not found in V19")
text = text.replace(old_gate, new_gate, 1)

# Patch V19 transition block directly. +50 uses continuous altitude/VS feedback
# with a sideslip-dependent collective cap. +200/+360 retain fixed 0.590.
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
                # Low-sideslip vertical-recovery phase. At this point the large
                # translational-lift transient is gone, so use extra collective
                # authority to arrest descent and return toward 300 ft before
                # handing the FDM back to Stage2 PPO.
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
    raise RuntimeError("feedback candidate: V19 transition initialization block not found")
text = text.replace(old_init, new_init, 1)

old_print = """        f\"heading_error={heading_error_s:+.2f}deg | safety_failure={settle_safety}\"
    )
"""
new_print = """        f\"heading_error={heading_error_s:+.2f}deg | collective={transition_collective:.4f} | \"
        f\"feedback_plus50={plus50_feedback} | safety_failure={settle_safety}\"
    )
"""
if old_print not in text:
    raise RuntimeError("feedback candidate: V19 transition print block not found")
text = text.replace(old_print, new_print, 1)

text = text.replace(
    "V19 DIRECT - 360 TRANSITION TRIM 0.590 -> SMOOTH STAGE2 HANDOFF",
    "PLUS50 FEEDBACK CANDIDATE -> SMOOTH STAGE2 HANDOFF",
)
text = text.replace(
    "360 transition collective=0.590; max settle=20.0s.",
    "+50 transition: sideslip-limited feedback plus low-sideslip vertical recovery; +200/+360 remain 0.590.",
)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
