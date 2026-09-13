from pathlib import Path

SOURCE = Path("validate_full_mission_v19_360_transition_trim_0590.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old_gate = 'if abs(requested_turn) >= 300.0:\n'
new_gate = 'if requested_turn > 0.0:\n'
if old_gate not in text:
    raise RuntimeError("V20: V19 transition gate not found")
text = text.replace(old_gate, new_gate, 1)

text = text.replace(
    'V19 DIRECT - 360 TRANSITION TRIM 0.590 -> SMOOTH STAGE2 HANDOFF',
    'V20 DIRECT - POSITIVE-TURN TRANSITION TRIM 0.590 -> SMOOTH STAGE2 HANDOFF',
)
text = text.replace(
    '360 transition collective=0.590; max settle=20.0s.',
    'Positive-turn transition collective=0.590; max settle=20.0s; -50 branch unchanged.',
)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
