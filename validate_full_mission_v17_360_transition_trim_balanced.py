from pathlib import Path

V16 = Path("validate_full_mission_v16_360_transition_trim.py")
if not V16.exists():
    raise FileNotFoundError(V16)

text = V16.read_text(encoding="utf-8")

old_collective = 'fdm["fcs/collective-cmd-norm"] = 0.540\\n'
new_collective = 'fdm["fcs/collective-cmd-norm"] = 0.575\\n'
if old_collective not in text:
    raise RuntimeError("Could not locate V16 transition collective")
text = text.replace(old_collective, new_collective, 1)

old_time = 'while settle_elapsed < 15.0:\\n'
new_time = 'while settle_elapsed < 20.0:\\n'
if old_time not in text:
    raise RuntimeError("Could not locate V16 settle duration")
text = text.replace(old_time, new_time, 1)

text = text.replace(
    'V16 ACTIVE - 360 TRANSITION TRIM -> SMOOTH STAGE2 HANDOFF',
    'V17 ACTIVE - 360 BALANCED TRANSITION TRIM -> SMOOTH STAGE2 HANDOFF',
)
text = text.replace(
    'Same FDM; built-in AFCS transition trim; runtime teacher OFF; thresholds unchanged.',
    'Same FDM; balanced AFCS transition trim (collective=0.575); runtime teacher OFF; thresholds unchanged.',
)

exec(compile(text, str(V16), "exec"), {"__name__": "__main__", "__file__": str(V16)})
