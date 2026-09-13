from pathlib import Path

V16 = Path("validate_full_mission_v16_360_transition_trim.py")
if not V16.exists():
    raise FileNotFoundError(V16)

text = V16.read_text(encoding="utf-8")

# V16 is already verified to execute.  Do not search for escaped code blocks
# again; only change the two scalar transition parameters.
if text.count("0.540") < 1:
    raise RuntimeError("V16 transition collective scalar not found")
text = text.replace("0.540", "0.575", 1)

if text.count("15.0") < 1:
    raise RuntimeError("V16 settle-duration scalar not found")
text = text.replace("15.0", "20.0", 1)

text = text.replace(
    "V16 ACTIVE - 360 TRANSITION TRIM -> SMOOTH STAGE2 HANDOFF",
    "V17 ACTIVE - 360 BALANCED TRANSITION TRIM -> SMOOTH STAGE2 HANDOFF",
    1,
)
text = text.replace(
    "Same FDM; built-in AFCS transition trim; runtime teacher OFF; thresholds unchanged.",
    "Same FDM; balanced AFCS transition trim (collective=0.575); runtime teacher OFF; thresholds unchanged.",
    1,
)

exec(compile(text, str(V16), "exec"), {"__name__": "__main__", "__file__": str(V16)})
