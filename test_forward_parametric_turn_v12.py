from pathlib import Path

# V12 keeps the V11 controller architecture and fixes only the terminal
# state machine. V11 physically reached the requested 50 deg turn, but never
# transitioned to COMPLETE because the instantaneous yaw-rate signal oscillated
# around the target while AFCS was intentionally reduced.
#
# V12 behavior:
#   TURN -> CAPTURE when heading error stays inside a tight band briefly.
#   CAPTURE restores normal roll/yaw AFCS and installs the final heading ref.
#   VERIFY requires 10 s of stable forward flight on the new heading.
# This avoids declaring success merely because the angle was crossed once.

source = Path("test_forward_parametric_turn_v11.py").read_text(encoding="utf-8")

# V11 itself builds its executable source from V8. Inject the V12 terminal
# changes immediately after that V8 source is loaded, before V11's own patches.
anchor = 'source = Path("test_forward_parametric_turn_v8.py").read_text(encoding="utf-8")\n'
if anchor not in source:
    raise RuntimeError("V12 could not find V11 source-load anchor")

inject = r'''

# ---------------- V12 terminal capture / verification ----------------
# Near the commanded angle, do not require the noisy instantaneous yaw-rate
# signal to remain below 0.5 deg/s while AFCS is reduced. Instead, capture the
# angle first, restore AFCS, then verify the new heading for 10 seconds.
source = source.replace(
    'TARGET_HEADING_TOL_DEG = 1.0',
    'TARGET_HEADING_TOL_DEG = 0.75',
    1,
)
source = source.replace(
    'TARGET_HOLD_SECONDS = 2.0',
    'TARGET_HOLD_SECONDS = 0.75',
    1,
)

old_target_ok = '''            target_ok = bool(\n                abs(remaining) <= TARGET_HEADING_TOL_DEG\n                and abs(yaw_rate) <= TARGET_YAW_RATE_TOL_DEG_S\n                and abs(roll) <= TARGET_MAX_ABS_ROLL_DEG\n            )\n            hold_s = hold_s + dt if target_ok else 0.0\n'''
new_target_ok = '''            # V12 terminal CAPTURE: heading + bank determine capture.\n            # Yaw-rate is deliberately excluded here because V11 proved that\n            # the angle can be held within a few tenths of a degree while the\n            # reduced-AFCS yaw-rate signal oscillates. Normal AFCS is restored\n            # immediately after capture and the new heading is then verified.\n            target_ok = bool(\n                abs(remaining) <= TARGET_HEADING_TOL_DEG\n                and abs(roll) <= 3.0\n            )\n            hold_s = hold_s + dt if target_ok else 0.0\n'''
if old_target_ok not in source:
    raise RuntimeError("V12 target_ok patch target not found in V8 source")
source = source.replace(old_target_ok, new_target_ok, 1)

old_post = '''        if complete:\n            post_s += dt\n            if post_s >= POST_TURN_FORWARD_SECONDS:\n                success = True\n                termination = "turn_and_new_heading_forward_pass"\n'''
new_post = '''        if complete:\n            # V12 VERIFY phase. After AFCS is restored by the V11 completion\n            # block, require the aircraft to keep the commanded heading while\n            # continuing safe forward flight. If it leaves this corridor, the\n            # verification timer resets rather than producing a false PASS.\n            verify_ok = bool(\n                abs(remaining) <= 2.0\n                and abs(roll) <= 5.0\n                and 285.0 <= state["altitude_ft"] <= 315.0\n                and abs(state["vertical_speed_fps"]) <= 1.5\n                and state["forward_speed_fps"] >= 8.0\n            )\n            post_s = post_s + dt if verify_ok else 0.0\n            if post_s >= POST_TURN_FORWARD_SECONDS:\n                success = True\n                termination = "turn_capture_and_new_heading_verified"\n'''
if old_post not in source:
    raise RuntimeError("V12 post-turn patch target not found in V8 source")
source = source.replace(old_post, new_post, 1)
# ---------------------------------------------------------------------
'''

source = source.replace(anchor, anchor + inject, 1)

# Give this run its own result directory / banner without changing V11 logic.
source = source.replace(
    "results_forward_parametric_turn_v11",
    "results_forward_parametric_turn_v12",
)
source = source.replace(
    "parametric_turn_v11_base",
    "parametric_turn_v12_base",
)
source = source.replace(
    "PARAMETRIC FORWARD-FLIGHT TURN V11",
    "PARAMETRIC FORWARD-FLIGHT TURN V12",
)

exec(compile(source, "test_forward_parametric_turn_v12.py", "exec"), globals(), globals())
