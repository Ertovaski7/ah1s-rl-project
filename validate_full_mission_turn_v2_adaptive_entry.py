from pathlib import Path

SOURCE = Path("validate_full_mission_turn_v1.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

text = text.replace(
    'rule("AH-1S FULL MISSION VALIDATION - PPO RUNTIME / TEACHER OFF")',
    'rule("AH-1S FULL MISSION VALIDATION V2 - ADAPTIVE PPO TURN ENTRY / TEACHER OFF")',
    1,
)

old = '''    if s["forward_ft"] >= ENTRY_FORWARD_FT:\n        entry_state = s\n        break\n'''

new = '''    # V5 was trained from a Stage-2-generated entry around 310 ft.\n    # In the true Stage1->Stage2 same-FDM mission, 160 ft was reached near\n    # 301 ft, which is physically safe but outside the distribution that\n    # produced the standalone V5 4/4 result. Do not hand off solely by\n    # distance: keep Stage-2 PPO active until the live state is also close\n    # to the V5 training-entry vertical envelope.\n    adaptive_entry = bool(\n        s["forward_ft"] >= ENTRY_FORWARD_FT\n        and 307.0 <= s["altitude_ft"] <= 313.5\n        and abs(s["vertical_speed_fps"]) <= 1.25\n        and 8.0 <= s["forward_speed_fps"] <= 18.0\n        and abs(math.degrees(s["roll_rad"])) <= 8.0\n        and abs(math.degrees(s["pitch_rad"])) <= 8.0\n    )\n\n    if step % max(1, int(2.0 / dt2)) == 0 and s["forward_ft"] >= ENTRY_FORWARD_FT:\n        print(\n            f"ENTRY SEARCH | fwd={s['forward_ft']:7.1f}ft | "\n            f"alt={s['altitude_ft']:7.2f}ft | "\n            f"vs={s['vertical_speed_fps']:+6.2f} | "\n            f"v={s['forward_speed_fps']:6.2f} | ready={adaptive_entry}"\n        )\n\n    if adaptive_entry:\n        entry_state = s\n        break\n'''

if old not in text:
    raise RuntimeError("Could not locate Stage-2 fixed entry block in V1 validator")
text = text.replace(old, new, 1)

old2 = 'raise RuntimeError("Stage2 did not reach turn-entry distance")'
new2 = 'raise RuntimeError("Stage2 did not reach the adaptive V5-compatible turn-entry envelope")'
text = text.replace(old2, new2, 1)

# Keep the rest of the validated mission logic exactly as V1.
ns = {"__name__": "__main__", "__file__": str(SOURCE)}
exec(compile(text, str(SOURCE), "exec"), ns, ns)
