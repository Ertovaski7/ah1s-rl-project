from __future__ import annotations

from pathlib import Path

SOURCE = Path("validate_full_mission_v7_final.py")
TRAINING_ENTRY_FORWARD_FT = 403.424

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old_attach = '''env2.fdm = fdm
if hasattr(env2, "phase"):
'''
new_attach = '''env2.fdm = fdm
# Reproduce the exact Stage-2 PPO forward-flight AFCS configuration on the
# ACTIVE live FDM. env2.reset() ran on a disposable FDM, so these values must
# be copied explicitly after the same-FDM attach.
fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
fdm["ap/afcs/roll-channel-active-norm"] = 1.0
fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
if hasattr(env2, "phase"):
'''
if old_attach not in text:
    raise RuntimeError("Could not locate Stage2 attach block in final validator")
text = text.replace(old_attach, new_attach, 1)

old_turn_distance = '''turn_env.forward_distance = env_forward
'''
new_turn_distance = f'''# IMPORTANT: forward_distance is ENVIRONMENT BOOKKEEPING, not a JSBSim
# physical state. V7 was trained with this coordinate near
# {TRAINING_ENTRY_FORWARD_FT:.3f} ft at turn activation. The live mission's
# Stage-2 counter starts from a different origin, so using its ~160-ft value
# creates a large observation-coordinate mismatch even though the physical
# aircraft state is already correct. Calibrate only this bookkeeping origin;
# the live FDM state is untouched.
physical_stage2_forward_at_handoff = float(env_forward)
turn_env.forward_distance = {TRAINING_ENTRY_FORWARD_FT:.3f}
print(
    "TURN PROGRESS COORDINATE CALIBRATION | "
    f"physical_stage2_forward={{physical_stage2_forward_at_handoff:.3f}}ft | "
    f"policy_forward_coordinate={{turn_env.forward_distance:.3f}}ft"
)
'''
if old_turn_distance not in text:
    raise RuntimeError("Could not locate turn forward_distance attach line")
text = text.replace(old_turn_distance, new_turn_distance, 1)

old_obs_line = '''obs_turn = np.asarray(turn_env._get_obs(), dtype=np.float32)
turn_entry_metrics = physical_metrics(fdm)
'''
new_obs_line = '''obs_turn = np.asarray(turn_env._get_obs(), dtype=np.float32)
turn_entry_metrics = physical_metrics(fdm)
print(
    "TURN ENTRY OBS CHECK | "
    f"obs10_forward_progress={float(obs_turn[10]):+.6f} | "
    f"training_reference={TRAINING_ENTRY_FORWARD_FT / 300.0:+.6f}"
)
'''
if old_obs_line not in text:
    raise RuntimeError("Could not locate turn observation creation block")
text = text.replace(old_obs_line, new_obs_line, 1)

# Keep Python's future import at the legal first executable position.
future_line = "from __future__ import annotations\n"
if not text.startswith(future_line):
    raise RuntimeError("Expected future import at start of V7 source")
text = text.replace(
    future_line,
    future_line + f"TRAINING_ENTRY_FORWARD_FT = {TRAINING_ENTRY_FORWARD_FT!r}\n",
    1,
)

print("=" * 120)
print("FINAL FULL-MISSION V9 - LIVE PHYSICS + CALIBRATED TURN PROGRESS COORDINATE")
print("No model weights are changed. No physical JSBSim state is modified by the calibration.")
print("Stage2 active-FDM AFCS settings are restored explicitly after same-FDM attach.")
print("Turn forward_progress bookkeeping is expressed in the coordinate used during V7 training.")
print("=" * 120)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
