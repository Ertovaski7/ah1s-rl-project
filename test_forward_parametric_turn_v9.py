from pathlib import Path

# V9 reuses the validated V8 mission/turn logic and changes only the
# vertical-control authority. Keeping the turn controller identical makes
# this an isolated altitude-control experiment.
source = Path("test_forward_parametric_turn_v8.py").read_text(encoding="utf-8")

replacements = {
    'results_forward_parametric_turn_v8': 'results_forward_parametric_turn_v9',
    'parametric_turn_v8_base': 'parametric_turn_v9_base',
    'PARAMETRIC FORWARD-FLIGHT TURN V8': 'PARAMETRIC FORWARD-FLIGHT TURN V9',
    'COLLECTIVE_FF = 0.5820': 'COLLECTIVE_FF = 0.5780',
    'MAX_DOWNWARD_VS_CMD = 2.00': 'MAX_DOWNWARD_VS_CMD = 2.50',
    'UPWARD_VS_GAIN = 0.020': 'UPWARD_VS_GAIN = 0.032',
    'DOWNWARD_VS_GAIN = 0.004': 'DOWNWARD_VS_GAIN = 0.003',
    'PHYS_COLLECTIVE_MIN = 0.500': 'PHYS_COLLECTIVE_MIN = 0.460',
}

for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"V9 patch target not found: {old}")
    source = source.replace(old, new, 1)

# Execute the complete V8 mission with only the constants above changed.
exec(compile(source, "test_forward_parametric_turn_v9.py", "exec"), globals(), globals())
