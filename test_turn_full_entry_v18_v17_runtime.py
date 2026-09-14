from pathlib import Path

import test_turn_full_entry_v16_randomized_entry_robustness as v16

# Reuse the exact V16 randomized-entry evaluator, but replace only the +50 specialist.
v16.PATCH_50 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt")

print("=" * 120)
print("V18 EVALUATION - V17 +50 SPECIALIST IN THE FROZEN MULTI-TARGET RUNTIME")
print("-50 and +360: no target-specific patch")
print("+200: V7 gated specialist")
print("+50: V17 robust gated specialist")
print("Runtime teacher/controller OFF")
print("=" * 120)

v16.main()
