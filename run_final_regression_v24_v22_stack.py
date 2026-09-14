from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path('.')
V11 = ROOT / 'validate_full_mission_v11_final_gated.py'
V19 = ROOT / 'validate_full_mission_v19_360_transition_trim_0590.py'
V23 = ROOT / 'validate_full_mission_final_v23.py'
TMP_BASE = ROOT / '_v24_base_v22_turn_stack.py'
TMP_TRANS = ROOT / '_v24_transition.py'
TMP_FINAL = ROOT / '_v24_final_validator.py'
ANGLES = [-50, 50, 200, 360]
LOG_DIR = ROOT / 'final_regression_logs_v24'

REQ = [
    Path('models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip'),
    Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt'),
    Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt'),
    Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt'),
    Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V21_STRONG_TERMINAL_50_PATCH.pt'),
]


def must_replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f'V24 transform failed: {label}')
    return text.replace(old, new, 1)


def build_base() -> None:
    text = V11.read_text(encoding='utf-8')

    # Retire the historical -50 V10 collective patch. The V22 robustness suite
    # passed -50 without a target-specific patch, so keep -50 on V5+V4 only.
    text = must_replace(
        text,
        'V10_PATCH_PATH = "models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt"\n',
        'V17_PATCH_PATH = "models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt"\n'
        'V21_PATCH_PATH = "models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V21_STRONG_TERMINAL_50_PATCH.pt"\n'
        'TARGET_50_NORM = 50.0 / 360.0\n'
        'TERM_50_DEG = 6.0\n'
        'V21_AIL_SCALE = 0.15\n'
        'V21_RUD_SCALE = 0.25\n',
        'replace V10 constants',
    )

    old_cls_start = 'class NegativeCollectivePatch(nn.Module):\n'
    old_cls_end = '\n\ndef rule(text: str):\n'
    s = text.find(old_cls_start)
    e = text.find(old_cls_end, s)
    if s < 0 or e < 0:
        raise RuntimeError('V24 transform failed: V10 class block')
    new_cls = '''class Terminal50Patch(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(16, 32), nn.Tanh(),\n            nn.Linear(32, 32), nn.Tanh(),\n            nn.Linear(32, 2), nn.Tanh(),\n        )\n\n    def forward(self, obs):\n        y = self.net(obs)\n        scale = torch.tensor([V21_AIL_SCALE, V21_RUD_SCALE], dtype=torch.float32, device=obs.device)\n        return y * scale\n\n\ndef is_target50(obs):\n    return abs(float(obs[12]) - TARGET_50_NORM) <= GATE_TOL\n\n\ndef terminal50_gate(obs):\n    return is_target50(obs) and abs(float(obs[13]) * 360.0) <= TERM_50_DEG\n'''
    text = text[:s] + new_cls + text[e:]

    old_req = '''V10_NEG_PATCH_PATH = Path(V10_PATCH_PATH)\nfor path in [TURN_PPO_PATH, V4_ADAPTER_PATH, V7_PATCH_PATH, V10_NEG_PATCH_PATH]:\n    require_file(path)\n'''
    new_req = '''V17_50_PATCH_PATH = Path(V17_PATCH_PATH)\nV21_50_PATCH_PATH = Path(V21_PATCH_PATH)\nfor path in [TURN_PPO_PATH, V4_ADAPTER_PATH, V7_PATCH_PATH, V17_50_PATCH_PATH, V21_50_PATCH_PATH]:\n    require_file(path)\n'''
    text = must_replace(text, old_req, new_req, 'required models')

    old_load = '''patch = ResidualAdapter(PATCH_SCALE)\npatch.load_state_dict(torch.load(V7_PATCH_PATH, map_location="cpu"), strict=True)\npatch.eval()\nneg_patch = NegativeCollectivePatch()\nneg_patch.load_state_dict(torch.load(V10_NEG_PATCH_PATH, map_location="cpu"), strict=True)\nneg_patch.eval()\nfor module in [base_adapter, patch, neg_patch]:\n    for p in module.parameters():\n        p.requires_grad = False\n'''
    new_load = '''patch = ResidualAdapter(PATCH_SCALE)\npatch.load_state_dict(torch.load(V7_PATCH_PATH, map_location="cpu"), strict=True)\npatch.eval()\npatch50 = ResidualAdapter(PATCH_SCALE)\npatch50.load_state_dict(torch.load(V17_50_PATCH_PATH, map_location="cpu"), strict=True)\npatch50.eval()\nterm50 = Terminal50Patch()\nterm50.load_state_dict(torch.load(V21_50_PATCH_PATH, map_location="cpu"), strict=True)\nterm50.eval()\nfor module in [base_adapter, patch, patch50, term50]:\n    for p in module.parameters():\n        p.requires_grad = False\n'''
    text = must_replace(text, old_load, new_load, 'load V17/V21')

    text = text.replace('print("V10 -50 gated collective patch:", V10_NEG_PATCH_PATH)\n',
                        'print("V17 +50 gated specialist:", V17_50_PATCH_PATH)\nprint("V21 +50 terminal specialist:", V21_50_PATCH_PATH)\n')

    old_action = '''def turn_action(turn_model, base_adapter, patch, obs):\n    """Final runtime turn action. Teacher/controller is not used here."""\n    base, _ = turn_model.predict(obs, deterministic=True)\n    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)\n    with torch.no_grad():\n        d_v4 = base_adapter(x).cpu().numpy()[0]\n        if gate_from_obs(obs) > 0.5:\n            d_v7 = patch(x).cpu().numpy()[0]\n        else:\n            d_v7 = np.zeros(4, dtype=np.float32)\n        d_v10 = float(neg_patch(x).cpu().numpy()[0]) if neg_gate_from_obs(obs) else 0.0\n    action = np.asarray(base, dtype=np.float32) + d_v4 + d_v7\n    action[0] += d_v10\n    action = np.clip(action, -1.0, 1.0)\n    return action, d_v4, d_v7\n'''
    new_action = '''def turn_action(turn_model, base_adapter, patch, obs):\n    """V22-validated runtime turn stack. Runtime teacher is OFF."""\n    base, _ = turn_model.predict(obs, deterministic=True)\n    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)\n    with torch.no_grad():\n        d_v4 = base_adapter(x).cpu().numpy()[0]\n        d_spec = np.zeros(4, dtype=np.float32)\n        if gate_from_obs(obs) > 0.5:\n            d_spec = patch(x).cpu().numpy()[0]\n        elif is_target50(obs):\n            d_spec = patch50(x).cpu().numpy()[0]\n        action = np.asarray(base, dtype=np.float32) + d_v4 + d_spec\n        if terminal50_gate(obs):\n            dt = term50(x).cpu().numpy()[0]\n            action[2] += float(dt[0])\n            action[3] += float(dt[1])\n    action = np.clip(action, -1.0, 1.0)\n    return action, d_v4, d_spec\n'''
    text = must_replace(text, old_action, new_action, 'turn_action')

    text = text.replace(
        'print(f"V10 -50 GATE AT TURN ENTRY: {\'ON\' if neg_gate_from_obs(obs_turn) else \'OFF\'}")\n',
        'print(f"V17 +50 GATE AT TURN ENTRY: {\'ON\' if is_target50(obs_turn) else \'OFF\'}")\n'
        'print(f"V21 +50 TERMINAL GATE AT TURN ENTRY: {\'ON\' if terminal50_gate(obs_turn) else \'OFF\'}")\n',
    )
    text = text.replace(
        'FINAL FULL-MISSION V11 - CALIBRATED PROGRESS + GATED -50 COLLECTIVE PATCH',
        'V24 BASE - V22 VALIDATED TURN STACK',
    )
    text = text.replace(
        'V7 correction is active only for +200; V10 collective correction is active only for -50.',
        'V7 is gated to +200; V17 and V21 are gated to +50; -50/+360 use V5+V4.',
    )

    TMP_BASE.write_text(text, encoding='utf-8')


def build_transition() -> None:
    text = V19.read_text(encoding='utf-8')
    text = must_replace(
        text,
        'SOURCE = Path("validate_full_mission_v11_final_gated.py")',
        f'SOURCE = Path("{TMP_BASE.name}")',
        'V19 source redirect',
    )
    TMP_TRANS.write_text(text, encoding='utf-8')


def build_final() -> None:
    text = V23.read_text(encoding='utf-8')
    text = must_replace(
        text,
        'SOURCE = Path("validate_full_mission_v19_360_transition_trim_0590.py")',
        f'SOURCE = Path("{TMP_TRANS.name}")',
        'V23 source redirect',
    )
    text = text.replace('FINAL V23 - HYBRID RL + CONDITIONAL POST-TURN AFCS STABILIZATION',
                        'FINAL V24 - V22 TURN STACK + CONDITIONAL POST-TURN AFCS STABILIZATION')
    TMP_FINAL.write_text(text, encoding='utf-8')


def extract_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.strip()
    return ''


def run_case(angle: int) -> dict:
    print('\n' + '=' * 120)
    print(f'FINAL REGRESSION V24 | requested_turn={angle:+d} deg')
    print('=' * 120)
    proc = subprocess.run(
        [sys.executable, str(TMP_FINAL)],
        input=f'{angle}\n',
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = proc.stdout
    print(out, end='' if out.endswith('\n') else '\n')
    LOG_DIR.mkdir(exist_ok=True)
    log = LOG_DIR / f'turn_{angle:+d}.log'
    log.write_text(out, encoding='utf-8')
    full = extract_line(out, 'FULL MISSION PASS:')
    return {
        'angle': angle,
        'passed': proc.returncode == 0 and full == 'FULL MISSION PASS: True',
        'returncode': proc.returncode,
        'log': str(log),
        'turn': extract_line(out, 'TURN CAPTURE:'),
        'transition': extract_line(out, 'POST-TURN TRANSITION TRIM'),
        'post': extract_line(out, 'POST-TURN FORWARD:'),
        'criterion': extract_line(out, 'FINAL CRITERION BREAKDOWN'),
    }


def main() -> int:
    for p in (V11, V19, V23, *REQ):
        if not p.exists():
            print('MISSING:', p)
            return 2
    build_base(); build_transition(); build_final()
    print('=' * 120)
    print('AH-1S FINAL REGRESSION SUITE — V24 / V22 TURN STACK')
    print('Turn stack: V5 + V4; +200 => V7; +50 => V17 + terminal V21; -50/+360 => no specialist.')
    print('Runtime teacher OFF. Same-FDM mission semantics preserved. V23 post-turn transition logic preserved.')
    print('=' * 120)
    results = [run_case(a) for a in ANGLES]
    print('\n' + '=' * 120)
    print('FINAL REGRESSION MATRIX — V24')
    print('=' * 120)
    for r in results:
        print(f"{r['angle']:+5d} | {'PASS' if r['passed'] else 'FAIL':4s} | return={r['returncode']} | {r['log']}")
    ok = all(r['passed'] for r in results)
    print('-' * 120)
    print('FINAL 4/4 REGRESSION PASS:', ok)
    if not ok:
        print('\nFAILED CASE DETAILS')
        for r in results:
            if r['passed']:
                continue
            print('-' * 120)
            print('ANGLE', f"{r['angle']:+d}")
            for k in ('turn','transition','post','criterion'):
                print(r[k] or f'{k}: not found')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
