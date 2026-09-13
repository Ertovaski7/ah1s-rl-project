from pathlib import Path

SOURCE = Path("train_turn_full_entry_v5_targeted_patch.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old = '''        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))'''
new = '''        # Keep scale OUT of state_dict for compatibility with the V4 adapter checkpoint.
        # V4 saved only net.* parameters and applied the scale as a runtime constant.
        self.scale = torch.as_tensor(scale, dtype=torch.float32)'''

if old not in text:
    raise RuntimeError("Could not locate ResidualAdapter scale registration in V5 source")

text = text.replace(old, new, 1)

ns = {"__name__": "__main__", "__file__": str(SOURCE)}
exec(compile(text, str(SOURCE), "exec"), ns, ns)
