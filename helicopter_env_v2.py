"""Compatibility shim for active Stage-1 code.

The historical implementation lives in deneme/helicopter_env_v2.py after
repository cleanup. Active Stage-1 code still imports `helicopter_env_v2`
from the repository root, so execute the preserved implementation here.
"""
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent / "deneme" / "helicopter_env_v2.py"
if not _SOURCE.exists():
    raise FileNotFoundError(f"Missing preserved environment source: {_SOURCE}")

exec(compile(_SOURCE.read_text(encoding="utf-8"), str(_SOURCE), "exec"), globals(), globals())
