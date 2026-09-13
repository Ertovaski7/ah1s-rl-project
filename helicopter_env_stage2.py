from pathlib import Path

_source = Path(__file__).resolve().parent / "deneme" / "helicopter_env_stage2.py"
exec(compile(_source.read_text(encoding="utf-8"), str(_source), "exec"), globals(), globals())
