"""Shared locations for generated StegoLoRA artifacts.

Defaults are anchored to this source directory so running a script from a
different working directory does not scatter artifacts across the repository.
Set STEGOLORA_OUTPUT_DIR to relocate the complete generated-output tree.
"""
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_ENV = "STEGOLORA_OUTPUT_DIR"


def output_root() -> Path:
    configured = os.environ.get(OUTPUT_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else PROJECT_DIR / "outputs"


def output_path(*parts: str) -> str:
    return str(output_root().joinpath(*parts))


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
