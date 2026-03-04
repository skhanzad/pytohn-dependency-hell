# Standalone core for EPLLM (no dependency on tools/pllm).
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
REF_FILES_DIR = CORE_DIR / "ref_files"


__all__ = [
    "CORE_DIR",
    "REF_FILES_DIR",
]
