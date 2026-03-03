# Standalone core for EPLLM (no dependency on tools/pllm).
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
REF_FILES_DIR = CORE_DIR / "ref_files"

from .ollama_helper_base import OllamaHelperBase
from .ollama_helper_tester import OllamaHelper
from .py_pi_query import PyPIQuery
from .deps_scraper import DepsScraper
from .build_dockerfile import DockerHelper
from .executor import Executor

__all__ = [
    "OllamaHelperBase",
    "OllamaHelper",
    "PyPIQuery",
    "DepsScraper",
    "DockerHelper",
    "Executor",
    "CORE_DIR",
    "REF_FILES_DIR",
]
