"""EPLLM: Enhanced Python LLM - Hybrid dependency resolver.

Combines PyEGo's static analysis (AST parsing, syntax-based version detection)
with PLLM's iterative Docker validation and error-driven refinement.
"""

__version__ = "2.0.0"

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))