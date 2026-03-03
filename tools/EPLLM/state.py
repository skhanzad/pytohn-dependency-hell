# Shared state for the EPLLM LangGraph pipeline.
# Mirrors the pllm executor state but structured for multi-agent flow.
from typing import TypedDict, Any, Optional


class EPLLMState(TypedDict, total=False):
    """State passed between graph nodes. All fields optional for incremental updates."""

    # Input
    file_path: str
    base_url: str
    model: str
    temp: float
    base_modules: str
    use_rag: bool

    # From analyzer
    llm_eval: dict  # python_version, python_modules (list then dict after resolve)
    scraped_imports: list

    # From version_resolver
    # llm_eval is updated in place with concrete module versions

    # Build/run
    docker_output: str
    build_complete: bool
    run_complete: bool
    last_error_type: Optional[str]
    last_fix_output: Optional[dict]

    # Error handling (mirrors pllm error_handler)
    error_handler: dict  # previous, error_modules, counts per error_type

    # Control
    iteration: int
    max_iterations: int
    python_version_index: int  # which of the version range we're trying
    python_versions: list  # [ "3.7", "3.8", ... ]

    # RAG context injected for the current step
    rag_context: str
