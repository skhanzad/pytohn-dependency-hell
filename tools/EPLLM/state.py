from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class ConstraintEntry(TypedDict):
    module_a: str
    version_a: str
    module_b: str
    version_b: str
    error_type: str        # e.g. "ImportError", "VersionNotFound"
    error_snippet: str     # truncated error message
    iteration: int         # when this was discovered

class AttemptRecord(TypedDict):
    iteration: int
    modules: dict[str, str]       # module -> version snapshot
    python_version: str
    build_passed: bool
    run_passed: bool
    error_type: str | None
    error_message: str | None
    action_taken: str             # "version_swap", "module_add", "replan", etc.

class AgentState(TypedDict):
    # ── Input ──
    python_file: str                          # raw source code of snippet
    file_path: str                            # path to snippet on disk

    # ── Import Analysis (from deps_scraper.py) ──
    raw_imports: list[str]                    # regex-extracted imports
    llm_imports: list[str]                    # LLM-inferred imports
    merged_imports: list[str]                 # union, cleaned via module_link.json

    # ── Planning ──
    python_version: str                       # LLM-inferred Python version
    python_version_candidates: list[str]      # from get_python_range()
    resolution_plan: list[dict]               # ordered module resolution steps
    plan_revision_count: int                  # how many times we've replanned

    # ── Resolution State ──
    resolved_modules: dict[str, str]          # module -> pinned version
    current_module: str | None                # module being resolved now
    pypi_versions: dict[str, list[str]]       # module -> available versions from PyPI
    standard_lib_modules: list[str]           # modules identified as stdlib (skip)

    # ── Constraint Memory (NOVEL: persists across iterations) ──
    constraint_table: list[ConstraintEntry]   # accumulated version constraints
    attempt_history: list[AttemptRecord]      # full log of every build/run
    failed_combos: set[tuple[str, str]]       # (module, version) known to fail
    error_module_versions: dict[str, list[str]]  # module -> [bad versions]

    # ── Build/Run State ──
    dockerfile_content: str                   # generated Dockerfile text
    docker_image_name: str                    # e.g. "test/pllm:snippet_3.7"
    build_log: str                            # raw Docker build output
    run_log: str                              # raw Docker run output
    build_passed: bool
    run_passed: bool

    # ── Error Diagnosis ──
    error_type: str | None                    # classified error category
    error_module: str | None                  # module identified from error
    error_version: str | None                 # version identified from error

    # ── Reflection (NOVEL: structured reasoning) ──
    reflection_notes: list[str]               # LLM's reasoning about failures
    root_cause_hypothesis: str | None         # e.g. "numpy version too new for tf"
    suggested_strategy: str                   # "swap_version" | "add_module" | "replan"

    # ── Control Flow ──
    iteration: int
    max_iterations: int                       # default 10 (from --loop)
    status: str                               # "init"|"planning"|"resolving"|"building"|"running"|"reflecting"|"success"|"failed"
    active_agent: str                         # which agent is currently executing