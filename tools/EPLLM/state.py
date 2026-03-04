from typing import TypedDict, Dict, List, Optional


class SnippetState(TypedDict, total=False):
    file_path: str
    snippet_name: str
    python_version: str
    imports: List[str]
    resolved_modules: Dict[str, str]       # module_name -> version
    failed_versions: Dict[str, List[str]]   # module_name -> list of failed versions
    build_passed: bool
    run_passed: bool
    build_log: str
    run_log: str
    error_type: Optional[str]
    error_module: Optional[str]
    iteration: int
    max_iterations: int
    status: str  # 'pending', 'building', 'running', 'success', 'failed'
