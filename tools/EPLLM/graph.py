from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph.message import MessagesState
from state import AgentState
from pathlib import Path
import sys
import json
import tempfile
import os

# Add tools/pllm so pllm helpers can be imported
_pllm_dir = Path(__file__).resolve().parent.parent
if _pllm_dir.is_dir() and str(_pllm_dir) not in sys.path:
    sys.path.insert(0, str(_pllm_dir))


from pllm.helpers.deps_scraper import DepsScraper
from pllm.helpers.ollama_helper_tester import OllamaHelper
from pllm.helpers.py_pi_query import PyPIQuery
from pllm.helpers.build_dockerfile import DockerHelper

def _load_module_link() -> dict:
    """Load module_link.json for canonical PyPI name mapping."""
    path = Path(__file__).resolve().parent.parent / "pllm" / "helpers" / "ref_files" / "module_link.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _apply_module_link(names: list[str], known_modules: dict) -> list[str]:
    """Map import names to canonical refs via module_link.json; normalize to top-level, lowercase."""
    out = []
    for name in names:
        top = name.split(".")[0].strip().replace(";", "").replace(",", "")
        if not top:
            continue
        key = top.lower()
        out.append(known_modules.get(key, {}).get("ref", key))
    return out


def extract_imports(state: AgentState) -> dict:
    """
    ANALYZER AGENT — Two-pronged import extraction:
    1. Regex scan (deps_scraper.find_word_in_file) for 'import' statements
    2. LLM inference (ollama_helper.evaluate_file) for modules + Python version

    Merge both lists, clean via module_link.json name mapping, filter stdlib modules.

    Maps to: DepsScraper.find_word_in_file() + OllamaHelper.evaluate_file()
    Updates: raw_imports, llm_imports, merged_imports, python_version
    """
    deps = DepsScraper()
    file_path = state["file_path"]

    # 1. Regex-extracted imports
    raw_imports = deps.find_word_in_file(file_path, "import", [])

    # 2. LLM-inferred modules and Python version (may return dict or Pydantic model)
    ollama = OllamaHelper(model="gemma2")
    llm_out = ollama.evaluate_file(file_path)
    if llm_out:
        llm_imports = list(
            llm_out.get("python_modules", []) if isinstance(llm_out, dict) else getattr(llm_out, "python_modules", [])
        )
        _pv = llm_out.get("python_version") if isinstance(llm_out, dict) else getattr(llm_out, "python_version", None)
        python_version = _pv or "3.10"
    else:
        llm_imports = []
        python_version = state.get("python_version") or "3.10"

    # 3. Merge: union of names, apply module_link, then clean_deps (filter stdlib)
    known_modules = _load_module_link()
    combined = _apply_module_link(raw_imports, known_modules) + _apply_module_link(llm_imports, known_modules)
    merged_imports = deps.clean_deps(list(dict.fromkeys(combined)))  # dedupe order-preserving

    return {
        "raw_imports": raw_imports,
        "llm_imports": llm_imports,
        "merged_imports": merged_imports,
        "python_version": python_version,
    }


def plan_resolution(state: AgentState) -> dict:
    """
    PLANNER AGENT — Creates an ordered resolution strategy:
    1. Query PyPI for each module's available versions (py_pi_query.get_module_specifics)
    2. Filter versions by Python version date range (Algorithm 1 from paper)
    3. Rank modules by constraint complexity (most-constrained first)
    4. Set fallback strategies per module

    NOVEL: Uses constraint_table + attempt_history to avoid known-bad paths.
    On replan: LLM reasons over full failure history to change strategy.

    Maps to: PyPIQuery.get_module_specifics() + new planning logic
    Updates: resolution_plan, pypi_versions, python_version_candidates
    """
    merged = state.get("merged_imports") or []
    python_version = state.get("python_version") or "3.10"
    constraint_table = state.get("constraint_table") or []
    error_module_versions = state.get("error_module_versions") or {}

    # 1. Python version candidates (date-range fallbacks)
    with tempfile.TemporaryDirectory() as tmpdir:
        pypi = PyPIQuery(logging=False, base_modules=tmpdir)
        python_version_candidates = pypi.get_python_range(python_version)

        # 2. Query PyPI per module; get_module_specifics writes version lists to tmpdir
        module_details = {"python_version": python_version, "python_modules": merged}
        try:
            modified_modules, _ = pypi.get_module_specifics(module_details)
        except Exception:
            modified_modules = merged

        # 3. Build pypi_versions from written files (module -> sorted version list)
        pypi_versions = {}
        for dep in modified_modules:
            path = os.path.join(tmpdir, f"{dep}_{python_version}.txt")
            if os.path.isfile(path):
                with open(path) as f:
                    raw = f.read().strip()
                    pypi_versions[dep] = [v.strip() for v in raw.split(",") if v.strip()]
            else:
                pypi_versions[dep] = []

    # 4. Rank modules: most-constrained first (in constraint_table or error_module_versions), then rest
    constrained = set()
    for entry in constraint_table:
        constrained.add(entry.get("module_a"))
        constrained.add(entry.get("module_b"))
    constrained.update(error_module_versions.keys())
    ordered = [m for m in modified_modules if m in constrained]
    ordered += [m for m in modified_modules if m not in constrained]

    resolution_plan = [{"module": m, "order": i} for i, m in enumerate(ordered)]

    return {
        "resolution_plan": resolution_plan,
        "pypi_versions": pypi_versions,
        "python_version_candidates": python_version_candidates,
    }


def replan(state: AgentState) -> str:
    """
    ROUTING based on suggested_strategy from Reflect:
    • 'swap_version' → go to Resolve Next Module (targeted fix)
    • 'add_module' → go to Resolve Next Module (with new module added)
    • 'replan' → go to Plan Resolution (full replanning with new constraints)
    • iteration >= max_iterations → FAILED

    NOVEL: PLLM never replans — it only retries with different versions.
    This allows the agent to fundamentally restructure its approach.
    """
    iteration = state.get("iteration") or 0
    max_iterations = state.get("max_iterations") or 10
    if iteration >= max_iterations:
        return "__end__"
    strategy = (state.get("suggested_strategy") or "").strip().lower()
    if strategy == "replan":
        return "plan_resolution"
    if strategy in ("swap_version", "add_module"):
        return "resolve_next_module"
    return "resolve_next_module"


def resolve_next_module(state: AgentState) -> dict:
    """
    RESOLVER AGENT — For the next unresolved module in the plan:
    1. Read filtered PyPI versions from state (pypi_versions)
    2. Exclude versions in failed_combos and error_module_versions
    3. Use OllamaHelper.get_module_versions() for constraint-aware version selection
    4. Update resolved_modules and current_module

    NOVEL: Instead of PLLM's 'equally distanced sampling', uses
    constraint-guided selection that respects accumulated knowledge.

    Maps to: OllamaHelper.get_module_versions() (enhanced)
    Updates: resolved_modules, current_module
    """
    resolution_plan = state.get("resolution_plan") or []
    resolved_modules = dict(state.get("resolved_modules") or {})
    pypi_versions = state.get("pypi_versions") or {}
    python_version = state.get("python_version") or "3.10"
    failed_combos = state.get("failed_combos") or set()
    error_module_versions = state.get("error_module_versions") or {}

    # 1. Next unresolved module from plan
    next_module = None
    for step in resolution_plan:
        mod = step.get("module") if isinstance(step, dict) else None
        if mod and mod not in resolved_modules:
            next_module = mod
            break

    if not next_module:
        return {"current_module": None, "resolved_modules": resolved_modules}

    # 2. Candidate versions: exclude failed_combos and error_module_versions
    all_versions = pypi_versions.get(next_module, [])
    bad_versions = set(error_module_versions.get(next_module, []))
    candidate_versions = [
        v for v in all_versions
        if (next_module, v) not in failed_combos and v not in bad_versions
    ]
    if not candidate_versions:
        candidate_versions = all_versions

    # 3. Use OllamaHelper.get_module_versions(): write filtered list to temp file, then call
    with tempfile.TemporaryDirectory() as tmpdir:
        versions_file = os.path.join(tmpdir, f"{next_module}_{python_version}.txt")
        with open(versions_file, "w") as f:
            f.write(", ".join(candidate_versions))
        ollama = OllamaHelper(model="gemma2", base_modules=tmpdir)
        details = {"python_modules": [next_module], "python_version": python_version}
        try:
            selected = ollama.get_module_versions(details)
        except Exception:
            selected = {next_module: candidate_versions[-1]} if candidate_versions else {next_module: "0.0.0"}
        if next_module in selected:
            resolved_modules[next_module] = selected[next_module]
        elif candidate_versions:
            resolved_modules[next_module] = candidate_versions[-1]
        else:
            resolved_modules[next_module] = "0.0.0"

    return {
        "resolved_modules": resolved_modules,
        "current_module": next_module,
    }

def all_resolved(state: AgentState) -> str:
    """
    ROUTING: Check if all modules in resolution_plan have a pinned version.
    If yes → proceed to Build. If no → resolve next module.
    """
    resolution_plan = state.get("resolution_plan") or []
    resolved_modules = state.get("resolved_modules") or {}
    planned = {step.get("module") for step in resolution_plan if isinstance(step, dict) and step.get("module")}
    if not planned:
        return "docker_build"
    if planned <= resolved_modules.keys():
        return "docker_build"
    return "resolve_next_module"


def docker_build(state: AgentState) -> dict:
    """
    BUILDER AGENT — Generate and build Dockerfile:
    1. Create Dockerfile with FROM python:{version}, pip installs (build_dockerfile.create_dockerfile)
    2. Build via Docker API (build_dockerfile.build_dockerfile)
    3. Capture full build log for error analysis

    Directly maps to: DockerHelper.create_dockerfile() + DockerHelper.build_dockerfile()
    Updates: dockerfile_content, docker_image_name, build_log, build_passed
    """
    file_path = state.get("file_path") or ""
    python_version = state.get("python_version") or "3.10"
    resolved_modules = state.get("resolved_modules") or {}

    llm_out = {
        "python_version": python_version,
        "python_modules": resolved_modules,
    }
    dh = DockerHelper(logging=False, image_name="", dockerfile_name="", container_name="")
    try:
        dh.create_dockerfile(llm_out, file_path)
        dockerfile_content = dh.dockerfile_out
        docker_image_name = dh.image_name
        build_passed, build_log = dh.build_dockerfile(file_path)
    except Exception as e:
        dockerfile_content = getattr(dh, "dockerfile_out", "") or ""
        docker_image_name = getattr(dh, "image_name", "") or ""
        build_passed = False
        build_log = str(e)

    return {
        "dockerfile_content": dockerfile_content,
        "docker_image_name": docker_image_name,
        "build_log": build_log,
        "build_passed": build_passed,
    }

def build_graph():
    """Build the LangGraph state graph."""
    builder = StateGraph(AgentState)
    builder.add_node("extract_imports", extract_imports)
    builder.add_node("plan_resolution", plan_resolution)
    builder.add_node("resolve_next_module", resolve_next_module)
    builder.add_node("docker_build", docker_build)

    builder.add_edge(START, "extract_imports")
    builder.add_edge("extract_imports", "plan_resolution")
    builder.add_edge("plan_resolution", "resolve_next_module")
    builder.add_conditional_edges("resolve_next_module", all_resolved)
    builder.add_edge("docker_build", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()

    res = graph.invoke(
        {
            "file_path": str(Path("/home/suren/programming/competition/pytohn-dependency-hell/hard-gists/0a2ac74d800a2eff9540/snippet.py"))
        }
    )

    print(res)