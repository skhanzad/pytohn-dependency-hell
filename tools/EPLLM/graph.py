from token import STAR
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph.message import MessagesState
from state import AgentState
from pathlib import Path
import sys
import json

# Add tools/pllm so pllm helpers can be imported
_pllm_dir = Path(__file__).resolve().parent.parent
if _pllm_dir.is_dir() and str(_pllm_dir) not in sys.path:
    sys.path.insert(0, str(_pllm_dir))


from pllm.helpers.deps_scraper import DepsScraper
from pllm.helpers.ollama_helper_tester import OllamaHelper

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



def build_graph():
    """Build the LangGraph state graph."""
    builder = StateGraph(AgentState)
    builder.add_node("extract_imports", extract_imports)

    builder.add_edge(START, "extract_imports")
    builder.add_edge("extract_imports", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()

    res = graph.invoke(
        {
            "file_path": str(Path("/home/suren/programming/competition/pytohn-dependency-hell/hard-gists/0a2ac74d800a2eff9540/snippet.py"))
        }
    )

    print(res)