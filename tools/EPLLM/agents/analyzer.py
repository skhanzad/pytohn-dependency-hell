# Agent 1: Analyzes the Python file and produces initial llm_eval (python_version + python_modules).
# Uses RAG to augment context when available.
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from state import EPLLMState


def analyze_file_node(state: EPLLMState, config: Optional[RunnableConfig] = None) -> dict:
    cfg = (config or {}).get("configurable", {})
    ollama_helper = cfg.get("ollama_helper")
    deps = cfg.get("deps_scraper")
    pypi = cfg.get("pypi_query")
    rag_store = cfg.get("rag_store")
    file_path = state["file_path"]
    use_rag = state.get("use_rag", True)

    if not ollama_helper or not file_path:
        return {"llm_eval": None, "scraped_imports": []}

    # Scrape imports from file (RAG-style: use file content for retrieval later)
    scraped_imports = []
    if deps:
        scraped_imports = deps.find_word_in_file(file_path, "import", [])

    # Initial LLM evaluation
    llm_eval = ollama_helper.evaluate_file(file_path)
    llm_eval["python_version"] = str(llm_eval["python_version"])
    python_modules = llm_eval.get("python_modules")
    if isinstance(python_modules, dict):
        llm_eval["python_modules"] = list(python_modules.keys()) if python_modules else []

    # Merge scraped imports and normalize via PyPI
    if pypi:
        combined = pypi.check_module_name(scraped_imports + list(llm_eval.get("python_modules", [])))
        llm_eval["python_modules"] = combined

    # Optional RAG: inject context from module/version docs for next step
    rag_context = ""
    if use_rag and rag_store and rag_store.is_available():
        for mod in llm_eval.get("python_modules", [])[:5]:
            rag_context += rag_store.retrieve_for_module_version(mod, llm_eval["python_version"], k=1)

    return {
        "llm_eval": llm_eval,
        "scraped_imports": scraped_imports,
        "rag_context": rag_context,
    }
