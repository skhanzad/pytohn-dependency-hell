# Agent 2: Resolves concrete module versions for current Python version using PyPI + LLM (and RAG).
from typing import Optional
from langchain_core.runnables.config import RunnableConfig
from state import EPLLMState


def resolve_versions_node(state: EPLLMState, config: Optional[RunnableConfig] = None) -> dict:
    cfg = (config or {}).get("configurable", {})
    ollama_helper = cfg.get("ollama_helper")
    llm_eval = state.get("llm_eval")

    if not ollama_helper or not llm_eval:
        return {}

    # get_module_specifics: PyPI + file; get_module_versions: LLM per module (uses RAG in pllm if rag=True)
    llm_eval = ollama_helper.get_module_specifics(llm_eval.copy())
    return {"llm_eval": llm_eval}
