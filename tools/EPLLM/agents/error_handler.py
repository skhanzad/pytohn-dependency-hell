# Agent 4: Interprets last_error_type and last_fix_output to update state for retry.
# In our flow, build_node and run_container_node already call process_error and update_llm_eval;
# this node can apply additional RAG-based suggestions or logging.
from typing import Optional
from langchain_core.runnables.config import RunnableConfig
from state import EPLLMState


def handle_error_node(state: EPLLMState, config: Optional[RunnableConfig] = None) -> dict:
    # State was already updated in build_node / run_container_node; we inject RAG context for next step.
    cfg = (config or {}).get("configurable", {})
    rag_store = cfg.get("rag_store")
    docker_output = state.get("docker_output", "")
    use_rag = state.get("use_rag", True)

    rag_context = ""
    if use_rag and rag_store and rag_store.is_available():
        rag_context = rag_store.retrieve_for_error(docker_output[:2000], k=2)

    iteration = state.get("iteration", 0) + 1
    return {"rag_context": rag_context, "iteration": iteration}
