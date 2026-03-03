# LangGraph pipeline: analyze -> resolve_versions -> build -> [run_container | handle_error -> resolve_versions].
from typing import Literal

from langgraph.graph import StateGraph, START, END

from state import EPLLMState
from agents.analyzer import analyze_file_node
from agents.version_resolver import resolve_versions_node
from agents.builder import build_node, run_container_node
from agents.error_handler import handle_error_node


def _route_after_build(state: EPLLMState) -> Literal["run_container", "handle_error"]:
    if state.get("build_complete"):
        return "run_container"
    return "handle_error"


def _route_after_run(state: EPLLMState) -> Literal["__end__", "handle_error"]:
    if state.get("run_complete"):
        return "__end__"
    return "handle_error"


def _route_after_handle_error(state: EPLLMState) -> Literal["build", "__end__"]:
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    if iteration >= max_iterations:
        return "__end__"
    return "build"


def build_graph():
    graph = StateGraph(EPLLMState)

    graph.add_node("analyze_file", analyze_file_node)
    graph.add_node("resolve_versions", resolve_versions_node)
    graph.add_node("build", build_node)
    graph.add_node("run_container", run_container_node)
    graph.add_node("handle_error", handle_error_node)

    graph.add_edge(START, "analyze_file")
    graph.add_edge("analyze_file", "resolve_versions")
    graph.add_edge("resolve_versions", "build")
    graph.add_conditional_edges("build", _route_after_build, {"run_container": "run_container", "handle_error": "handle_error"})
    graph.add_conditional_edges("run_container", _route_after_run, {"__end__": END, "handle_error": "handle_error"})
    graph.add_conditional_edges("handle_error", _route_after_handle_error, {"build": "build", "__end__": END})

    return graph.compile()
