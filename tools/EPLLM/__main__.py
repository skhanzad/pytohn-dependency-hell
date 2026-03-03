#!/usr/bin/env python3
"""
EPLLM: Standalone tool (LangGraph + multi-agent + RAG).
Does not depend on tools/pllm. Evaluate snippets and compare to pllm_results, readpy-results, pyego-results.
"""
import argparse
import os
import sys
from pathlib import Path

_EPLLM_ROOT = Path(__file__).resolve().parent
if str(_EPLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_EPLLM_ROOT))


def _str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got "{value}".')


def process_args():
    p = argparse.ArgumentParser(description="EPLLM: evaluate Python file (standalone, no pllm dependency)")
    p.add_argument("-f", "--file", type=str, required=True, help="Full path to the Python file to evaluate")
    p.add_argument("-b", "--base", type=str, default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("-m", "--model", type=str, default="phi3:medium", help="Model name")
    p.add_argument("-t", "--temp", type=float, default=0.2, help="Temperature (lower=less hallucination, default 0.2)")
    p.add_argument("-l", "--loop", type=int, default=5, help="Max iterations for build/error loop")
    p.add_argument("-r", "--range", type=int, default=0, help="Search range for Python versions (0 = single version)")
    p.add_argument("-ra", "--rag", type=_str2bool, default=True, help="Enable RAG")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = process_args()

    from state import EPLLMState
    from graph import build_graph
    from rag.store import RAGStore
    from core import OllamaHelper, PyPIQuery, DepsScraper, DockerHelper, Executor

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    file_dir = str(file_path.parent)
    file_path = str(file_path)
    base_modules = os.path.join(file_dir, "modules")
    os.makedirs(base_modules, exist_ok=True)

    ollama_helper = OllamaHelper(
        base_url=args.base,
        model=args.model,
        logging=bool(args.verbose),
        temp=args.temp,
        base_modules=base_modules,
        rag=args.rag,
    )
    pypi = PyPIQuery(logging=bool(args.verbose), base_modules=base_modules)
    deps = DepsScraper(logging=bool(args.verbose))
    docker_helper = DockerHelper(logging=bool(args.verbose))
    executor = Executor(base_modules=base_modules, logging=bool(args.verbose))

    rag_store = RAGStore(base_modules_dir=base_modules, base_url=args.base)
    if args.rag:
        rag_store.build_from_modules_dir()

    initial_state: EPLLMState = {
        "file_path": file_path,
        "base_url": args.base,
        "model": args.model,
        "temp": args.temp,
        "base_modules": base_modules,
        "use_rag": args.rag,
        "error_handler": {
            "previous": "",
            "error_modules": {},
            "ImportError": 0,
            "ModuleNotFound": 0,
            "VersionNotFound": 0,
            "DependencyConflict": 0,
            "AttributeError": 0,
            "NonZeroCode": 0,
            "SyntaxError": 0,
        },
        "iteration": 0,
        "max_iterations": args.loop,
    }

    config = {
        "configurable": {
            "ollama_helper": ollama_helper,
            "pypi_query": pypi,
            "deps_scraper": deps,
            "docker_helper": docker_helper,
            "executor": executor,
            "rag_store": rag_store,
        }
    }

    graph = build_graph()
    print("Running EPLLM graph (analyze -> resolve -> build -> run | handle_error)...")
    try:
        for event in graph.stream(initial_state, config=config):
            for node_name, node_out in event.items():
                if args.verbose:
                    print(f"  [{node_name}] -> {list(node_out.keys())}")
                if node_name == "build" and node_out.get("build_complete"):
                    print("Build complete.")
                if node_name == "run_container":
                    if node_out.get("run_complete"):
                        print("Run complete.")
                    else:
                        print("Run failed; error handler will retry.")
    except Exception as e:
        print(f"Graph run failed: {e}")
        raise
    finally:
        try:
            docker_helper.delete_container()
        except Exception:
            pass
        try:
            docker_helper.delete_image()
        except Exception:
            pass
    print("Done.")


if __name__ == "__main__":
    main()
