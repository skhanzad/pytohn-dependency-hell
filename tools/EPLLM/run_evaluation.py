#!/usr/bin/env python3
"""
Run EPLLM on the hard-gists dataset and write results to epllm-results/ for comparison with
pllm_results, readpy-results, and pyego-results.
Usage:
  python run_evaluation.py --hard-gists /path/to/hard-gists [--output epllm-results/epllm_results.csv] [--limit 10] [--model gemma2] ...
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

_EPLLM_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _EPLLM_ROOT.parent.parent
if str(_EPLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_EPLLM_ROOT))


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got "{v}".')


def main():
    p = argparse.ArgumentParser(description="EPLLM: run evaluation on hard-gists, output CSV for comparison")
    p.add_argument("--hard-gists", type=str, required=True, help="Path to hard-gists directory (contains <id>/snippet.py)")
    p.add_argument("--output", type=str, default=None, help="Output CSV path (default: epllm-results/epllm_results.csv under repo root)")
    p.add_argument("--limit", type=int, default=None, help="Max number of snippets to run (default: all)")
    p.add_argument("-b", "--base", type=str, default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("-m", "--model", type=str, default="phi3:medium", help="Model name")
    p.add_argument("-t", "--temp", type=float, default=0.2, help="Temperature (lower=less hallucination, default 0.2)")
    p.add_argument("-l", "--loop", type=int, default=5)
    p.add_argument("-r", "--range", type=int, default=0)
    p.add_argument("-ra", "--rag", type=_str2bool, default=True)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    hard_gists = Path(args.hard_gists).resolve()
    if not hard_gists.is_dir():
        print(f"Not a directory: {hard_gists}")
        sys.exit(1)

    # Discover snippet.py paths: hard-gists/<name>/snippet.py
    snippets = []
    for d in sorted(hard_gists.iterdir()):
        if d.is_dir():
            snip = d / "snippet.py"
            if snip.is_file():
                snippets.append((d.name, str(snip)))
    if args.limit:
        snippets = snippets[: args.limit]
    print(f"Found {len(snippets)} snippets under {hard_gists}")

    out_path = args.output
    if not out_path:
        out_dir = _REPO_ROOT / "epllm-results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "epllm_results.csv"
    else:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    from state import EPLLMState
    from graph import build_graph
    from rag.store import RAGStore
    from core import OllamaHelper, PyPIQuery, DepsScraper, DockerHelper, Executor

    graph = build_graph()
    rows = []

    for idx, (name, file_path) in enumerate(snippets):
        file_dir = str(Path(file_path).parent)
        base_modules = os.path.join(file_dir, "modules")
        os.makedirs(base_modules, exist_ok=True)

        ollama_helper = OllamaHelper(
            base_url=args.base, model=args.model, logging=bool(args.verbose),
            temp=args.temp, base_modules=base_modules, rag=args.rag,
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
                "previous": "", "error_modules": {},
                "ImportError": 0, "ModuleNotFound": 0, "VersionNotFound": 0,
                "DependencyConflict": 0, "AttributeError": 0, "NonZeroCode": 0, "SyntaxError": 0,
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

        start = time.time()
        result_label = "OtherFailure"
        passed = False
        python_modules_str = ""

        try:
            merged = dict(initial_state)
            for event in graph.stream(initial_state, config=config):
                for node_name, node_out in event.items():
                    merged.update(node_out)
                    if node_name == "run_container":
                        if node_out.get("run_complete"):
                            passed = True
                            result_label = "OtherPass" if merged.get("last_error_type") in (None, "None") else (merged.get("last_error_type") or "OtherPass")
                        else:
                            result_label = merged.get("last_error_type") or "OtherFailure"
            mods = merged.get("llm_eval", {}).get("python_modules", {})
            if isinstance(mods, dict):
                python_modules_str = ";".join(mods.keys())
            elif isinstance(mods, list):
                python_modules_str = ";".join(str(m) for m in mods)
        except Exception as e:
            if args.verbose:
                print(f"[{name}] error: {e}")
            result_label = "Error"
        finally:
            try:
                docker_helper.delete_container()
            except Exception:
                pass
            try:
                docker_helper.delete_image()
            except Exception:
                pass

        duration = round(time.time() - start, 2)
        rows.append({
            "id": 1,
            "name": name,
            "result": result_label,
            "duration": duration,
            "python_modules": python_modules_str,
            "passed": passed,
        })
        if args.verbose or (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(snippets)}] {name} -> {result_label} passed={passed} {duration}s")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "result", "duration", "python_modules", "passed"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Compare with: pllm_results/csv/*.csv, readpy-results/readpy_results_total.csv, pyego-results/pyego_results.csv")


if __name__ == "__main__":
    main()
