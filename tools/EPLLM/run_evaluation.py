#!/usr/bin/env python3
"""
Run EPLLM graph on all snippets under --hard-gists and write id,name,result,duration,python_modules,passed,error to CSV.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# Ensure EPLLM dir is on path so we can import graph
_epllm_dir = Path(__file__).resolve().parent
if str(_epllm_dir) not in sys.path:
    sys.path.insert(0, str(_epllm_dir))

from graph import build_graph


def discover_snippets(hard_gists_root: Path, limit: int | None) -> list[Path]:
    """Discover all snippet.py files under hard_gists_root, optionally limited."""
    snippets = sorted(hard_gists_root.rglob("snippet.py"))
    if limit is not None:
        snippets = snippets[:limit]
    return snippets


def row_from_state(
    row_id: int,
    name: str,
    state: dict,
    duration_sec: float,
) -> dict:
    """Build a CSV row from final graph state and timing."""
    passed = bool(state.get("run_passed"))
    result = "OtherPass" if passed else (state.get("error_type") or "Failed")
    resolved = state.get("resolved_modules") or {}
    python_modules = ",".join(f"{k}=={v}" for k, v in resolved.items()) if resolved else ""
    error = ""
    if not passed:
        error = (state.get("run_log") or state.get("build_log") or "")[:500]
    return {
        "id": row_id,
        "name": name,
        "result": result,
        "duration": f"{duration_sec:.2f}",
        "python_modules": python_modules,
        "passed": passed,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EPLLM on hard-gists snippets and write epllm_results.csv",
    )
    parser.add_argument(
        "--hard-gists",
        type=Path,
        default=Path("hard-gists"),
        help="Root directory containing snippet subdirs (default: hard-gists)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("epllm-results/epllm_results.csv"),
        help="Output CSV path (default: epllm-results/epllm_results.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of snippets to run (default: all)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemma2",
        help="Ollama model name (default: gemma2)",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=int,
        default=5,
        dest="max_iterations",
        help="Max build/error-handling iterations (default: 5)",
    )
    args = parser.parse_args()

    hard_gists = args.hard_gists.resolve()
    if not hard_gists.is_dir():
        print(f"Error: --hard-gists is not a directory: {hard_gists}", file=sys.stderr)
        sys.exit(1)

    snippets = discover_snippets(hard_gists, args.limit)
    if not snippets:
        print(f"No snippet.py files under {hard_gists}", file=sys.stderr)
        sys.exit(1)

    print(f"Running EPLLM on {len(snippets)} snippets (model={args.model}, max_iterations={args.max_iterations})")
    graph = build_graph()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "name", "result", "duration", "python_modules", "passed", "error"]
    rows: list[dict] = []

    for i, snippet_path in enumerate(snippets, start=1):
        name = snippet_path.parent.name
        file_path = str(snippet_path)
        print(f"  [{i}/{len(snippets)}] {name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            state = graph.invoke(
                {
                    "file_path": file_path,
                    "max_iterations": args.max_iterations,
                }
            )
        except Exception as e:
            state = {
                "run_passed": False,
                "build_passed": False,
                "error_type": "Error",
                "resolved_modules": {},
                "run_log": str(e),
                "build_log": "",
            }
        duration = time.perf_counter() - t0
        row = row_from_state(i, name, state, duration)
        rows.append(row)
        print(f"    -> {row['result']} ({duration:.2f}s)", flush=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
