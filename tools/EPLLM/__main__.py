"""CLI entry point: python -m tools.EPLLM"""

import argparse
import os
import sys
import time

from tools.EPLLM.evaluator import Evaluator, _run_one
from tools.EPLLM.resolver import SnippetResolver


def main():
    parser = argparse.ArgumentParser(
        description="EPLLM — Enhanced PLLM for Python dependency resolution"
    )
    parser.add_argument(
        "--hard-gists", type=str, default="hard-gists",
        help="Path to gists directory (default: hard-gists)"
    )
    parser.add_argument(
        "--output", type=str, default="epllm-results/epllm_results.csv",
        help="CSV output path (default: epllm-results/epllm_results.csv)"
    )
    parser.add_argument(
        "-m", "--model", type=str, default="gemma2",
        help="Ollama model (default: gemma2)"
    )
    parser.add_argument(
        "-b", "--base-url", type=str, default="http://localhost:11434",
        help="Ollama URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "-l", "--loop", type=int, default=5,
        help="Max iterations per snippet (default: 5)"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker count (default: cpu_count-1)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of snippets (for testing)"
    )
    parser.add_argument(
        "--modules-dir", type=str, default="modules",
        help="Shared modules cache directory (default: modules)"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Single file mode for testing"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.file:
        # Single file mode
        print(f"[EPLLM] Single file mode: {args.file}")
        start = time.time()
        resolver = SnippetResolver(
            base_url=args.base_url, model=args.model, temp=0.2,
            max_iterations=args.loop, modules_dir=args.modules_dir,
            logging=args.verbose or True,
        )
        state = resolver.resolve(args.file)
        elapsed = time.time() - start

        print(f"\n{'='*60}")
        print(f"Result:  {state['status']}")
        print(f"Python:  {state['python_version']}")
        print(f"Modules: {state['resolved_modules']}")
        print(f"Time:    {elapsed:.1f}s")
        print(f"Iters:   {state['iteration'] + 1}")
        if state['error_type']:
            print(f"Error:   {state['error_type']}")
        print(f"{'='*60}")
    else:
        # Batch mode
        evaluator = Evaluator(
            hard_gists_dir=args.hard_gists,
            output_csv=args.output,
            base_url=args.base_url,
            model=args.model,
            temp=0.2,
            max_iterations=args.loop,
            jobs=args.jobs,
            limit=args.limit,
            modules_dir=args.modules_dir,
            logging=args.verbose,
        )
        evaluator.run_all()


if __name__ == "__main__":
    main()
