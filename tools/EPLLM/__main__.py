"""EPLLM: Enhanced Python LLM - Hybrid dependency resolver.
CLI entry point supporting both single-file and batch evaluation modes.

Usage:
    # Single snippet
    python -m EPLLM -f /path/to/snippet.py

    # Batch evaluation (all snippets in directory)
    python -m EPLLM -d /path/to/hard-gists/ -o results.csv

    # Batch with limit
    python -m EPLLM -d /path/to/hard-gists/ -o results.csv --limit 50

    # Without LLM (pure deterministic)
    python -m EPLLM -f /path/to/snippet.py --no-llm

    # With custom model
    python -m EPLLM -f /path/to/snippet.py -m gemma2 -b http://localhost:11434
"""
import argparse
import os
import sys

from EPLLM.resolver import SnippetResolver  # noqa: E402
from EPLLM.evaluator import BatchEvaluator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='EPLLM: Hybrid Python dependency resolver '
                    '(PyEGo static analysis + PLLM iterative Docker validation)'
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-f', '--file', type=str,
        help='Single Python file to resolve'
    )
    input_group.add_argument(
        '-d', '--directory', type=str,
        help='Directory containing snippet subdirectories (batch mode)'
    )

    # Output
    parser.add_argument(
        '-o', '--output', type=str, default='epllm_results.csv',
        help='Output CSV path for batch mode (default: epllm_results.csv)'
    )

    # Resolution parameters
    parser.add_argument(
        '-l', '--loop', type=int, default=7,
        help='Max iterations per Python version attempt (default: 7)'
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='Limit number of snippets in batch mode (0 = all)'
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=0,
        help='Number of parallel workers (default: cpu_count - 1, use 1 for debugging)'
    )

    # LLM parameters
    parser.add_argument(
        '-b', '--base', type=str, default='http://localhost:11434',
        help='Ollama API base URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '-m', '--model', type=str, default='phi3:medium',
        help='LLM model name (default: phi3:medium)'
    )
    parser.add_argument(
        '-t', '--temp', type=float, default=0.3,
        help='LLM temperature (default: 0.3)'
    )
    parser.add_argument(
        '--no-llm', action='store_true',
        help='Disable LLM fallback (pure deterministic mode)'
    )

    # Misc
    parser.add_argument(
        '--modules-dir', type=str, default='',
        help='Base directory for version + success-memory cache '
             '(default: snippet dir + /modules)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose logging'
    )

    return parser.parse_args()


def run_single(args):
    """Resolve a single snippet."""
    filepath = os.path.abspath(args.file)
    file_dir = os.path.dirname(filepath)
    modules_dir = args.modules_dir or os.path.join(file_dir, 'modules')

    print("EPLLM Hybrid Resolver v2.0")
    print(f"File: {filepath}")
    print(f"Mode: {'deterministic' if args.no_llm else 'hybrid (deterministic + LangGraph fallback)'}")
    print()

    resolver = SnippetResolver(
        base_modules=modules_dir,
        max_iterations=args.loop,
        llm_base_url=args.base,
        llm_model=args.model,
        llm_temp=args.temp,
        use_llm=not args.no_llm,
        logging=args.verbose,
    )

    result = resolver.resolve(filepath)

    print(f"\n{'='*50}")
    if result.success:
        print(f"SUCCESS in {result.iterations} iterations ({result.duration:.1f}s)")
        print(f"Python version: {result.python_version}")
        print("Resolved modules:")
        for pkg, ver in result.modules.items():
            print(f"  {pkg}=={ver}")
    else:
        print(f"FAILED: {result.error_type} ({result.duration:.1f}s)")
        print(f"Python version tried: {result.python_version}")
        print(f"Iterations: {result.iterations}")
        if result.error_message:
            print(f"Last error: {result.error_message[:200]}")
    print(f"{'='*50}")

    return 0 if result.success else 1


def run_batch(args):
    """Evaluate multiple snippets."""
    base_dir = os.path.abspath(args.directory)
    modules_dir = args.modules_dir or os.path.join(
        os.path.dirname(base_dir), 'modules'
    )

    print("EPLLM Hybrid Resolver v2.0 - Batch Mode")
    print(f"Directory: {base_dir}")
    print(f"Mode: {'deterministic' if args.no_llm else 'hybrid (deterministic + LangGraph fallback)'}")
    print()

    config = {
        'base_modules': modules_dir,
        'max_iterations': args.loop,
        'llm_base_url': args.base,
        'llm_model': args.model,
        'llm_temp': args.temp,
        'use_llm': not args.no_llm,
        'logging': args.verbose,
    }

    evaluator = BatchEvaluator(config)

    # Discover snippets
    limit = args.limit if args.limit > 0 else None
    snippets = evaluator.discover_snippets(base_dir, limit=limit)

    if not snippets:
        print(f"No snippets found in {base_dir}")
        return 1

    # Run evaluation
    workers = args.workers if args.workers > 0 else None
    evaluator.evaluate(snippets, workers=workers)

    # Output results
    evaluator.write_csv(args.output)
    evaluator.print_summary()

    return 0


def main():
    args = parse_args()

    if args.file:
        sys.exit(run_single(args))
    else:
        sys.exit(run_batch(args))


if __name__ == '__main__':
    main()
