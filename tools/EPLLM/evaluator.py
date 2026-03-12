"""Batch evaluation with multiprocessing and CSV output.

Processes multiple snippets in parallel, collects results,
and outputs in the normalized CSV format for comparison.
"""
import csv
import os
import time
import multiprocessing as mp
from pathlib import Path

from EPLLM.resolver import SnippetResolver  # noqa: E402
from EPLLM.state import ResolveResult  # noqa: E402


def _resolve_snippet(args):
    """Worker function for multiprocessing pool."""
    snippet_path, config = args
    snippet_id = os.path.basename(os.path.dirname(snippet_path))
    start = time.time()

    try:
        resolver = SnippetResolver(
            base_modules=config['base_modules'],
            max_iterations=config['max_iterations'],
            llm_base_url=config.get('llm_base_url', 'http://localhost:11434'),
            llm_model=config.get('llm_model', 'phi3:medium'),
            llm_temp=config.get('llm_temp', 0.3),
            use_llm=config.get('use_llm', True),
            logging=config.get('logging', False),
        )
        result = resolver.resolve(snippet_path)
    except Exception as e:
        result = ResolveResult(
            success=False, error_type='Exception',
            error_message=str(e)[:200],
            duration=time.time() - start,
        )

    # Always set duration to wall-clock elapsed time
    result.duration = time.time() - start
    return snippet_id, result


class BatchEvaluator:
    """Evaluates multiple snippets in parallel and produces CSV results."""

    def __init__(self, config):
        self.config = config
        self.results = []

    def discover_snippets(self, base_dir, limit=None):
        """Find all snippet.py files under the base directory."""
        snippets = []
        base = Path(base_dir)

        if base.is_file() and base.name.endswith('.py'):
            return [str(base)]

        for snippet_dir in sorted(base.iterdir()):
            if snippet_dir.is_dir():
                snippet_file = snippet_dir / 'snippet.py'
                if snippet_file.exists():
                    snippets.append(str(snippet_file))

        if limit and limit > 0:
            snippets = snippets[:limit]

        return snippets

    def evaluate(self, snippets, workers=None):
        """Run evaluation on all snippets using multiprocessing.

        Args:
            snippets: List of snippet file paths.
            workers: Number of parallel workers (default: cpu_count - 1).

        Returns:
            List of (snippet_id, ResolveResult) tuples.
        """
        if workers is None:
            workers = max(1, mp.cpu_count() - 1)

        # Prepare args for pool
        args_list = [(s, self.config) for s in snippets]

        total = len(snippets)
        print(f"Evaluating {total} snippets with {workers} workers...")
        start = time.time()
        completed = 0

        if workers == 1:
            # Sequential mode (easier debugging)
            for args in args_list:
                snippet_id, result = _resolve_snippet(args)
                self.results.append((snippet_id, result))
                completed += 1
                status = "PASS" if result.success else f"FAIL({result.error_type})"
                print(f"  [{completed}/{total}] {snippet_id}: {status} "
                      f"({result.duration:.1f}s)")
        else:
            # Parallel mode
            with mp.Pool(workers) as pool:
                for snippet_id, result in pool.imap_unordered(
                    _resolve_snippet, args_list
                ):
                    self.results.append((snippet_id, result))
                    completed += 1
                    status = "PASS" if result.success else f"FAIL({result.error_type})"
                    print(f"  [{completed}/{total}] {snippet_id}: {status} "
                          f"({result.duration:.1f}s)")

        elapsed = time.time() - start
        passed = sum(1 for _, r in self.results if r.success)
        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")

        return self.results

    def write_csv(self, output_path):
        """Write results to CSV in the normalized comparison format.

        Columns: id, name, result, duration, python_modules, passed
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'result', 'duration',
                             'python_modules', 'passed'])

            for snippet_id, result in sorted(self.results, key=lambda x: x[0]):
                if result.success:
                    result_str = 'OtherPass'
                else:
                    result_str = result.error_type or 'Unknown'

                modules_str = result.modules_str()

                writer.writerow([
                    snippet_id,
                    snippet_id,
                    result_str,
                    f"{result.duration:.2f}",
                    modules_str,
                    result.success,
                ])

        print(f"Results written to {output_path}")

    def print_summary(self):
        """Print a summary of the evaluation results."""
        total = len(self.results)
        if total == 0:
            print("No results to summarize.")
            return

        passed = sum(1 for _, r in self.results if r.success)
        failed = total - passed
        avg_time = sum(r.duration for _, r in self.results) / total

        # Count error types
        error_counts = {}
        for _, r in self.results:
            if not r.success:
                et = r.error_type or 'Unknown'
                error_counts[et] = error_counts.get(et, 0) + 1

        print(f"\n{'='*50}")
        print(f"EPLLM Evaluation Summary")
        print(f"{'='*50}")
        print(f"Total:     {total}")
        print(f"Passed:    {passed} ({100*passed/total:.1f}%)")
        print(f"Failed:    {failed} ({100*failed/total:.1f}%)")
        print(f"Avg time:  {avg_time:.2f}s")
        print()

        if error_counts:
            print("Error breakdown:")
            for et, count in sorted(error_counts.items(),
                                    key=lambda x: -x[1]):
                print(f"  {et:25s} {count:4d} ({100*count/total:.1f}%)")

        print(f"{'='*50}")
