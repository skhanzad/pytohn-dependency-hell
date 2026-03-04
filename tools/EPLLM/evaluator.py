"""Batch evaluator with multiprocessing."""

import csv
import glob
import os
import time
import statistics
from multiprocessing import Pool
from typing import List, Dict, Optional

from tools.EPLLM.resolver import SnippetResolver


def _run_one(args: tuple) -> Dict:
    """Worker function for multiprocessing. Must be top-level for pickling."""
    file_path, base_url, model, temp, max_iterations, modules_dir, logging = args

    start = time.time()
    try:
        resolver = SnippetResolver(
            base_url=base_url, model=model, temp=temp,
            max_iterations=max_iterations, modules_dir=modules_dir,
            logging=logging,
        )
        state = resolver.resolve(file_path)
    except Exception as e:
        elapsed = time.time() - start
        snippet_name = file_path.split("/")[-2] if "/" in file_path else "unknown"
        return {
            "id": snippet_name,
            "name": snippet_name,
            "result": "error",
            "duration": round(elapsed, 2),
            "python_modules": "",
            "passed": False,
            "error": str(e),
        }

    elapsed = time.time() - start
    modules_str = ", ".join(
        f"{k}=={v}" for k, v in state.get("resolved_modules", {}).items()
    )

    passed = state.get("status") == "success"
    result = "success" if passed else state.get("error_type", "failed")

    return {
        "id": state.get("snippet_name", "unknown"),
        "name": state.get("snippet_name", "unknown"),
        "result": result,
        "duration": round(elapsed, 2),
        "python_modules": modules_str,
        "passed": passed,
        "error": state.get("error_type", "") if not passed else "",
    }


class Evaluator:
    """Batch processor for snippet evaluation."""

    CSV_COLUMNS = ["id", "name", "result", "duration", "python_modules", "passed", "error"]

    def __init__(self, hard_gists_dir: str = "hard-gists",
                 output_csv: str = "epllm-results/epllm_results.csv",
                 base_url: str = "http://localhost:11434",
                 model: str = "gemma2", temp: float = 0.2,
                 max_iterations: int = 5, jobs: int = 1,
                 limit: Optional[int] = None,
                 modules_dir: str = "./modules",
                 logging: bool = False):
        self.hard_gists_dir = hard_gists_dir
        self.output_csv = output_csv
        self.base_url = base_url
        self.model = model
        self.temp = temp
        self.max_iterations = max_iterations
        self.jobs = jobs
        self.limit = limit
        self.modules_dir = modules_dir
        self.logging = logging

    def discover_snippets(self) -> List[str]:
        """Find all snippet.py files in hard-gists subdirectories."""
        pattern = os.path.join(self.hard_gists_dir, "*/snippet.py")
        files = sorted(glob.glob(pattern))
        if self.limit:
            files = files[:self.limit]
        return files

    def run_all(self) -> List[Dict]:
        """Run evaluation on all discovered snippets using multiprocessing."""
        snippets = self.discover_snippets()
        if not snippets:
            print("[EPLLM] No snippets found!")
            return []

        print(f"[EPLLM] Found {len(snippets)} snippets, "
              f"running with {self.jobs} workers")

        # Build args for each snippet
        args_list = [
            (f, self.base_url, self.model, self.temp,
             self.max_iterations, self.modules_dir, self.logging)
            for f in snippets
        ]

        total_start = time.time()

        if self.jobs <= 1:
            # Sequential for debugging
            rows = [_run_one(a) for a in args_list]
        else:
            with Pool(processes=self.jobs) as pool:
                rows = pool.map(_run_one, args_list)

        total_elapsed = time.time() - total_start

        self._write_csv(rows)
        self._print_summary(rows, total_elapsed)

        return rows

    def _write_csv(self, rows: List[Dict]):
        """Write results to CSV."""
        os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)

        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"[EPLLM] Results written to {self.output_csv}")

    def _print_summary(self, rows: List[Dict], total_time: float = 0):
        """Print success rate, timing stats, and error distribution."""
        total = len(rows)
        if total == 0:
            print("[EPLLM] No results to summarize.")
            return

        passed = sum(1 for r in rows if r["passed"])
        durations = [r["duration"] for r in rows]
        avg_dur = statistics.mean(durations) if durations else 0
        med_dur = statistics.median(durations) if durations else 0

        # Error distribution
        error_counts: Dict[str, int] = {}
        for r in rows:
            if not r["passed"] and r["error"]:
                err = r["error"]
                error_counts[err] = error_counts.get(err, 0) + 1

        print("\n" + "=" * 60)
        print(f"EPLLM Results Summary")
        print("=" * 60)
        print(f"Total snippets:  {total}")
        print(f"Passed:          {passed} ({100 * passed / total:.1f}%)")
        print(f"Failed:          {total - passed} ({100 * (total - passed) / total:.1f}%)")
        print(f"Avg duration:    {avg_dur:.1f}s")
        print(f"Median duration: {med_dur:.1f}s")
        print(f"Total time:      {total_time:.1f}s")

        if error_counts:
            print(f"\nError distribution:")
            for err, count in sorted(error_counts.items(),
                                     key=lambda x: -x[1]):
                print(f"  {err}: {count}")
        print("=" * 60)
