#!/usr/bin/env python3
"""
Compare EPLLM, PLLM, pyego, and readpy results on hard-gists.
Usage: from repo root, python scripts/compare_tool_results.py
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "EPLLM": REPO_ROOT / "epllm-results" / "epllm_results.csv",
    "PLLM": REPO_ROOT / "pllm_results" / "csv" / "hard-gists-l10-r1-1-final.csv",
    "pyego": REPO_ROOT / "pyego-results" / "pyego_results.csv",
    "readpy": REPO_ROOT / "readpy-results" / "readpy_results_total.csv",
}


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def main():
    data = {}
    for tool, path in FILES.items():
        if not path.exists():
            print(f"Skip {tool}: {path} not found")
            continue
        data[tool] = load_csv(path)
        print(f"{tool}: {len(data[tool])} rows from {path.name}")

    # ---- Side-by-side summary: EPLLM vs PLLM vs pyego vs readpy ----
    print()
    print("=" * 72)
    print("  EPLLM  vs  PLLM  vs  pyego  vs  readpy")
    print("=" * 72)
    tools = list(data.keys())
    # Header
    print(f"{'Metric':<22} " + "  ".join(f"{t:>10}" for t in tools))
    print("-" * 72)
    # Rows
    n_row = [len(data[t]) for t in tools]
    print(f"{'N (snippets)':<22} " + "  ".join(f"{n_row[i]:>10}" for i in range(len(tools))))
    passed_row = []
    rate_row = []
    for t in tools:
        rows = data[t]
        n = len(rows)
        p = sum(1 for r in rows if str(r.get("passed", "")).lower() == "true")
        passed_row.append(p)
        rate_row.append(100.0 * p / n if n else 0)
    print(f"{'Passed':<22} " + "  ".join(f"{passed_row[i]:>10}" for i in range(len(tools))))
    print(f"{'Pass rate %':<22} " + "  ".join(f"{rate_row[i]:>9.1f}%" for i in range(len(tools))))
    dur_row = []
    for t in tools:
        rows = data[t]
        durs = []
        for r in rows:
            try:
                durs.append(float(r.get("duration", 0) or 0))
            except (ValueError, TypeError):
                pass
        dur_row.append(sum(durs) / len(durs) if durs else 0)
    print(f"{'Avg duration (s)':<22} " + "  ".join(f"{dur_row[i]:>10.1f}" for i in range(len(tools))))
    print("=" * 72)

    print()
    print("OVERALL (per-tool details)")
    print("-" * 70)

    for tool, rows in data.items():
        n = len(rows)
        if not n:
            continue
        # passed: can be "True"/"False" string or boolean
        passed = sum(1 for r in rows if str(r.get("passed", "")).lower() == "true")
        rate = 100.0 * passed / n
        durations = []
        for r in rows:
            try:
                d = float(r.get("duration", 0) or 0)
                durations.append(d)
            except (ValueError, TypeError):
                pass
        avg_dur = sum(durations) / len(durations) if durations else 0
        print(f"\n{tool} (n={n})")
        print(f"  Passed:     {passed} / {n}  ({rate:.1f}%)")
        print(f"  Avg time:   {avg_dur:.2f}s")
        # result distribution
        results: dict[str, int] = {}
        for r in rows:
            res = (r.get("result") or "None").strip() or "None"
            results[res] = results.get(res, 0) + 1
        print("  Result distribution:")
        for res, count in sorted(results.items(), key=lambda x: -x[1]):
            print(f"    {res}: {count}")

    # Intersection: snippets present in all datasets (by name)
    names_by_tool = {tool: {r.get("name") for r in rows if r.get("name")} for tool, rows in data.items()}
    common = set(names_by_tool[list(data.keys())[0]])
    for tool, names in names_by_tool.items():
        common &= names
    common = sorted(common)
    print()
    print("=" * 70)
    print(f"INTERSECTION: {len(common)} snippets present in all tools")
    print("=" * 70)

    if not common:
        print("No common names; datasets may use different name columns.")
        return

    # Build name -> row for each tool (take first if duplicate name)
    by_name: dict[str, dict[str, dict]] = {tool: {} for tool in data}
    for tool, rows in data.items():
        for r in rows:
            name = r.get("name")
            if name and name not in by_name[tool]:
                by_name[tool][name] = r

    print()
    for tool in data:
        passed = sum(1 for name in common if str(by_name[tool].get(name, {}).get("passed", "")).lower() == "true")
        print(f"  {tool}: {passed} / {len(common)} passed ({100.0 * passed / len(common):.1f}%)")

    # Per-snippet agreement on common set
    print()
    print("Agreement (on intersection):")
    all_pass = sum(1 for name in common if all(str(by_name[t].get(name, {}).get("passed", "")).lower() == "true" for t in data))
    all_fail = sum(1 for name in common if all(str(by_name[t].get(name, {}).get("passed", "")).lower() != "true" for t in data))
    print(f"  All tools passed:  {all_pass}")
    print(f"  All tools failed:  {all_fail}")
    print(f"  Disagreement:      {len(common) - all_pass - all_fail}")


if __name__ == "__main__":
    main()
