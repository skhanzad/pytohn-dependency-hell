# EPLLM evaluation results

This directory holds EPLLM output when run on the **hard-gists** dataset for comparison with:

- **pllm_results/** — PLLM baseline (e.g. `pllm_results/csv/hard-gists-l10-r1-1-final.csv`)
- **readpy-results/** — readpy tool (e.g. `readpy-results/readpy_results_total.csv`)
- **pyego-results/** — pyego tool (e.g. `pyego-results/pyego_results.csv`)

## Generating results

From the repo root, with **hard-gists** extracted (e.g. from `hard-gists.tar.gz`):

```bash
# Run EPLLM on all snippets under hard-gists (requires Ollama + Docker)
python tools/EPLLM/run_evaluation.py --hard-gists /path/to/hard-gists

# Optional: limit snippets, set model, output path
python tools/EPLLM/run_evaluation.py --hard-gists ./hard-gists --output epllm-results/epllm_results.csv --limit 100 -m gemma2 -l 5
```

Output CSV columns: `id`, `name`, `result`, `duration`, `python_modules`, `passed` — aligned with pllm/readpy for comparison.

## Comparing tools

- **Success rate**: count rows where `passed` is True.
- **Result distribution**: group by `result` (OtherPass, ImportError, ModuleNotFound, SyntaxError, etc.).
- **Efficiency**: compare `duration` and number of resolved snippets across pllm_results, readpy-results, pyego-results, and epllm-results.
