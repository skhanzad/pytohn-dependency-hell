# EPLLM — Standalone tool (LangGraph + multi-agent + RAG)

EPLLM is **separate from** [pllm](../pllm/): it does not import or depend on `tools/pllm/`. It implements its own core (vendored logic for LLM, PyPI, deps, Docker) and adds:

- **LangGraph** — Orchestrates the pipeline as a state graph with clear nodes and conditional edges.
- **Multi-agent** — Separate agents for: file analysis, version resolution, Docker build/run, and error handling.
- **RAG** — Vector store over module version docs (and optional error→fix examples) to augment LLM context.

Same high-level behavior as pllm: given a Python snippet file, infer Python version and dependencies, build a Docker image, run it, and interpret build/run errors to retry with updated versions. Results can be evaluated on the **hard-gists** dataset and compared to **pllm_results**, **readpy-results**, and **pyego-results**.

## Requirements

- Python 3.11+
- Docker (for building and running snippet containers)
- Ollama (or compatible API) and a model (e.g. phi3:medium, gemma2)
- EPLLM deps: langgraph, langchain-core, langchain-community, docarray; plus pypi-json, docker, requests, etc. (see requirements.txt). **No dependency on tools/pllm.**

## Install

From repo root or `tools/pllm` env:

```bash
pip install -r tools/EPLLM/requirements.txt
```

Optional: for RAG embeddings, an embedding model in Ollama (e.g. `nomic-embed-text`) is used; the store falls back to no retrieval if dependencies are missing.

## Usage

Same CLI as pllm’s `test_executor.py`:

```bash
# From repo root (or ensure tools/pllm is on PYTHONPATH and cwd for ref_files)
python tools/EPLLM/main.py -f /path/to/snippet.py -m gemma2 -b http://ollama:11434 -l 5 -ra true -v
```

| Flag | Description |
|------|-------------|
| `-f` / `--file` | Full path to the Python file to evaluate (required). |
| `-b` / `--base` | Ollama base URL (default: `http://localhost:11434`). |
| `-m` / `--model` | Model name (default: `phi3:medium`). |
| `-t` / `--temp` | Temperature (default: `0.7`). |
| `-l` / `--loop` | Max build/error-handling iterations (default: `5`). |
| `-r` / `--range` | Python version search range; `0` = single version (default: `0`). |
| `-ra` / `--rag` | Enable RAG over module docs (default: `true`). |
| `-v` / `--verbose` | Verbose logging. |

## Graph Overview

1. **analyze_file** — Reads the file, scrapes imports, runs LLM to get `python_version` and `python_modules`; optionally augments with RAG over module docs.
2. **resolve_versions** — Uses PyPI + LLM (and pllm’s RAG when `-ra true`) to resolve concrete module versions for the chosen Python version.
3. **build** — Creates Dockerfile, runs build; on failure, runs pllm’s `process_error` and updates `llm_eval` / `error_handler`, then routes to **handle_error**.
4. **run_container** — Runs the container; on failure, routes to **handle_error**.
5. **handle_error** — Optionally retrieves similar error→fix context from RAG, increments iteration; then routes back to **build** (or END if `iteration >= max_iterations`).

RAG is used in the analyzer (module/version docs) and in the error handler (similar errors). EPLLM uses its own **core** (under `core/`: OllamaHelper, PyPIQuery, DepsScraper, DockerHelper, Executor) and does not depend on `tools/pllm/`.

### Evaluation on hard-gists (compare to pllm / readpy / pyego)

Run EPLLM on the same dataset as pllm_results, readpy-results, and pyego-results:

```bash
# From repo root; extract hard-gists first (e.g. tar -xzf hard-gists.tar.gz)
python tools/EPLLM/run_evaluation.py --hard-gists /path/to/hard-gists --output epllm-results/epllm_results.csv
```

Output CSV has columns `id`, `name`, `result`, `duration`, `python_modules`, `passed` for comparison. See **epllm-results/README.md** in the repo root.
