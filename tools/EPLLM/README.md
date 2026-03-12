# EPLLM Docker Setup

`EPLLM` needs Docker access because it validates dependency guesses by building
and running snippet-specific containers. The image in this directory now follows
the same pattern as `pllm/Dockerfile`: it creates a host-mapped user, fixes
Docker socket group access in the entrypoint, and starts as an interactive
workspace container by default.

## Build

Run these commands on the host machine from `tools/EPLLM`:

```bash
docker build \
  --build-arg UNAME="$(whoami)" \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  --build-arg DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)" \
  -t epllm:latest \
  .
```

If you are on macOS, use:

```bash
stat -f '%g' /var/run/docker.sock
```

instead of `stat -c`.

## Run The Workspace Container

```bash
docker run -it --rm \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v "$(pwd):/app" \
  --name epllm \
  epllm:latest bash
```

Inside the container, you can then run:

```bash
python -m EPLLM --help
```

## Run EPLLM Directly

Single snippet, deterministic mode:

```bash
docker run -it --rm \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v "$(pwd):/app" \
  --name epllm \
  epllm:latest \
  python -m EPLLM \
  -f /app/path/to/snippet.py \
  --modules-dir /app/modules \
  --no-llm
```

Batch mode:

```bash
docker run -it --rm \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v "$(pwd):/app" \
  --name epllm \
  epllm:latest \
  python -m EPLLM \
  -d /app/hard-gists \
  -o /app/epllm_results.csv \
  --modules-dir /app/modules \
  --no-llm
```

## Ollama Fallback

If you want LangGraph + Ollama fallback, point EPLLM at a host-reachable Ollama
server:

```bash
docker run -it --rm \
  --add-host=host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v "$(pwd):/app" \
  --name epllm \
  epllm:latest \
  python -m EPLLM \
  -f /app/path/to/snippet.py \
  --modules-dir /app/modules \
  -b http://host.docker.internal:11434 \
  -m phi3:medium
```

On Docker Desktop, `host.docker.internal` usually already works. On Linux,
keep the `--add-host=host.docker.internal:host-gateway` flag.

## OpenAI Models

To use a GPT model instead of Ollama:

```bash
docker run -it --rm \
  -e OPENAI_KEY=your_api_key \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v "$(pwd):/app" \
  --name epllm \
  epllm:latest \
  python -m EPLLM \
  -f /app/path/to/snippet.py \
  --modules-dir /app/modules \
  -m gpt-4.1-mini
```

## Notes

- `/app/modules` is a good place to persist the PyPI cache and
  `.epllm_success_memory.json`.
- The default container command is `tail -f /dev/null`, matching `pllm`, so you
  can keep the container alive and run commands inside it.
- If Docker access still fails, inspect `/var/run/docker.sock` on the host and
  confirm the mounted socket is reachable from the container.
