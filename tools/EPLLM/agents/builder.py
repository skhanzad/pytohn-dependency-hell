# Agent 3: Builds Docker image and runs container. Returns build_complete / run_complete and docker_output.
from typing import Optional
from langchain_core.runnables.config import RunnableConfig
from state import EPLLMState


def build_node(state: EPLLMState, config: Optional[RunnableConfig] = None) -> dict:
    cfg = (config or {}).get("configurable", {})
    docker_helper = cfg.get("docker_helper")
    ollama_helper = cfg.get("ollama_helper")
    executor = cfg.get("executor")
    file_path = state["file_path"]
    llm_eval = state.get("llm_eval")
    error_handler = state.get("error_handler") or _default_error_handler()

    if not docker_helper or not ollama_helper or not executor or not llm_eval or not file_path:
        return {"build_complete": False, "docker_output": "", "last_error_type": None, "last_fix_output": None}

    docker_helper.create_dockerfile(llm_eval, file_path)
    passed, docker_output = docker_helper.build_dockerfile(file_path)

    if passed:
        return {
            "build_complete": True,
            "docker_output": docker_output or "",
            "last_error_type": None,
            "last_fix_output": None,
        }

    output, error_type = ollama_helper.process_error(docker_output, error_handler, llm_eval)
    error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
    llm_eval = executor.update_llm_eval(output, llm_eval)

    # Ordering / PATH handling from pllm
    if error_type == "ImportError" and "returned a non-zero code: 1" in (docker_output or ""):
        zero_code_module = ollama_helper.non_zero_error(docker_output)
        move_module = zero_code_module.get("module") if isinstance(zero_code_module, dict) else zero_code_module
        if move_module and output:
            llm_eval = executor.shuffle_modules(output.get("module"), move_module, llm_eval)
    if error_type == "NonZeroCode" and output and "PATH environment" in (docker_output or ""):
        mod = output.get("module") if isinstance(output, dict) else None
        if mod and mod in llm_eval.get("python_modules", {}):
            llm_eval["python_modules"].pop(mod, None)

    return {
        "build_complete": False,
        "docker_output": docker_output or "",
        "last_error_type": error_type,
        "last_fix_output": output,
        "error_handler": error_handler,
        "llm_eval": llm_eval,
    }


def run_container_node(state: EPLLMState, config: Optional[RunnableConfig] = None) -> dict:
    cfg = (config or {}).get("configurable", {})
    docker_helper = cfg.get("docker_helper")
    ollama_helper = cfg.get("ollama_helper")
    executor = cfg.get("executor")
    llm_eval = state.get("llm_eval")
    error_handler = state.get("error_handler") or _default_error_handler()

    if not docker_helper or not ollama_helper or not executor:
        return {"run_complete": False, "docker_output": ""}

    docker_output = docker_helper.run_container_test()
    output, error_type = ollama_helper.process_error(docker_output, error_handler, llm_eval)

    if error_type == "None" or error_type is None:
        return {"run_complete": True, "docker_output": docker_output}

    error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
    llm_eval = executor.update_llm_eval(output, llm_eval)

    return {
        "run_complete": False,
        "docker_output": docker_output,
        "last_error_type": error_type,
        "last_fix_output": output,
        "error_handler": error_handler,
        "llm_eval": llm_eval,
    }


def _default_error_handler():
    return {
        "previous": "",
        "error_modules": {},
        "ImportError": 0,
        "ModuleNotFound": 0,
        "VersionNotFound": 0,
        "DependencyConflict": 0,
        "AttributeError": 0,
        "NonZeroCode": 0,
        "SyntaxError": 0,
    }
