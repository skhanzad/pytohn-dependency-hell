"""Core snippet resolver: one LLM call for analysis, then deterministic error handling."""

import os
import sys
import re
import time
from pathlib import Path
from typing import Optional

# Add pllm helpers to path
_PLLM_DIR = Path(__file__).resolve().parent.parent / "pllm"
sys.path.insert(0, str(_PLLM_DIR))

from helpers.ollama_helper_tester import OllamaHelper
from helpers.py_pi_query import PyPIQuery
from helpers.build_dockerfile import DockerHelper
from helpers.deps_scraper import DepsScraper

from tools.EPLLM.state import SnippetState
from tools.EPLLM.error_parser import ErrorParser
from tools.EPLLM.version_selector import VersionSelector


class SnippetResolver:
    """Resolves dependencies for a single Python snippet."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "gemma2", temp: float = 0.2,
                 max_iterations: int = 5, modules_dir: str = "./modules",
                 logging: bool = False):
        self.base_url = base_url
        self.model_name = model
        self.temp = temp
        self.max_iterations = max_iterations
        self.modules_dir = modules_dir
        self.logging = logging

        os.makedirs(modules_dir, exist_ok=True)

        self.error_parser = ErrorParser()
        self.pypi = PyPIQuery(logging=logging, base_modules=modules_dir)
        self.deps = DepsScraper(logging=logging)

        # LLM helper — only used for evaluate_file()
        self.llm = OllamaHelper(
            base_url=base_url, model=model, temp=temp,
            logging=logging, base_modules=modules_dir, rag=True
        )

    def resolve(self, file_path: str) -> SnippetState:
        """Main entry point. Returns final state dict."""
        split = file_path.split("/")
        snippet_name = split[-2] if len(split) >= 2 else "unknown"

        state: SnippetState = {
            "file_path": file_path,
            "snippet_name": snippet_name,
            "python_version": "3.8",
            "imports": [],
            "resolved_modules": {},
            "failed_versions": {},
            "build_passed": False,
            "run_passed": False,
            "build_log": "",
            "run_log": "",
            "error_type": None,
            "error_module": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "pending",
        }

        try:
            self._analyze(state)
            self._resolve_versions(state)
            self._iteration_loop(state)
        except Exception as e:
            if self.logging:
                print(f"[EPLLM] resolve error for {snippet_name}: {e}")
            state["status"] = "failed"
            state["error_type"] = str(e)

        return state

    # --- Phase 1: Analysis (single LLM call) ---

    def _analyze(self, state: SnippetState):
        """Detect Python version and imports. One LLM call + regex scraping."""
        file_path = state["file_path"]

        # Read source for heuristic checks
        try:
            with open(file_path, "r") as f:
                source = f.read()
        except Exception:
            source = ""

        # Python 2 heuristic before LLM
        is_py2 = self.error_parser.detect_python2(source)

        # Regex-based import scraping
        scraped = self.deps.find_word_in_file(file_path, "import", [])

        # Single LLM call
        llm_eval = self.llm.evaluate_file(file_path)

        if llm_eval:
            py_ver = str(llm_eval.get("python_version", "3.8"))
            llm_modules = llm_eval.get("python_modules", [])
            if isinstance(llm_modules, dict):
                llm_modules = list(llm_modules.keys())
        else:
            py_ver = "3.8"
            llm_modules = []

        # Override version if heuristic strongly suggests Python 2
        if is_py2:
            py_ver = "2.7"

        # Merge LLM modules with scraped imports, deduplicate
        combined = list(dict.fromkeys(llm_modules + scraped))

        # Clean through PyPI name resolution
        cleaned = self.pypi.check_module_name(combined)

        state["python_version"] = self.pypi.check_format(py_ver)
        state["imports"] = cleaned

        if self.logging:
            print(f"[EPLLM] Analyzed {state['snippet_name']}: "
                  f"py={state['python_version']}, modules={cleaned}")

    # --- Phase 2: Deterministic version resolution (0 LLM calls) ---

    def _resolve_versions(self, state: SnippetState):
        """For each module, read cached version file or query PyPI, pick latest."""
        modules = state["imports"]
        py_ver = state["python_version"]

        # First, generate version files via PyPI query
        module_details = {"python_version": py_ver, "python_modules": modules}
        resolved_modules, py_ver = self.pypi.get_module_specifics(module_details)
        state["python_version"] = py_ver

        resolved = {}
        for module in resolved_modules:
            version_str = self._read_versions_file(module, py_ver)
            if not version_str or not version_str.strip():
                # No versions available — skip or use placeholder
                resolved[module] = "0.0.0"
                continue

            versions = [v.strip() for v in version_str.split(",") if v.strip()]
            if versions:
                # Pick latest deterministically
                pick = VersionSelector.pick_latest(versions, [])
                resolved[module] = pick if pick else versions[-1]
            else:
                resolved[module] = "0.0.0"

        state["resolved_modules"] = resolved

        if self.logging:
            print(f"[EPLLM] Resolved versions: {resolved}")

    def _read_versions_file(self, module: str, py_ver: str) -> str:
        """Read module versions from cache file."""
        fpath = f"{self.modules_dir}/{module}_{py_ver}.txt"
        if os.path.isfile(fpath):
            with open(fpath, "r") as f:
                return f.read()
        return ""

    def _get_version_list(self, module: str, py_ver: str) -> list:
        """Get sorted list of versions for a module."""
        content = self._read_versions_file(module, py_ver)
        if not content or not content.strip():
            # Try to generate it
            content = self.pypi.read_module_file(module, py_ver)
        if not content or not content.strip():
            return []
        return [v.strip() for v in content.split(",") if v.strip()]

    # --- Phase 3: Build/run iteration loop ---

    def _iteration_loop(self, state: SnippetState):
        """Loop: build → handle error → run → handle error, up to max_iterations."""
        docker = DockerHelper(logging=self.logging)
        state["status"] = "building"

        for i in range(state["max_iterations"]):
            state["iteration"] = i

            # Build
            build_ok = self._docker_build(state, docker)
            if not build_ok:
                handled = self._handle_error(state, state["build_log"], docker)
                if not handled:
                    state["status"] = "failed"
                    self._cleanup(docker)
                    return
                continue

            # Run
            state["build_passed"] = True
            run_ok = self._docker_run(state, docker)
            if run_ok:
                state["run_passed"] = True
                state["status"] = "success"
                self._cleanup(docker)
                return
            else:
                # Handle run error
                handled = self._handle_error(state, state["run_log"], docker)
                if not handled:
                    state["status"] = "failed"
                    self._cleanup(docker)
                    return

        state["status"] = "failed"
        self._cleanup(docker)

    def _docker_build(self, state: SnippetState, docker: DockerHelper) -> bool:
        """Create Dockerfile and build. Returns True if build succeeds."""
        llm_out = {
            "python_version": state["python_version"],
            "python_modules": state["resolved_modules"],
        }
        docker.create_dockerfile(llm_out, state["file_path"])
        passed, error_log = docker.build_dockerfile(state["file_path"])
        state["build_log"] = error_log
        state["build_passed"] = passed
        if self.logging and not passed:
            print(f"[EPLLM] Build failed (iter {state['iteration']}): "
                  f"{error_log[:200]}")
        return passed

    def _docker_run(self, state: SnippetState, docker: DockerHelper) -> bool:
        """Run the container and check output."""
        try:
            output = docker.run_container_test()
        except Exception as e:
            output = str(e)

        state["run_log"] = output

        # Check for errors in run output
        error_type, _, _ = self.error_parser.parse_error(output)
        if error_type == "None":
            return True
        return False

    def _handle_error(self, state: SnippetState, log: str,
                      docker: DockerHelper) -> bool:
        """Deterministic error handling. Returns True if state was updated and we should retry."""
        error_type, module, version = self.error_parser.parse_error(log)
        state["error_type"] = error_type
        state["error_module"] = module

        if error_type == "None":
            return False

        py_ver = state["python_version"]
        resolved = state["resolved_modules"]
        failed = state["failed_versions"]

        # Record failed version
        if module and version:
            failed.setdefault(module, [])
            if version not in failed[module]:
                failed[module].append(version)

        # Also record current resolved version as failed
        if module and module in resolved:
            cur_ver = resolved[module]
            failed.setdefault(module, [])
            if cur_ver not in failed[module]:
                failed[module].append(cur_ver)

        # Module failed too many times — remove it
        if module and len(failed.get(module, [])) >= 3:
            if module in resolved:
                if self.logging:
                    print(f"[EPLLM] Removing {module} after 3 failures")
                del resolved[module]
                return True

        if error_type == "VersionNotFound":
            return self._handle_version_not_found(state, log, module)

        elif error_type in ("ModuleNotFound", "ImportError"):
            return self._handle_missing_module(state, module)

        elif error_type == "SyntaxError":
            return self._handle_syntax_error(state, log)

        elif error_type == "NonZeroCode":
            return self._handle_non_zero(state, module)

        elif error_type == "AttributeError":
            return self._handle_attribute_error(state, module)

        elif error_type == "DependencyConflict":
            return self._handle_dependency_conflict(state, module)

        elif error_type == "InvalidVersion":
            return self._handle_invalid_version(state, module)

        return False

    def _handle_version_not_found(self, state: SnippetState, log: str,
                                  module: Optional[str]) -> bool:
        """Parse available versions from error, pick next via VersionSelector."""
        if not module:
            return False

        available = self.error_parser.extract_available_versions(log)
        failed_list = state["failed_versions"].get(module, [])

        # Also try cached versions
        cached = self._get_version_list(module, state["python_version"])
        all_versions = available if available else cached

        pick = VersionSelector.select(
            module, all_versions, failed_list,
            state["iteration"], available
        )

        if pick:
            state["resolved_modules"][module] = pick
            if self.logging:
                print(f"[EPLLM] VersionNotFound: {module} -> {pick}")
            return True

        # No version found — remove the module
        if module in state["resolved_modules"]:
            del state["resolved_modules"][module]
            return True

        return False

    def _handle_missing_module(self, state: SnippetState,
                               module: Optional[str]) -> bool:
        """Add missing module, query PyPI, pick latest."""
        if not module:
            return False

        # Resolve canonical name
        canonical = self.pypi.check_module_name(module)
        mod_name = canonical[0] if canonical else module

        # Skip system packages
        if self.error_parser.is_system_package(mod_name):
            if self.logging:
                print(f"[EPLLM] Skipping system package: {mod_name}")
            return True

        # Query versions and pick latest
        versions = self._get_version_list(mod_name, state["python_version"])
        if not versions:
            # Try to generate
            self.pypi.read_module_file(mod_name, state["python_version"])
            versions = self._get_version_list(mod_name, state["python_version"])

        failed_list = state["failed_versions"].get(mod_name, [])
        pick = VersionSelector.pick_latest(versions, failed_list) if versions else "0.0.0"

        state["resolved_modules"][mod_name] = pick if pick else "0.0.0"
        if self.logging:
            print(f"[EPLLM] Added missing module: {mod_name}=={pick}")
        return True

    def _handle_syntax_error(self, state: SnippetState, log: str) -> bool:
        """If syntax error looks like Python 2 issue, switch to 2.7."""
        if state["python_version"] != "2.7":
            if self.logging:
                print("[EPLLM] SyntaxError: switching to Python 2.7")
            state["python_version"] = "2.7"
            # Re-resolve all module versions for Python 2.7
            self._resolve_versions(state)
            return True

        # Already on 2.7 — try to identify the module from the error
        # Look for module path in traceback
        mod_match = re.search(
            r"site-packages/(\w+)", log
        )
        if mod_match:
            module = mod_match.group(1)
            failed = state["failed_versions"].get(module, [])
            versions = self._get_version_list(module, state["python_version"])
            pick = VersionSelector.select(module, versions, failed, state["iteration"])
            if pick and module in state["resolved_modules"]:
                state["resolved_modules"][module] = pick
                return True

        return False

    def _handle_non_zero(self, state: SnippetState,
                         module: Optional[str]) -> bool:
        """Try removing the failing module or picking a different version."""
        if not module:
            return False

        failed_list = state["failed_versions"].get(module, [])
        versions = self._get_version_list(module, state["python_version"])

        if versions:
            pick = VersionSelector.select(
                module, versions, failed_list, state["iteration"]
            )
            if pick:
                state["resolved_modules"][module] = pick
                return True

        # Can't find a working version — remove
        if module in state["resolved_modules"]:
            del state["resolved_modules"][module]
            if self.logging:
                print(f"[EPLLM] NonZeroCode: removing {module}")
            return True

        return False

    def _handle_attribute_error(self, state: SnippetState,
                                module: Optional[str]) -> bool:
        """Try older version of identified module."""
        if not module:
            # Try to match against resolved modules
            return False

        # If module isn't in resolved, check if it's a known alias
        if module not in state["resolved_modules"]:
            canonical = self.pypi.check_module_name(module)
            mod_name = canonical[0] if canonical else module
            # Check if canonical name is in resolved
            if mod_name in state["resolved_modules"]:
                module = mod_name
            else:
                return False

        failed_list = state["failed_versions"].get(module, [])
        versions = self._get_version_list(module, state["python_version"])

        # For AttributeError, prefer older versions
        pick = VersionSelector.pick_by_binary_search(
            versions, failed_list, prefer_newer=False
        )
        if pick:
            state["resolved_modules"][module] = pick
            if self.logging:
                print(f"[EPLLM] AttributeError: {module} -> {pick}")
            return True

        return False

    def _handle_dependency_conflict(self, state: SnippetState,
                                    module: Optional[str]) -> bool:
        """Handle dependency conflicts by trying a different version."""
        if not module or module not in state["resolved_modules"]:
            return False

        failed_list = state["failed_versions"].get(module, [])
        versions = self._get_version_list(module, state["python_version"])

        pick = VersionSelector.select(
            module, versions, failed_list, state["iteration"]
        )
        if pick:
            state["resolved_modules"][module] = pick
            return True

        return False

    def _handle_invalid_version(self, state: SnippetState,
                                module: Optional[str]) -> bool:
        """Handle InvalidVersion by picking a proper version."""
        if not module:
            return False

        failed_list = state["failed_versions"].get(module, [])
        versions = self._get_version_list(module, state["python_version"])

        pick = VersionSelector.pick_latest(versions, failed_list)
        if pick:
            state["resolved_modules"][module] = pick
            return True

        # Remove if no valid version found
        if module in state["resolved_modules"]:
            del state["resolved_modules"][module]
        return True

    # --- Cleanup ---

    def _cleanup(self, docker: DockerHelper):
        """Delete container and image."""
        try:
            docker.delete_container()
        except Exception:
            pass
        try:
            docker.delete_image()
        except Exception:
            pass
