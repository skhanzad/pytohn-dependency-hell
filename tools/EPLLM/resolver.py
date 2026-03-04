"""Core snippet resolver: deterministic first, LLM-assisted (adaptive RAG) fallback."""

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
from tools.EPLLM.version_selector import VersionSelector, FALLBACK_PACKAGES


class SnippetResolver:
    """Resolves dependencies for a single Python snippet with adaptive RAG."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "gemma2", temp: float = 0.2,
                 max_iterations: int = 8, modules_dir: str = "./modules",
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

        # LLM helper — used for initial analysis AND adaptive RAG error recovery
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
            self._filter_bad_modules(state)
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
                # Try compatibility-aware pick first, then latest
                pick = VersionSelector.select(
                    module, versions, [], 0,
                    resolved_modules=resolved
                )
                resolved[module] = pick if pick else versions[-1]
            else:
                resolved[module] = "0.0.0"

        state["resolved_modules"] = resolved

        if self.logging:
            print(f"[EPLLM] Resolved versions: {resolved}")

    def _filter_bad_modules(self, state: SnippetState):
        """Remove modules that can't be pip-installed before even trying."""
        to_remove = []
        for module, version in state["resolved_modules"].items():
            # Skip system / non-pip packages
            if self.error_parser.is_system_package(module):
                to_remove.append(module)
                continue
            # Skip modules with 0.0.0 placeholder — they aren't real pip packages
            if version == "0.0.0":
                # Check if there's a fallback package name
                fallback = VersionSelector.get_fallback_package(module)
                if fallback and fallback != module:
                    # Try the fallback package
                    versions = self._get_version_list(fallback, state["python_version"])
                    if not versions:
                        self.pypi.read_module_file(fallback, state["python_version"])
                        versions = self._get_version_list(fallback, state["python_version"])
                    if versions:
                        pick = VersionSelector.pick_latest(versions, [])
                        if pick:
                            state["resolved_modules"][fallback] = pick
                to_remove.append(module)

        for module in to_remove:
            if module in state["resolved_modules"]:
                del state["resolved_modules"][module]
                if self.logging:
                    print(f"[EPLLM] Filtered out non-pip module: {module}")

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

    # --- Phase 3: Build/run iteration loop with adaptive RAG ---

    def _iteration_loop(self, state: SnippetState):
        """Loop: build → handle error → run → handle error, up to max_iterations.
        Uses deterministic handling first, then LLM-assisted (adaptive RAG) fallback."""
        docker = DockerHelper(logging=self.logging)
        state["status"] = "building"

        consecutive_deterministic_failures = 0

        for i in range(state["max_iterations"]):
            state["iteration"] = i

            # Build
            build_ok = self._docker_build(state, docker)
            if not build_ok:
                # Try deterministic first
                handled = self._handle_error(state, state["build_log"], docker)
                if not handled:
                    consecutive_deterministic_failures += 1
                    # Adaptive RAG: use LLM when deterministic fails
                    if consecutive_deterministic_failures >= 1:
                        handled = self._llm_error_recovery(state, state["build_log"])
                    if not handled:
                        state["status"] = "failed"
                        self._cleanup(docker)
                        return
                else:
                    consecutive_deterministic_failures = 0
                continue

            # Run
            state["build_passed"] = True
            consecutive_deterministic_failures = 0
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
                    # Adaptive RAG fallback for run errors too
                    handled = self._llm_error_recovery(state, state["run_log"])
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

    # --- Deterministic error handling ---

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

        # Module failed too many times — try fallback package or remove
        if module and len(failed.get(module, [])) >= 4:
            # Try fallback package name first
            fallback = VersionSelector.get_fallback_package(module)
            if fallback and fallback != module and fallback not in resolved:
                versions = self._get_version_list(fallback, py_ver)
                if not versions:
                    self.pypi.read_module_file(fallback, py_ver)
                    versions = self._get_version_list(fallback, py_ver)
                if versions:
                    pick = VersionSelector.pick_latest(versions, [])
                    if pick:
                        del resolved[module]
                        resolved[fallback] = pick
                        if self.logging:
                            print(f"[EPLLM] Swapped {module} -> {fallback}=={pick}")
                        return True

            if module in resolved:
                if self.logging:
                    print(f"[EPLLM] Removing {module} after {len(failed[module])} failures")
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
            state["iteration"], available,
            resolved_modules=state["resolved_modules"]
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

        if not versions:
            # Module isn't on PyPI — skip it
            if self.logging:
                print(f"[EPLLM] No PyPI versions for {mod_name}, skipping")
            return True

        failed_list = state["failed_versions"].get(mod_name, [])
        pick = VersionSelector.select(
            mod_name, versions, failed_list, state["iteration"],
            resolved_modules=state["resolved_modules"]
        )

        if pick and pick != "0.0.0":
            state["resolved_modules"][mod_name] = pick
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
            self._filter_bad_modules(state)
            return True

        # Already on 2.7 — try to identify the module from the error
        mod_match = re.search(r"site-packages/(\w+)", log)
        if mod_match:
            module = mod_match.group(1)
            failed = state["failed_versions"].get(module, [])
            versions = self._get_version_list(module, state["python_version"])
            pick = VersionSelector.select(
                module, versions, failed, state["iteration"],
                resolved_modules=state["resolved_modules"]
            )
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
                module, versions, failed_list, state["iteration"],
                resolved_modules=state["resolved_modules"]
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
            # Try to find the module from the traceback
            module = self.error_parser.extract_traceback_module(
                state.get("run_log", "") or state.get("build_log", ""),
                state["resolved_modules"]
            )
            if not module:
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
        """Handle dependency conflicts by trying compatible versions."""
        if not module or module not in state["resolved_modules"]:
            return False

        failed_list = state["failed_versions"].get(module, [])
        versions = self._get_version_list(module, state["python_version"])

        # Try compatibility-aware selection
        pick = VersionSelector.select(
            module, versions, failed_list, state["iteration"],
            resolved_modules=state["resolved_modules"]
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

    # --- Adaptive RAG: LLM-assisted error recovery ---

    def _llm_error_recovery(self, state: SnippetState, log: str) -> bool:
        """Use LLM to analyze error and suggest fix when deterministic handling fails.

        This is the 'adaptive' part of adaptive RAG: the LLM gets the error context
        plus available version lists (retrieved from cache) to make an informed decision.
        """
        if self.logging:
            print(f"[EPLLM] Adaptive RAG: consulting LLM for error recovery "
                  f"(iter {state['iteration']})")

        # Build the context for LLM — this is the RAG retrieval step
        error_details = {
            "error_modules": {}
        }
        for mod, vers in state["failed_versions"].items():
            error_details["error_modules"][mod] = vers

        llm_eval = {
            "python_version": state["python_version"],
            "python_modules": state["resolved_modules"],
        }

        try:
            # Use the OllamaHelper's process_error which routes to
            # specialized LLM prompts for each error type
            output, error_type = self.llm.process_error(
                log[:3000],  # Truncate long logs
                error_details,
                llm_eval
            )

            if self.logging:
                print(f"[EPLLM] LLM suggested: type={error_type}, output={output}")

            if output and isinstance(output, dict):
                mod = output.get("module")
                ver = output.get("version")

                if mod and ver and ver != "None" and ver is not None:
                    # Validate the version string
                    ver_str = str(ver).strip()
                    if len(ver_str) <= 25 and any(c.isdigit() for c in ver_str):
                        # Check if this module name needs resolution
                        canonical = self.pypi.check_module_name(mod)
                        mod_name = canonical[0] if canonical else mod

                        # Skip system packages
                        if self.error_parser.is_system_package(mod_name):
                            if mod_name in state["resolved_modules"]:
                                del state["resolved_modules"][mod_name]
                            return True

                        state["resolved_modules"][mod_name] = ver_str
                        if self.logging:
                            print(f"[EPLLM] LLM fix: {mod_name}=={ver_str}")
                        return True

                # LLM identified the module but couldn't find a version
                if mod:
                    canonical = self.pypi.check_module_name(mod)
                    mod_name = canonical[0] if canonical else mod
                    if mod_name in state["resolved_modules"]:
                        del state["resolved_modules"][mod_name]
                        if self.logging:
                            print(f"[EPLLM] LLM says remove: {mod_name}")
                        return True

            # If LLM returned a string (from non_zero_error), it's a module name
            if output and isinstance(output, str):
                mod_name = output
                if mod_name in state["resolved_modules"]:
                    del state["resolved_modules"][mod_name]
                    return True

        except Exception as e:
            if self.logging:
                print(f"[EPLLM] LLM error recovery failed: {e}")

        return False

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
