"""Core resolution pipeline for a single Python snippet.

This is the heart of the EPLLM hybrid approach, combining:
1. PyEGo's AST-based static analysis for import extraction + Python version detection
2. PyPI querying with deterministic version selection strategies
3. PLLM's Docker-based iterative validation loop
4. Regex-based error parsing with targeted recovery
5. LLM as strategic fallback for ambiguous errors

Resolution flow:
  Snippet -> AST parse -> detect Python version -> map imports to packages
  -> query PyPI versions -> select initial versions -> Docker build/test
  -> if error: parse error + apply fix -> iterate
  -> if Python version error: try next version
"""
import os
import time

from EPLLM.state import ResolveResult  # noqa: E402
from EPLLM.memory import SuccessfulVersionMemory  # noqa: E402
from EPLLM.import_analyzer import (  # noqa: E402
    analyze_imports, detect_python_versions, map_import_to_package, is_stdlib
)
from EPLLM.pypi_client import PyPIClient  # noqa: E402
from EPLLM.version_selector import (  # noqa: E402
    select_version, pick_alternative_version, get_strategy_for_iteration
)
from EPLLM.error_parser import parse_error, is_python_version_error  # noqa: E402


class SnippetResolver:
    """Resolves dependencies for a single Python snippet."""

    def __init__(self, base_modules='./modules', max_iterations=7,
                 llm_base_url="http://localhost:11434", llm_model="phi3:medium",
                 llm_temp=0.3, use_llm=True, logging=False):
        self.pypi = PyPIClient(cache_dir=base_modules, logging=logging)
        self.success_memory = SuccessfulVersionMemory(
            cache_dir=base_modules, logging=logging
        )
        self.max_iterations = max_iterations
        self.logging = logging
        self.use_llm = bool(use_llm)
        self.llm = None

        if self.use_llm:
            try:
                from EPLLM.llm_client import LLMClient  # noqa: E402

                self.llm = LLMClient(
                    base_url=llm_base_url,
                    model=llm_model,
                    temperature=llm_temp,
                )
            except Exception as exc:
                self.use_llm = False
                if self.logging:
                    print(f"  LangGraph fallback disabled: {exc}")

    def resolve(self, snippet_path):
        """Resolve dependencies for a Python snippet.

        Returns a ResolveResult with success/failure, resolved modules,
        Python version, and metadata.
        """
        start_time = time.time()
        snippet_path = os.path.abspath(snippet_path)

        if not os.path.isfile(snippet_path):
            return ResolveResult(
                success=False, error_type='FileNotFound',
                error_message=f'File not found: {snippet_path}',
                duration=time.time() - start_time
            )

        # ── Phase 1: Static Analysis (PyEGo-inspired) ───────────────────
        if self.logging:
            print(f"\n{'='*60}")
            print(f"Resolving: {snippet_path}")

        # Extract imports using AST
        packages = analyze_imports(snippet_path)
        if self.logging:
            print(f"  Detected packages: {packages}")

        # Detect Python version from syntax features
        python_versions = detect_python_versions(snippet_path)
        if self.logging:
            print(f"  Python version candidates: {python_versions}")

        if not packages:
            # No third-party imports found
            return ResolveResult(
                success=True, python_version=python_versions[0],
                modules={}, iterations=0,
                duration=time.time() - start_time
            )

        # ── Phase 2: Try each Python version ────────────────────────────
        last_result = None
        total_iterations = 0

        for py_ver in python_versions:
            result = self._try_python_version(
                snippet_path, packages, py_ver, start_time
            )
            total_iterations += result.iterations

            if result.success:
                result.iterations = total_iterations
                return result

            last_result = result

            if self.logging:
                print(f"  Python {py_ver} failed: {result.error_type}")

        # All versions exhausted - return last attempt's result
        if last_result:
            last_result.iterations = total_iterations
            last_result.duration = time.time() - start_time
            return last_result

        return ResolveResult(
            success=False, python_version=python_versions[0],
            error_type='NoVersions', error_message='All versions exhausted',
            iterations=total_iterations,
            duration=time.time() - start_time
        )

    def _try_python_version(self, snippet_path, packages, python_version, start_time):
        """Try to resolve dependencies for a specific Python version."""
        if self.logging:
            print(f"\n  Trying Python {python_version}...")

        # ── Phase 3: Fetch available versions from PyPI ──────────────────
        version_cache = {}  # {package: [versions]}
        valid_packages = []

        for pkg in packages:
            versions = self.pypi.get_versions(pkg, python_version)
            if versions:
                version_cache[pkg] = versions
                valid_packages.append(pkg)
            elif self.logging:
                print(f"    No versions found for {pkg} on Python {python_version}")

        if not valid_packages:
            return ResolveResult(
                success=False, python_version=python_version,
                error_type='NoVersions',
                error_message='No packages found on PyPI',
                duration=time.time() - start_time
            )

        # ── Phase 4: Select initial versions ─────────────────────────────
        modules = {}  # {package: version}
        for pkg in valid_packages:
            ver = self._choose_version(
                package=pkg,
                versions=version_cache[pkg],
                python_version=python_version,
                iteration=0,
            )
            if ver:
                modules[pkg] = ver

        if self.logging:
            print(f"    Initial versions: {modules}")

        # ── Phase 5: Iterative Docker validation ─────────────────────────
        failed_versions = {}  # {package: set(failed_versions)}
        from EPLLM.docker_tester import DockerTester  # noqa: E402
        docker = DockerTester(logging=self.logging)

        try:
            for iteration in range(self.max_iterations):
                if self.logging:
                    print(f"\n    Iteration {iteration + 1}/{self.max_iterations}")
                    print(f"    Modules: {modules}")

                # Build Dockerfile and Docker image
                docker.create_dockerfile(snippet_path, python_version, modules)
                build_ok, build_output = docker.build(snippet_path)

                if build_ok:
                    # Build succeeded - run the container
                    run_ok, run_output = docker.run(timeout=60)

                    if run_ok:
                        # SUCCESS!
                        if self.logging:
                            print(f"    SUCCESS on iteration {iteration + 1}")
                        self.success_memory.remember_resolution(
                            python_version, modules
                        )
                        self._log_result(snippet_path, python_version, modules,
                                         'NoError', '', iteration + 1, start_time, True)
                        return ResolveResult(
                            success=True, python_version=python_version,
                            modules=dict(modules), iterations=iteration + 1,
                            duration=time.time() - start_time
                        )

                    # Runtime error - parse and fix
                    error = parse_error(run_output)
                    if self.logging:
                        print(f"    Runtime error: {error}")
                else:
                    # Build error - parse and fix
                    error = parse_error(build_output)
                    if self.logging:
                        print(f"    Build error: {error}")

                # ── Phase 6: Error Recovery ──────────────────────────────
                if is_python_version_error(error):
                    # Need different Python version
                    self._log_result(snippet_path, python_version, modules,
                                     error.error_type, error.message,
                                     iteration + 1, start_time, False)
                    return ResolveResult(
                        success=False, python_version=python_version,
                        modules=dict(modules), error_type=error.error_type,
                        error_message='Python version mismatch',
                        iterations=iteration + 1,
                        duration=time.time() - start_time
                    )

                fixed = self._apply_fix(
                    error, modules, version_cache, failed_versions,
                    python_version, iteration
                )

                if not fixed:
                    if self.logging:
                        print(f"    No fix available, stopping")
                    self._log_result(snippet_path, python_version, modules,
                                     error.error_type, error.message,
                                     iteration + 1, start_time, False)
                    break

                self._log_result(snippet_path, python_version, modules,
                                 error.error_type, error.message,
                                 iteration + 1, start_time, False)

        finally:
            docker.cleanup()

        return ResolveResult(
            success=False, python_version=python_version,
            modules=dict(modules), error_type=error.error_type if 'error' in dir() else 'MaxIterations',
            error_message=error.message if 'error' in dir() else 'Max iterations reached',
            iterations=min(iteration + 1, self.max_iterations) if 'iteration' in dir() else 0,
            duration=time.time() - start_time
        )

    def _apply_fix(self, error, modules, version_cache, failed_versions,
                   python_version, iteration):
        """Apply a targeted fix based on the error type.

        Returns True if a fix was applied, False if no fix is possible.
        """
        etype = error.error_type

        if etype == 'VersionNotFound':
            return self._fix_version_not_found(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype == 'DependencyConflict':
            return self._fix_dependency_conflict(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype == 'ModuleNotFound':
            return self._fix_module_not_found(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype == 'ImportError':
            return self._fix_import_error(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype == 'AttributeError':
            return self._fix_attribute_error(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype == 'NonZeroCode':
            return self._fix_non_zero_code(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype == 'SyntaxError':
            return self._fix_syntax_error(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype == 'InvalidVersion':
            return self._fix_invalid_version(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        elif etype in ('NameError', 'TypeError'):
            # These are usually code issues, not dependency issues
            # But sometimes a different module version helps
            return self._fix_generic_version(
                error, modules, version_cache, failed_versions, python_version, iteration
            )

        return False

    def _get_package_versions(self, package, python_version, version_cache):
        """Get versions from local cache or PyPI."""
        versions = version_cache.get(package, [])
        if versions:
            return versions

        versions = self.pypi.get_versions(package, python_version)
        if versions:
            version_cache[package] = versions
        return versions

    def _choose_version(self, package, versions, python_version, iteration=0,
                        excluded=None, error=None, current_version=None,
                        prefer_older=False, allow_llm=True):
        """Choose a version using success memory, determinism, then LLM."""
        versions = list(versions or [])
        excluded = set(excluded or [])

        if not versions:
            return None

        remembered = self.success_memory.get_preferred_version(
            package=package,
            python_version=python_version,
            available_versions=versions,
            excluded=excluded,
            current_version=current_version,
            prefer_older=prefer_older,
        )
        if remembered:
            if self.logging:
                print(f"    Using remembered version for {package}: {remembered}")
            return remembered

        if current_version:
            candidate = pick_alternative_version(
                versions,
                current_version,
                excluded,
                prefer_older=prefer_older,
            )
        else:
            candidate = select_version(
                versions,
                get_strategy_for_iteration(iteration, error),
                excluded,
                error,
            )

        if candidate:
            if self.logging:
                print(f"    Using deterministic version for {package}: {candidate}")
            return candidate

        if allow_llm and self.use_llm and self.llm:
            llm_version = self.llm.suggest_version(
                package,
                ', '.join(versions),
                python_version,
                excluded_versions=', '.join(sorted(excluded)),
                error_context=error.message if error else '',
            )
            valid_llm_versions = set(versions)
            if error and error.available_versions:
                valid_llm_versions.update(error.available_versions)
            if llm_version and llm_version not in excluded and llm_version in valid_llm_versions:
                if self.logging:
                    print(f"    Using LangGraph fallback for {package}: {llm_version}")
                return llm_version

        return None

    # ── Fix implementations ──────────────────────────────────────────────

    def _fix_version_not_found(self, error, modules, version_cache,
                               failed_versions, python_version, iteration):
        """Fix: pick a different version from the available list."""
        module = error.module
        if not module:
            return False

        # Track failed version
        if module in modules:
            failed_versions.setdefault(module, set()).add(modules[module])

        # If error contains available versions, use those (pip's version list
        # is filtered to the current Python version - much more accurate)
        if error.available_versions:
            version_cache[module] = error.available_versions
            # Also update disk cache with the Python-version-specific list
            try:
                cache_path = self.pypi._cache_path(module, python_version)
                with open(cache_path, 'w') as f:
                    f.write(', '.join(error.available_versions))
            except (IOError, OSError):
                pass

        versions = self._get_package_versions(module, python_version, version_cache)

        if not versions:
            # Remove the package entirely
            modules.pop(module, None)
            return True

        excluded = failed_versions.get(module, set())
        new_ver = self._choose_version(
            package=module,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
        )

        if new_ver:
            modules[module] = new_ver
            return True

        # Remove the package
        modules.pop(module, None)
        return True

    def _fix_dependency_conflict(self, error, modules, version_cache,
                                 failed_versions, python_version, iteration):
        """Fix: adjust the conflicting package's version."""
        module = error.module
        if not module:
            # Try to find the conflicting module in the error message
            # If we can't, try adjusting all modules
            for pkg in list(modules.keys()):
                if pkg in error.message:
                    module = pkg
                    break

        if not module:
            # Can't identify the conflicting module
            # Try downgrading the most recently added package
            if modules:
                module = list(modules.keys())[-1]
            else:
                return False

        # Track failed version
        if module in modules:
            failed_versions.setdefault(module, set()).add(modules[module])

        versions = self._get_package_versions(module, python_version, version_cache)
        excluded = failed_versions.get(module, set())

        new_ver = self._choose_version(
            package=module,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
            current_version=modules.get(module, ''),
            prefer_older=True,
        )

        if new_ver:
            modules[module] = new_ver
            return True

        # If the conflicting module was required by another module,
        # try adjusting the requiring module instead
        if error.required_by and error.required_by in modules:
            req_by = error.required_by
            failed_versions.setdefault(req_by, set()).add(modules[req_by])
            req_versions = self._get_package_versions(
                req_by, python_version, version_cache
            )
            req_excluded = failed_versions.get(req_by, set())
            new_ver = self._choose_version(
                package=req_by,
                versions=req_versions,
                python_version=python_version,
                iteration=iteration,
                excluded=req_excluded,
                error=error,
                current_version=modules[req_by],
                prefer_older=True,
            )
            if new_ver:
                modules[req_by] = new_ver
                return True

        return False

    def _fix_module_not_found(self, error, modules, version_cache,
                              failed_versions, python_version, iteration):
        """Fix: add the missing module as a new package."""
        module = error.module
        if not module:
            return False

        # Check if this is a stdlib module we shouldn't install
        if is_stdlib(module):
            return False

        # Map to package name
        pkg = map_import_to_package(module)
        if not pkg:
            pkg = module.lower()

        # Check if already installed (might be a submodule issue)
        if pkg in modules:
            # Try a different version
            failed_versions.setdefault(pkg, set()).add(modules[pkg])
            versions = self._get_package_versions(pkg, python_version, version_cache)
            excluded = failed_versions.get(pkg, set())
            new_ver = self._choose_version(
                package=pkg,
                versions=versions,
                python_version=python_version,
                iteration=iteration,
                excluded=excluded,
                error=error,
                current_version=modules[pkg],
                prefer_older=True,
            )
            if new_ver:
                modules[pkg] = new_ver
                return True
            return False

        # Add new package
        versions = self._get_package_versions(pkg, python_version, version_cache)
        if not versions:
            # Try resolving the package name
            resolved = self.pypi.resolve_package_name(module)
            if resolved != pkg:
                versions = self._get_package_versions(
                    resolved, python_version, version_cache
                )
                if versions:
                    pkg = resolved

        if versions:
            new_ver = self._choose_version(
                package=pkg,
                versions=versions,
                python_version=python_version,
                iteration=iteration,
                error=error,
            )
            if new_ver:
                modules[pkg] = new_ver
                return True

        return False

    def _fix_import_error(self, error, modules, version_cache,
                          failed_versions, python_version, iteration):
        """Fix: try a different version of the module causing ImportError."""
        module = error.module
        if not module:
            return False

        # Map to package name
        pkg = map_import_to_package(module)
        if not pkg:
            pkg = module.lower()

        if pkg not in modules:
            # Module not installed - add it
            return self._fix_module_not_found(error, modules, version_cache,
                                              failed_versions, python_version, iteration)

        # Track failed version
        failed_versions.setdefault(pkg, set()).add(modules[pkg])
        versions = self._get_package_versions(pkg, python_version, version_cache)
        excluded = failed_versions.get(pkg, set())

        new_ver = self._choose_version(
            package=pkg,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
            current_version=modules[pkg],
            prefer_older=(iteration >= 2),
        )

        if new_ver:
            modules[pkg] = new_ver
            return True

        return False

    def _fix_attribute_error(self, error, modules, version_cache,
                             failed_versions, python_version, iteration):
        """Fix: try a different version of the module with AttributeError."""
        module = error.module

        # For AttributeError, regex might miss the module - use LLM
        if not module and self.llm and self.use_llm:
            module = self.llm.identify_module_from_error(
                error.message, list(modules.keys())
            )

        if not module:
            return False

        pkg = map_import_to_package(module)
        if not pkg:
            pkg = module.lower()

        if pkg not in modules:
            # Try finding in installed modules
            for installed_pkg in modules:
                if module.lower() in installed_pkg.lower():
                    pkg = installed_pkg
                    break
            else:
                return False

        # Track failed version and try older
        failed_versions.setdefault(pkg, set()).add(modules[pkg])
        versions = self._get_package_versions(pkg, python_version, version_cache)
        excluded = failed_versions.get(pkg, set())

        new_ver = self._choose_version(
            package=pkg,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
            current_version=modules[pkg],
            prefer_older=True,
        )

        if new_ver:
            modules[pkg] = new_ver
            return True

        return False

    def _fix_non_zero_code(self, error, modules, version_cache,
                           failed_versions, python_version, iteration):
        """Fix: handle pip install failure (compilation error, missing dep)."""
        module = error.module
        if not module:
            return False

        if module not in modules:
            return False

        # Track failed version
        failed_versions.setdefault(module, set()).add(modules[module])
        versions = self._get_package_versions(module, python_version, version_cache)
        excluded = failed_versions.get(module, set())

        # Check if it's a compilation error (Cython, C extension)
        if 'Cython' in error.message or 'gcc' in error.message or 'setup.py' in error.message:
            # Try a different version (often older wheel-based versions work)
            new_ver = self._choose_version(
                package=module,
                versions=versions,
                python_version=python_version,
                iteration=iteration,
                excluded=excluded,
                error=error,
                current_version=modules[module],
                prefer_older=True,
            )
            if new_ver:
                modules[module] = new_ver
                return True

            # If all versions fail to compile, remove the package
            modules.pop(module, None)
            return True

        # Try different version
        new_ver = self._choose_version(
            package=module,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
            current_version=modules[module],
            prefer_older=True,
        )
        if new_ver:
            modules[module] = new_ver
            return True

        # Remove the package as last resort
        modules.pop(module, None)
        return True

    def _fix_syntax_error(self, error, modules, version_cache,
                          failed_versions, python_version, iteration):
        """Fix: SyntaxError in an installed module (version mismatch)."""
        module = error.module
        if not module:
            # SyntaxError in snippet itself - need different Python version
            return False

        if module not in modules:
            return False

        # Track and try older version
        failed_versions.setdefault(module, set()).add(modules[module])
        versions = self._get_package_versions(module, python_version, version_cache)
        excluded = failed_versions.get(module, set())

        new_ver = self._choose_version(
            package=module,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
            current_version=modules[module],
            prefer_older=True,
        )

        if new_ver:
            modules[module] = new_ver
            return True

        return False

    def _fix_invalid_version(self, error, modules, version_cache,
                             failed_versions, python_version, iteration):
        """Fix: invalid version string."""
        module = error.module
        if not module or module not in modules:
            return False

        failed_versions.setdefault(module, set()).add(modules[module])
        versions = self._get_package_versions(module, python_version, version_cache)
        excluded = failed_versions.get(module, set())

        new_ver = self._choose_version(
            package=module,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
        )
        if new_ver:
            modules[module] = new_ver
            return True

        return False

    def _fix_generic_version(self, error, modules, version_cache,
                             failed_versions, python_version, iteration):
        """Fix: try downgrading the most likely offending module."""
        # Try to find the module from the error message
        module = error.module
        if not module:
            # Check each installed module in the error message
            for pkg in modules:
                if pkg in error.message:
                    module = pkg
                    break

        if not module:
            return False

        if module not in modules:
            pkg = map_import_to_package(module)
            if pkg and pkg in modules:
                module = pkg
            else:
                return False

        failed_versions.setdefault(module, set()).add(modules[module])
        versions = self._get_package_versions(module, python_version, version_cache)
        excluded = failed_versions.get(module, set())

        new_ver = self._choose_version(
            package=module,
            versions=versions,
            python_version=python_version,
            iteration=iteration,
            excluded=excluded,
            error=error,
            current_version=modules[module],
            prefer_older=True,
        )
        if new_ver:
            modules[module] = new_ver
            return True

        return False

    # ── Logging ──────────────────────────────────────────────────────────

    def _log_result(self, snippet_path, python_version, modules,
                    error_type, error_msg, iteration, start_time, success):
        """Log iteration result to YAML file (PLLM-compatible format)."""
        project_dir = os.path.dirname(snippet_path)
        log_file = os.path.join(project_dir, f"output_data_{python_version}.yml")

        try:
            with open(log_file, 'a') as f:
                if iteration == 1:
                    f.write('---\n')
                    f.write(f"python_version: {python_version}\n")
                    f.write(f"start_time: {start_time}\n")
                    f.write('iterations:\n')

                f.write(f"  iteration_{iteration}:\n")
                f.write(f"    - python_module: {modules}\n")
                f.write(f"    - error_type: {error_type}\n")
                f.write(f"    - error: |\n")
                for line in str(error_msg)[:500].split('\n'):
                    if line.strip():
                        f.write(f"        {line}\n")

                if success:
                    end_time = time.time()
                    f.write(f"end_time: {end_time}\n")
                    f.write(f"total_time: {end_time - start_time}\n")
        except (IOError, OSError):
            pass
