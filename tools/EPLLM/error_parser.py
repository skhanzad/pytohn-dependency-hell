"""Regex-based error classification from Docker build/run output.

Replaces PLLM's LLM-based error parsing with fast deterministic regex.
Handles all 8+ error types with structured information extraction.
"""
import re
from EPLLM.state import ErrorInfo  # noqa: E402


def parse_error(output):
    """Classify and extract structured info from Docker build/run output.

    Returns an ErrorInfo with the error type and extracted details.
    Error types: VersionNotFound, DependencyConflict, ModuleNotFound,
    ImportError, AttributeError, SyntaxError, NonZeroCode, InvalidVersion,
    NameError, TypeError, OtherError, NoError.
    """
    if not output or not isinstance(output, str):
        return ErrorInfo('NoError')

    # ── VersionNotFound ──────────────────────────────────────────────────
    if 'Could not find a version' in output:
        return _parse_version_not_found(output)

    # ── DependencyConflict ───────────────────────────────────────────────
    if 'dependency conflict' in output.lower() or 'incompatible' in output.lower():
        return _parse_dependency_conflict(output)

    # ── ModuleNotFoundError (runtime) ────────────────────────────────────
    if 'ModuleNotFoundError' in output:
        return _parse_module_not_found(output)

    # ── ImportError ──────────────────────────────────────────────────────
    if 'ImportError' in output:
        return _parse_import_error(output)

    # ── SyntaxError ──────────────────────────────────────────────────────
    if 'SyntaxError' in output:
        return _parse_syntax_error(output)

    # ── AttributeError ───────────────────────────────────────────────────
    if 'AttributeError' in output:
        return _parse_attribute_error(output)

    # ── NonZeroCode (pip install failure) ────────────────────────────────
    if 'non-zero code' in output or 'returned a non-zero' in output:
        return _parse_non_zero_code(output)

    # ── InvalidVersion ───────────────────────────────────────────────────
    if 'InvalidVersion' in output:
        return _parse_invalid_version(output)

    # ── NameError ────────────────────────────────────────────────────────
    if 'NameError' in output:
        return ErrorInfo('NameError', message=output[:500])

    # ── TypeError ────────────────────────────────────────────────────────
    if 'TypeError' in output:
        return ErrorInfo('TypeError', message=output[:500])

    # ── Check for any traceback (runtime error) ──────────────────────────
    if 'Traceback (most recent call last)' in output:
        return ErrorInfo('OtherError', message=output[:500])

    # ── Check for ERROR in build output ──────────────────────────────────
    if 'ERROR' in output or 'errorDetail' in output:
        return _parse_generic_error(output)

    return ErrorInfo('NoError')


def _parse_version_not_found(output):
    """Extract module name and available versions from pip's error."""
    # Pattern: "requirement package_name==version (from versions: v1, v2, ...)"
    m = re.search(r'requirement\s+(\S+?)(?:==\S+)?\s+\(from versions:\s*([^)]+)\)', output)
    module = None
    available = []
    if m:
        module = m.group(1).strip()
        available = [v.strip() for v in m.group(2).split(',') if v.strip()]
    else:
        # Fallback: look for the pip install command
        m = re.search(r'pip install[^"]*"([^"=]+)==([^"]+)"', output)
        if m:
            module = m.group(1).strip()
        else:
            m = re.search(r'No matching distribution found for\s+(\S+)', output)
            if m:
                module = m.group(1).split('==')[0].strip()

    return ErrorInfo('VersionNotFound', module=module,
                     available_versions=available, message=output[:500])


def _parse_dependency_conflict(output):
    """Extract conflict details from pip's dependency resolution error."""
    # Pattern: "package X requires Y>=V, but you have Y==V2"
    module = None
    required_by = None
    required_version = None

    m = re.search(r'(\S+)\s+requires\s+(\S+?)([><=!]+\S+)?(?:,|\s|$)', output)
    if m:
        required_by = m.group(1).strip()
        module = m.group(2).strip().lower()
        required_version = m.group(3) if m.group(3) else None

    if not module:
        # Fallback: look for "is incompatible" patterns
        m = re.search(r'(\S+)==\S+\s+is incompatible', output)
        if m:
            module = m.group(1).strip()

    return ErrorInfo('DependencyConflict', module=module,
                     required_by=required_by, required_version=required_version,
                     message=output[:500])


def _parse_module_not_found(output):
    """Extract missing module from ModuleNotFoundError."""
    # Pattern: "No module named 'module_name'"
    m = re.search(r"No module named '([^']+)'", output)
    if not m:
        m = re.search(r'No module named (\S+)', output)

    module = None
    if m:
        module = m.group(1).strip().strip("'\"")
        module = module.split('.')[0]  # top-level module

    return ErrorInfo('ModuleNotFound', module=module, message=output[:500])


def _parse_import_error(output):
    """Extract module causing ImportError."""
    module = None

    # Pattern: "cannot import name 'X' from 'module'"
    m = re.search(r"cannot import name '?(\w+)'?\s+from\s+'?(\w[^'\"]*)'?", output)
    if m:
        module = m.group(2).split('.')[0]
    else:
        # First, try to find the import statement in the traceback
        # This is more reliable than parsing the error message alone.
        # e.g., "from django.test.simple import X\nImportError: No module named simple"
        # → should return "django", not "simple"
        m = re.search(r'(?:from\s+(\S+)\s+import|import\s+(\S+)).*\n\s*ImportError', output)
        if m:
            imp = m.group(1) or m.group(2)
            module = imp.split('.')[0]
        else:
            # Fallback: parse "ImportError: No module named X"
            m = re.search(r'ImportError:\s*No module named\s+(\S+)', output)
            if m:
                missing = m.group(1).strip("'\"")
                # Check if the missing name appears as a submodule in a from...import
                # e.g., "from django.test.simple" → missing is "simple" → use "django"
                parent_m = re.search(
                    r'from\s+(\w[\w.]*)\.' + re.escape(missing) + r'\s+import',
                    output
                )
                if parent_m:
                    module = parent_m.group(1).split('.')[0]
                else:
                    module = missing.split('.')[0]
            else:
                # Last resort: any import line before ImportError
                m = re.search(r'import\s+(\w+).*\n.*ImportError', output)
                if m:
                    module = m.group(1)

    return ErrorInfo('ImportError', module=module, message=output[:500])


def _parse_syntax_error(output):
    """Extract info from SyntaxError."""
    module = None

    # Check if SyntaxError is in a site-packages file (module issue)
    m = re.search(r'site-packages/(\w+)', output)
    if m:
        module = m.group(1)
    else:
        # Check if it's in the snippet itself (Python version issue)
        if '/app/snippet.py' in output:
            return ErrorInfo('SyntaxError', module=None,
                             message='snippet_syntax_error')

    return ErrorInfo('SyntaxError', module=module, message=output[:500])


def _parse_attribute_error(output):
    """Extract module causing AttributeError."""
    module = None

    # Pattern: "'module' object has no attribute 'X'"
    # Look at the import/module in the traceback
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'AttributeError' in line:
            # Look at previous lines for the module reference
            for j in range(max(0, i - 5), i):
                m = re.search(r'import\s+(\w+)', lines[j])
                if m:
                    module = m.group(1)
                    break
                m = re.search(r'from\s+(\w+)', lines[j])
                if m:
                    module = m.group(1)
                    break
            break

    if not module:
        # Pattern: "module 'X' has no attribute 'Y'"
        m = re.search(r"module '(\w+)'", output)
        if m:
            module = m.group(1)

    return ErrorInfo('AttributeError', module=module, message=output[:500])


def _parse_non_zero_code(output):
    """Extract failing module from pip install failure."""
    module = None

    # Pattern: "pip install ... package==version returned a non-zero code"
    m = re.search(r'pip install[^"]*["\s](\S+)==(\S+)["\s].*(?:non-zero|returned)', output)
    if m:
        module = m.group(1).strip()
    else:
        # Look for the package in error detail
        m = re.search(r'"pip","install"[^]]*"([^"=]+)==', output)
        if m:
            module = m.group(1).strip()
        else:
            m = re.search(r'pip install\s+\S+\s+\S+\s+(\S+)==', output)
            if m:
                module = m.group(1).strip()

    return ErrorInfo('NonZeroCode', module=module, message=output[:500])


def _parse_invalid_version(output):
    """Extract module with invalid version."""
    module = None
    m = re.search(r'InvalidVersion.*?(\w[\w-]+)==', output)
    if m:
        module = m.group(1).strip()
    return ErrorInfo('InvalidVersion', module=module, message=output[:500])


def _parse_generic_error(output):
    """Parse generic Docker build errors."""
    module = None
    # Try to find the package that caused the error
    m = re.search(r'pip install[^"]*"?(\S+)==', output)
    if m:
        module = m.group(1).strip()
    return ErrorInfo('OtherError', module=module, message=output[:500])


def is_python_version_error(error_info):
    """Check if the error suggests we need a different Python version."""
    if error_info.error_type == 'SyntaxError' and error_info.message == 'snippet_syntax_error':
        return True
    if error_info.error_type == 'SyntaxError' and error_info.module is None:
        return True
    return False
