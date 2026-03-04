import re
from typing import Optional, Tuple, List


class ErrorParser:
    """Regex-based error parser replacing LLM error diagnosis."""

    # Patterns for error type detection
    VERSION_NOT_FOUND = re.compile(
        r"Could not find a version that satisfies the requirement\s+(\S+)==(\S+)"
    )
    MODULE_NOT_FOUND = re.compile(
        r"ModuleNotFoundError:\s*No module named ['\"]?([a-zA-Z0-9_.]+)['\"]?"
    )
    IMPORT_ERROR_NO_MODULE = re.compile(
        r"ImportError:\s*No module named\s+['\"]?([a-zA-Z0-9_.]+)['\"]?"
    )
    IMPORT_ERROR_CANNOT = re.compile(
        r"ImportError:\s*cannot import name ['\"]?(\w+)['\"]? from ['\"]?([a-zA-Z0-9_.]+)['\"]?"
    )
    ATTRIBUTE_ERROR = re.compile(
        r"AttributeError:\s*(?:module\s+['\"]?([a-zA-Z0-9_.]+)['\"]?\s+has no attribute|'module' object has no attribute)"
    )
    SYNTAX_ERROR = re.compile(r"SyntaxError:\s*(?:invalid syntax|Missing parentheses)")
    NON_ZERO_CODE = re.compile(
        r"pip install\s+\S+\s+(\S+)==(\S+).*returned a non-zero code"
    )
    DEPENDENCY_CONFLICT = re.compile(r"dependency conflict", re.IGNORECASE)
    INVALID_VERSION = re.compile(r"InvalidVersion", re.IGNORECASE)

    # Extract available versions from "from versions: ..." in error messages
    AVAILABLE_VERSIONS = re.compile(
        r"from versions:\s*([0-9][0-9a-zA-Z.,\s]*)\)"
    )

    # Python 2 markers in code
    PY2_PRINT = re.compile(r"^\s*print\s+['\"\w]", re.MULTILINE)
    PY2_URLLIB2 = re.compile(r"(?:^|\s)import\s+urllib2|from\s+urllib2\s+", re.MULTILINE)
    PY2_EXECFILE = re.compile(r"\bexecfile\s*\(", re.MULTILINE)
    PY2_RAW_INPUT = re.compile(r"\braw_input\s*\(", re.MULTILINE)
    PY2_XRANGE = re.compile(r"\bxrange\s*\(", re.MULTILINE)
    PY2_HAS_KEY = re.compile(r"\.has_key\s*\(", re.MULTILINE)
    PY2_EXCEPT_COMMA = re.compile(r"except\s+\w+\s*,\s*\w+\s*:", re.MULTILINE)

    # Non-PyPI system packages that frequently fail
    SYSTEM_PACKAGES = {
        "gtk", "gi", "gobject", "glib", "appindicator", "dbus",
        "apt", "apt_pkg", "aptdaemon", "software_properties",
        "xdg", "Xlib", "cairo", "pango", "atk", "wnck",
        "gconf", "gnome", "gnomekeyring", "pynotify",
    }

    def detect_python2(self, source_code: str) -> bool:
        """Heuristic Python 2 detection from source code."""
        markers = [
            self.PY2_PRINT, self.PY2_URLLIB2, self.PY2_EXECFILE,
            self.PY2_RAW_INPUT, self.PY2_XRANGE, self.PY2_HAS_KEY,
            self.PY2_EXCEPT_COMMA,
        ]
        hits = sum(1 for p in markers if p.search(source_code))
        return hits >= 2

    def parse_error(self, log: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse an error log and return (error_type, module_name, version).

        Returns:
            (error_type, module_name, failed_version)
        """
        if not log:
            return ("None", None, None)

        # Order matters: check more specific patterns first
        m = self.VERSION_NOT_FOUND.search(log)
        if m:
            return ("VersionNotFound", m.group(1), m.group(2))

        if self.DEPENDENCY_CONFLICT.search(log):
            # Try to extract module from the conflict message
            mod_match = re.search(r"(\S+)==\S+", log)
            mod = mod_match.group(1) if mod_match else None
            return ("DependencyConflict", mod, None)

        if self.INVALID_VERSION.search(log):
            mod_match = re.search(r"(\S+)==(\S+)", log)
            if mod_match:
                return ("InvalidVersion", mod_match.group(1), mod_match.group(2))
            return ("InvalidVersion", None, None)

        m = self.NON_ZERO_CODE.search(log)
        if m:
            return ("NonZeroCode", m.group(1), m.group(2))

        m = self.MODULE_NOT_FOUND.search(log)
        if m:
            mod = m.group(1).split(".")[0]
            return ("ModuleNotFound", mod, None)

        m = self.IMPORT_ERROR_NO_MODULE.search(log)
        if m:
            mod = m.group(1).split(".")[0]
            return ("ImportError", mod, None)

        m = self.IMPORT_ERROR_CANNOT.search(log)
        if m:
            # The parent module is more useful for version changes
            mod = m.group(2).split(".")[0]
            return ("ImportError", mod, None)

        m = self.ATTRIBUTE_ERROR.search(log)
        if m:
            mod = m.group(1) if m.group(1) else None
            if mod:
                mod = mod.split(".")[0]
            return ("AttributeError", mod, None)

        if self.SYNTAX_ERROR.search(log):
            return ("SyntaxError", None, None)

        return ("None", None, None)

    def extract_available_versions(self, log: str) -> List[str]:
        """Extract version list from 'from versions: ...' in error messages."""
        m = self.AVAILABLE_VERSIONS.search(log)
        if not m:
            return []
        raw = m.group(1)
        versions = [v.strip() for v in raw.split(",") if v.strip()]
        return versions

    def is_system_package(self, module_name: str) -> bool:
        """Check if a module is likely a system-only (non-PyPI) package."""
        return module_name.lower() in self.SYSTEM_PACKAGES

    def has_python2_syntax_error(self, log: str) -> bool:
        """Check if a SyntaxError in the log looks like Python 2 code running on Python 3."""
        if "SyntaxError" not in log:
            return False
        # Common Python 2 syntax that fails on 3
        py2_markers = [
            "print ",  # print statement
            "def quit_all(self) -> None:",  # arrow syntax in py2 context
            "except ",
        ]
        return any(marker in log for marker in py2_markers)
