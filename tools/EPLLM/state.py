"""Shared data structures for EPLLM hybrid resolver."""


class ErrorInfo:
    """Structured error information extracted from Docker build/run output."""
    __slots__ = ('error_type', 'module', 'version', 'available_versions',
                 'message', 'required_by', 'required_version')

    def __init__(self, error_type, module=None, version=None,
                 available_versions=None, message='',
                 required_by=None, required_version=None):
        self.error_type = error_type
        self.module = module
        self.version = version
        self.available_versions = available_versions or []
        self.message = message
        self.required_by = required_by
        self.required_version = required_version

    def __repr__(self):
        return (f"ErrorInfo(type={self.error_type!r}, module={self.module!r}, "
                f"version={self.version!r})")


class ResolveResult:
    """Result of resolving dependencies for a single snippet."""
    __slots__ = ('success', 'python_version', 'modules', 'error_type',
                 'iterations', 'duration', 'error_message')

    def __init__(self, success, python_version=None, modules=None,
                 error_type=None, iterations=0, duration=0.0, error_message=''):
        self.success = success
        self.python_version = python_version
        self.modules = modules or {}
        self.error_type = error_type
        self.iterations = iterations
        self.duration = duration
        self.error_message = error_message

    def __repr__(self):
        status = "PASS" if self.success else f"FAIL({self.error_type})"
        return f"ResolveResult({status}, py={self.python_version}, mods={len(self.modules)})"

    def modules_str(self):
        """Format modules as semicolon-separated string for CSV output."""
        return "; ".join(f"{k}=={v}" for k, v in self.modules.items())
