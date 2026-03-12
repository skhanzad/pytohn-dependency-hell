"""In-memory and disk-backed cache of successful package versions."""
import json
import os
import tempfile
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX
    fcntl = None


class SuccessfulVersionMemory:
    """Stores versions that have already succeeded for a Python version.

    The cache is kept in memory for fast reuse within a process and also
    persisted to disk so later resolver instances can reuse prior wins.
    """

    def __init__(self, cache_dir='./modules',
                 filename='.epllm_success_memory.json', logging=False):
        self.cache_dir = cache_dir
        self.filename = filename
        self.logging = logging
        self.memory_file = os.path.join(cache_dir, filename)
        self.lock_file = f"{self.memory_file}.lock"
        self._entries = {}

        os.makedirs(cache_dir, exist_ok=True)
        self._entries = self._read_entries()

    @contextmanager
    def _locked(self):
        """Best-effort inter-process lock for batch workers."""
        with open(self.lock_file, 'a') as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _read_entries(self):
        if not os.path.isfile(self.memory_file):
            return {}

        try:
            with open(self.memory_file, 'r') as handle:
                raw = json.load(handle)
        except (IOError, OSError, ValueError, json.JSONDecodeError):
            return {}

        normalized = {}
        for python_version, packages in raw.items():
            if not isinstance(packages, dict):
                continue
            normalized[python_version] = {}
            for package, versions in packages.items():
                if not isinstance(versions, list):
                    continue
                clean_versions = []
                for version in versions:
                    if isinstance(version, str) and version and version not in clean_versions:
                        clean_versions.append(version)
                if clean_versions:
                    normalized[python_version][package] = clean_versions

        return normalized

    def _write_entries(self, entries):
        directory = os.path.dirname(self.memory_file) or '.'
        fd, tmp_path = tempfile.mkstemp(
            prefix='.epllm_success_memory.',
            suffix='.tmp',
            dir=directory,
        )
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(entries, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, self.memory_file)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def remember_success(self, package, python_version, version):
        """Persist a successful package version."""
        if not package or not python_version or not version:
            return

        with self._locked():
            current = self._read_entries()
            py_versions = current.setdefault(python_version, {})
            remembered = py_versions.setdefault(package, [])

            if version in remembered:
                remembered.remove(version)
            remembered.insert(0, version)
            py_versions[package] = remembered[:10]

            self._write_entries(current)
            self._entries = current

        if self.logging:
            print(f"    Remembered success: {package}=={version} on Python {python_version}")

    def remember_resolution(self, python_version, modules):
        """Persist all modules from a successful resolution."""
        for package, version in sorted((modules or {}).items()):
            self.remember_success(package, python_version, version)

    def get_preferred_version(self, package, python_version, available_versions,
                              excluded=None, current_version=None,
                              prefer_older=False):
        """Return a remembered version if it is still installable."""
        excluded = set(excluded or [])
        remembered = (
            self._entries
            .get(python_version, {})
            .get(package, [])
        )
        if not remembered or not available_versions:
            return None

        order = {version: idx for idx, version in enumerate(available_versions)}
        candidates = [
            version for version in remembered
            if version in order and version not in excluded
        ]
        if not candidates:
            return None

        if prefer_older and current_version in order:
            current_idx = order[current_version]
            older = [
                version for version in candidates
                if order[version] < current_idx
            ]
            if older:
                return older[0]
            return None

        return candidates[0]
