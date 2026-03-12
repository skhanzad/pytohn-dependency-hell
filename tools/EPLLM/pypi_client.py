"""PyPI API client with disk-based version caching.

Queries PyPI for package metadata and caches version lists to disk.
Compatible with PLLM's modules/ cache directory format.
"""
import os
import re
import json
import requests
from pathlib import Path


PYPI_BASE_URL = "https://pypi.org/pypi"

# Python version release dates for filtering versions
PYTHON_VERSIONS = [
    {"cycle": "3.12", "releaseDate": "2023-10-02"},
    {"cycle": "3.11", "releaseDate": "2022-10-24"},
    {"cycle": "3.10", "releaseDate": "2021-10-04"},
    {"cycle": "3.9", "releaseDate": "2020-10-05"},
    {"cycle": "3.8", "releaseDate": "2019-10-14"},
    {"cycle": "3.7", "releaseDate": "2018-06-26"},
    {"cycle": "3.6", "releaseDate": "2016-12-22"},
    {"cycle": "3.5", "releaseDate": "2015-09-12"},
    {"cycle": "3.4", "releaseDate": "2014-03-15"},
    {"cycle": "3.3", "releaseDate": "2012-09-29"},
    {"cycle": "2.7", "releaseDate": "2010-07-03"},
    {"cycle": "2.6", "releaseDate": "2008-10-01"},
]


def _version_sort_key(version_str):
    """Sort key for version strings (e.g., '1.2.3', '2.0.0rc1')."""
    parts = re.split(r'(\d+)', version_str)
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return result


class PyPIClient:
    """Queries PyPI and caches version lists to disk."""

    def __init__(self, cache_dir='./modules', logging=False):
        self.cache_dir = cache_dir
        self.logging = logging
        self._memory_cache = {}
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, package, python_version):
        """Path to the cached version file for a package + Python version."""
        return os.path.join(self.cache_dir, f"{package}_{python_version}.txt")

    def get_versions(self, package, python_version):
        """Get available versions for a package, using cache if available.

        Returns a sorted list of version strings (oldest to newest).
        """
        cache_key = f"{package}_{python_version}"

        # Memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Disk cache (compatible with PLLM format)
        cache_file = self._cache_path(package, python_version)
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = f.read().strip()
                if data:
                    versions = [v.strip() for v in data.split(',') if v.strip()]
                    self._memory_cache[cache_key] = versions
                    return versions
            except (IOError, OSError):
                pass

        # Query PyPI
        versions = self._fetch_from_pypi(package, python_version)
        if versions:
            self._memory_cache[cache_key] = versions
            # Write to disk cache
            try:
                with open(cache_file, 'w') as f:
                    f.write(', '.join(versions))
            except (IOError, OSError):
                pass

        return versions

    def _fetch_from_pypi(self, package, python_version):
        """Fetch version list from PyPI API."""
        try:
            resp = requests.get(
                f"{PYPI_BASE_URL}/{package}/json",
                timeout=15
            )
            if resp.status_code == 404:
                if self.logging:
                    print(f"  PyPI: {package} not found")
                return []
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if self.logging:
                print(f"  PyPI query failed for {package}: {e}")
            return []

        releases = data.get('releases', {})
        if not releases:
            return []

        # Filter versions: exclude yanked, pre-release only when we have stable
        versions = []
        for ver, files in releases.items():
            if not files:
                continue
            # Skip yanked versions
            if all(f.get('yanked', False) for f in files):
                continue
            # Skip versions with invalid format
            if not re.match(r'^\d+', ver):
                continue
            versions.append(ver)

        # Sort by version
        try:
            versions.sort(key=_version_sort_key)
        except (TypeError, ValueError):
            versions.sort()

        return versions

    def package_exists(self, package):
        """Check if a package exists on PyPI."""
        try:
            resp = requests.head(
                f"{PYPI_BASE_URL}/{package}/json",
                timeout=10
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_dependencies(self, package, version=None):
        """Get dependency list for a specific package version.

        Returns a list of (dep_name, version_spec) tuples.
        """
        try:
            url = f"{PYPI_BASE_URL}/{package}/json"
            if version:
                url = f"{PYPI_BASE_URL}/{package}/{version}/json"
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError):
            return []

        requires = data.get('info', {}).get('requires_dist') or []
        deps = []
        for req in requires:
            # Skip extras/conditional deps
            if 'extra ==' in req:
                continue
            # Parse "package_name (>=1.0,<2.0)"
            m = re.match(r'^([a-zA-Z0-9_.-]+)\s*(?:\(([^)]+)\))?', req)
            if m:
                dep_name = m.group(1).strip().lower()
                dep_spec = m.group(2) or ''
                deps.append((dep_name, dep_spec))
        return deps

    def resolve_package_name(self, import_name):
        """Try to find the correct PyPI package name for an import.

        Tries several common naming patterns.
        """
        candidates = [
            import_name,
            import_name.lower(),
            import_name.replace('_', '-'),
            import_name.replace('-', '_'),
            f"python-{import_name}",
            f"py{import_name}",
        ]
        for candidate in candidates:
            if self.package_exists(candidate):
                return candidate
        return import_name.lower()
