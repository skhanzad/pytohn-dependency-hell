import re
from typing import List, Optional, Dict


# Known compatible version combinations for common package groups.
# When one module is resolved, try to use compatible versions for related modules.
COMPAT_GROUPS = {
    "tensorflow": {
        "tensorflow": {
            "2.1.0": {"numpy": "1.18.5"},
            "2.4.0": {"numpy": "1.19.5"},
            "2.10.0": {"numpy": "1.23.5"},
            "2.13.1": {"numpy": "1.24.3"},
            "1.15.0": {"numpy": "1.16.6"},
        },
    },
    "pymc3": {
        "pymc3": {
            "3.11.5": {"theano-pymc": "1.1.2", "scipy": "1.7.3", "numpy": "1.21.6"},
            "3.11.2": {"theano-pymc": "1.1.2", "scipy": "1.7.3"},
            "3.9.3": {"theano-pymc": "1.0.11", "scipy": "1.5.4"},
        },
    },
    "django": {
        "django": {
            "1.11.29": {"djangorestframework": "3.9.4"},
            "2.2.28": {"djangorestframework": "3.12.4"},
            "3.2.25": {"djangorestframework": "3.14.0"},
        },
    },
}

# Modules that should be replaced with a different pip package name
# when the original fails (beyond what module_link.json covers).
FALLBACK_PACKAGES = {
    "theano": "theano-pymc",
    "image": "Pillow",
    "path": "path",  # not pathpy
}


class VersionSelector:
    """Deterministic version selection with compatibility awareness."""

    @staticmethod
    def _version_key(version: str):
        """Sort key for version strings like '1.2.3', '1.0.0rc1'."""
        return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', version)]

    @staticmethod
    def _sort_versions(versions: List[str]) -> List[str]:
        """Sort versions from oldest to newest."""
        try:
            return sorted(versions, key=VersionSelector._version_key)
        except Exception:
            return versions

    @classmethod
    def pick_latest(cls, versions: List[str], exclude: List[str]) -> Optional[str]:
        """Pick the newest non-excluded version."""
        sorted_v = cls._sort_versions(versions)
        exclude_set = set(exclude)
        for v in reversed(sorted_v):
            if v not in exclude_set and v != "0.0.0":
                return v
        return None

    @classmethod
    def pick_by_binary_search(cls, versions: List[str], exclude: List[str],
                              prefer_newer: bool = True) -> Optional[str]:
        """Binary search the version space, picking midpoint of remaining candidates."""
        sorted_v = cls._sort_versions(versions)
        exclude_set = set(exclude)
        candidates = [v for v in sorted_v if v not in exclude_set and v != "0.0.0"]
        if not candidates:
            return None
        if prefer_newer:
            # Pick from upper half midpoint
            mid = (len(candidates) + len(candidates) - 1) // 2
        else:
            # Pick from lower half midpoint
            mid = len(candidates) // 4
        mid = max(0, min(mid, len(candidates) - 1))
        return candidates[mid]

    @classmethod
    def pick_from_error_versions(cls, available: List[str],
                                 exclude: List[str]) -> Optional[str]:
        """Pick from versions listed in a Docker error message."""
        if not available:
            return None
        return cls.pick_latest(available, exclude)

    @classmethod
    def pick_compatible(cls, module: str, resolved_modules: Dict[str, str],
                        versions: List[str], exclude: List[str]) -> Optional[str]:
        """Pick a version that's known to be compatible with other resolved modules."""
        for group_name, group_data in COMPAT_GROUPS.items():
            if module in group_data:
                mod_compat = group_data[module]
                # Try each known-good version for this module
                for compat_ver, deps in sorted(mod_compat.items(),
                                                key=lambda x: cls._version_key(x[0]),
                                                reverse=True):
                    if compat_ver in exclude:
                        continue
                    # Check if this version is in available versions
                    if versions and compat_ver not in versions:
                        continue
                    return compat_ver
            # Also check if this module appears as a dependency in a compat group
            for anchor_mod, anchor_data in group_data.items():
                if anchor_mod in resolved_modules:
                    anchor_ver = resolved_modules[anchor_mod]
                    if anchor_ver in anchor_data:
                        suggested_ver = anchor_data[anchor_ver].get(module)
                        if suggested_ver and suggested_ver not in exclude:
                            if not versions or suggested_ver in versions:
                                return suggested_ver
        return None

    @classmethod
    def get_fallback_package(cls, module: str) -> Optional[str]:
        """Return an alternative pip package name for a module that keeps failing."""
        return FALLBACK_PACKAGES.get(module)

    @classmethod
    def select(cls, module: str, versions: List[str],
               failed: List[str], iteration: int,
               available_from_error: Optional[List[str]] = None,
               resolved_modules: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Main version selection entry point. Escalates strategy by iteration.

        Strategy escalation:
        - iteration 0-1: try compatible version first, then pick latest
        - iteration 2: pick from error versions if available, else binary search (newer)
        - iteration 3-4: binary search (older)
        - iteration 5+: try LLM-suggested versions (handled by resolver)
        """
        if not versions and not available_from_error:
            return None

        # Always try compatibility-aware selection first
        if resolved_modules:
            compat = cls.pick_compatible(module, resolved_modules, versions, failed)
            if compat:
                return compat

        # If we have versions from the error message, prefer those
        if available_from_error and iteration >= 2:
            pick = cls.pick_from_error_versions(available_from_error, failed)
            if pick:
                return pick

        if not versions:
            versions = available_from_error or []

        if iteration <= 1:
            return cls.pick_latest(versions, failed)
        elif iteration == 2:
            return cls.pick_by_binary_search(versions, failed, prefer_newer=True)
        else:
            return cls.pick_by_binary_search(versions, failed, prefer_newer=False)
