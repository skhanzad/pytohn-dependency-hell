import re
from typing import List, Optional, Dict


class VersionSelector:
    """Deterministic version selection replacing LLM version picking."""

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
            if v not in exclude_set:
                return v
        return None

    @classmethod
    def pick_by_binary_search(cls, versions: List[str], exclude: List[str],
                              prefer_newer: bool = True) -> Optional[str]:
        """Binary search the version space, picking midpoint of remaining candidates."""
        sorted_v = cls._sort_versions(versions)
        exclude_set = set(exclude)
        candidates = [v for v in sorted_v if v not in exclude_set]
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
    def select(cls, module: str, versions: List[str],
               failed: List[str], iteration: int,
               available_from_error: Optional[List[str]] = None) -> Optional[str]:
        """Main version selection entry point. Escalates strategy by iteration.

        Strategy escalation:
        - iteration 0-1: pick latest
        - iteration 2: pick from error versions if available, else binary search (newer)
        - iteration 3+: binary search (older)
        """
        if not versions and not available_from_error:
            return None

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
