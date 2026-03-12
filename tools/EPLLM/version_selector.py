"""Multi-strategy version selection for dependency resolution.

Implements deterministic version selection strategies that replace
PLLM's LLM-based version picking. Strategies include:
- Latest version (most common default)
- Binary search (newer/older halves)
- Error-message-guided selection
- Dependency-conflict-aware selection
"""
import re


def select_version(versions, strategy='latest', excluded=None, error_info=None):
    """Select a version from the available versions list.

    Args:
        versions: List of version strings sorted oldest to newest.
        strategy: One of 'latest', 'binary_newer', 'binary_older',
                  'from_error', 'quartile_newer', 'quartile_older',
                  'middle', 'oldest_stable'.
        excluded: Set of version strings to exclude.
        error_info: ErrorInfo object for error-guided selection.

    Returns:
        A version string, or None if no suitable version found.
    """
    if not versions:
        return None

    excluded = excluded or set()
    available = [v for v in versions if v not in excluded]

    if not available:
        return None

    # Filter out pre-release versions for initial attempts
    stable = [v for v in available if _is_stable(v)]
    pool = stable if stable else available

    if not pool:
        return None

    if strategy == 'latest':
        return pool[-1]

    elif strategy == 'binary_newer':
        # Pick from the newer 75th percentile
        idx = len(pool) * 3 // 4
        return pool[min(idx, len(pool) - 1)]

    elif strategy == 'binary_older':
        # Pick from the older 25th percentile
        idx = len(pool) // 4
        return pool[idx]

    elif strategy == 'middle':
        # Pick the middle version
        return pool[len(pool) // 2]

    elif strategy == 'quartile_newer':
        # Pick from the 87.5th percentile
        idx = len(pool) * 7 // 8
        return pool[min(idx, len(pool) - 1)]

    elif strategy == 'quartile_older':
        # Pick from the 12.5th percentile
        idx = len(pool) // 8
        return pool[idx]

    elif strategy == 'oldest_stable':
        return pool[0]

    elif strategy == 'from_error':
        return _select_from_error(pool, error_info, excluded)

    return pool[-1]  # fallback to latest


def get_strategy_for_iteration(iteration, error_info=None):
    """Determine the version selection strategy based on iteration number.

    Iteration 0: Try latest version (most likely to work)
    Iteration 1: If error has available versions, use those; else binary newer
    Iteration 2: Binary older (try older, more battle-tested versions)
    Iteration 3: Middle version
    Iteration 4: Quartile older
    Iteration 5: Quartile newer
    Iteration 6+: Oldest stable
    """
    if error_info and error_info.available_versions:
        return 'from_error'

    strategies = [
        'latest',           # 0: start with latest
        'binary_newer',     # 1: try newer side
        'binary_older',     # 2: try older side
        'middle',           # 3: try middle ground
        'quartile_older',   # 4: try pretty old
        'quartile_newer',   # 5: try pretty new
        'oldest_stable',    # 6: try oldest
    ]

    if iteration < len(strategies):
        return strategies[iteration]
    return 'oldest_stable'


def select_version_for_iteration(versions, iteration, excluded=None, error_info=None):
    """Convenience: select version using iteration-based strategy."""
    strategy = get_strategy_for_iteration(iteration, error_info)
    return select_version(versions, strategy, excluded, error_info)


def _is_stable(version_str):
    """Check if a version string represents a stable release."""
    # Pre-release indicators: alpha, beta, rc, dev, pre
    return not re.search(r'(a|b|rc|dev|pre|alpha|beta)\d*$', version_str, re.IGNORECASE)


def _select_from_error(pool, error_info, excluded):
    """Select a version from the error message's available versions list.

    When pip reports "from versions: v1, v2, v3...", we pick from those
    versions using a distributed sampling strategy.
    """
    if not error_info or not error_info.available_versions:
        return pool[-1] if pool else None

    # Find versions that are both in the error's list and our pool
    error_versions = set(error_info.available_versions)
    candidates = [v for v in pool if v in error_versions]

    if not candidates:
        # If no overlap, use the error versions directly (filtered by excluded)
        candidates = [v for v in error_info.available_versions
                      if v not in excluded and _is_stable(v)]

    if not candidates:
        candidates = [v for v in error_info.available_versions
                      if v not in excluded]

    if not candidates:
        return pool[-1] if pool else None

    # Pick from the newer end of available versions
    return candidates[-1]


def pick_alternative_version(versions, current_version, excluded=None, prefer_older=False):
    """Pick an alternative version when the current one fails.

    Tries to find a version that's significantly different from the current one.
    """
    if not versions:
        return None

    excluded = set(excluded or [])
    excluded.add(current_version)
    available = [v for v in versions if v not in excluded]

    if not available:
        return None

    # Find current version's position
    try:
        current_idx = versions.index(current_version)
    except ValueError:
        current_idx = len(versions) - 1

    if prefer_older:
        # Try versions significantly older
        target_idx = max(0, current_idx - max(1, len(versions) // 4))
        # Find nearest available
        for i in range(target_idx, -1, -1):
            if versions[i] in available:
                return versions[i]
        # Nothing older, try newer
        for i in range(current_idx + 1, len(versions)):
            if versions[i] in available:
                return versions[i]
    else:
        # Try versions significantly newer
        target_idx = min(len(versions) - 1, current_idx + max(1, len(versions) // 4))
        for i in range(target_idx, len(versions)):
            if versions[i] in available:
                return versions[i]
        # Nothing newer, try older
        for i in range(current_idx - 1, -1, -1):
            if versions[i] in available:
                return versions[i]

    return available[-1] if available else None
