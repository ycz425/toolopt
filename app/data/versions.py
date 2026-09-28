import re
from pathlib import Path

DATASET_ROOT = Path("datasets")

_VERSION_DIR_RE = re.compile(r"v(\d+)")


def _version_numbers(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    return [
        int(match.group(1))
        for path in root.iterdir()
        if path.is_dir() and (match := _VERSION_DIR_RE.fullmatch(path.name))
    ]


def latest_version_dir(root: Path = DATASET_ROOT) -> Path:
    """Returns root/vK for the highest existing K (compared numerically, so v10 is newer than v9)."""
    versions = _version_numbers(root)
    if not versions:
        raise FileNotFoundError(f"no dataset versions (v1, v2, ...) found in {root}; run scripts.generate_data first")
    return root / f"v{max(versions)}"


def next_version_dir(root: Path = DATASET_ROOT) -> Path:
    """Returns root/v{K+1}, where K is the highest existing version (v1 if there are none). Does not create it."""
    return root / f"v{max(_version_numbers(root), default=0) + 1}"


def resolve_version_dir(version: str | None, root: Path = DATASET_ROOT) -> Path:
    """Returns root/<version> (e.g. "v3"), or the latest version when version is None."""
    if version is None:
        return latest_version_dir(root)
    path = root / version
    if not path.is_dir():
        raise FileNotFoundError(f"dataset version {path} does not exist")
    return path
