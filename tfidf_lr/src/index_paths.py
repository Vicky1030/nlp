"""Portable path handling for stage-1 model indexes.

Version 2 indexes store model/probability paths relative to the index file.
Older indexes with absolute paths remain readable only at their explicit path;
there is no fallback to project defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

INDEX_FORMAT_VERSION = 2


def load_model_index(index_path: Path) -> dict:
    index_path = Path(index_path).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"model index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    version = index.get("index_format_version", 1)
    if version not in (1, INDEX_FORMAT_VERSION):
        raise ValueError(f"unsupported model index format version: {version!r}")
    if not isinstance(index.get("configs"), list):
        raise ValueError(f"model index configs is not a list: {index_path}")
    index["_index_path"] = str(index_path)
    return index


def resolve_index_path(index_path: Path, index: dict, stored: str, field: str) -> Path:
    """Resolve one index field independently of the current working directory."""
    if not isinstance(stored, str) or not stored:
        raise ValueError(f"model index field {field!r} is empty")
    raw = Path(stored)
    if raw.is_absolute():
        # Explicit compatibility for historical v1 indexes. This never falls
        # back to a project-relative path when the old absolute path is gone.
        resolved = raw.resolve()
    else:
        base = Path(index_path).resolve().parent
        if index.get("path_base", "index_directory") != "index_directory":
            raise ValueError("unsupported model index path_base; expected index_directory")
        resolved = (base / raw).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"model index {field} path is missing: {resolved}")
    return resolved


def relative_index_path(index_path: Path, target: Path) -> str:
    """Return a stable POSIX path relative to the index directory."""
    return target.resolve().relative_to(Path(index_path).resolve().parent).as_posix()
