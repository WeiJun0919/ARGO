"""Utilities for saving and loading ARGO dataset artifacts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def save_dataset_artifact(dataset, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Save a dataset object plus lightweight metadata to a pickle artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "metadata": metadata or {},
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_path = path.with_suffix(path.suffix + ".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload["metadata"], f, indent=2, ensure_ascii=False, default=str)
    return path


def load_dataset_artifact(path: str | Path):
    """Load a dataset artifact saved by save_dataset_artifact.

    For backward compatibility, if the pickle contains a raw dataset instead of a
    payload dict, return that dataset and empty metadata.
    """
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "dataset" in payload:
        return payload["dataset"], payload.get("metadata", {})
    return payload, {}
