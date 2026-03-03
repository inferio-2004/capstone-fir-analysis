from __future__ import annotations

import json
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _benchmark_file() -> Path:
    return _repo_root() / "output" / "model_benchmark_latest.json"


def get_preferred_groq_model(default_model: str) -> str:
    """
    Resolve Groq model in this priority order:
    1) `GROQ_MODEL` env var
    2) `output/model_benchmark_latest.json` -> `best_model`
    3) fallback `default_model`
    """
    env_model = os.environ.get("GROQ_MODEL", "").strip()
    if env_model:
        return env_model

    benchmark_path = _benchmark_file()
    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = str(data.get("best_model", "")).strip()
            if model:
                return model
        except Exception:
            pass

    return default_model
