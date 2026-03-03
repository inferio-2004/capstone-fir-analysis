"""
Shared disk-based caching for Indian Kanoon API responses and other data.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "output" / "kanoon_cache"


def cache_key(query: str) -> str:
    """Generate an MD5-based cache key from an arbitrary string."""
    return hashlib.md5(query.encode()).hexdigest()


def load_cache(key: str) -> Optional[dict]:
    """Load a cached JSON blob by key, or return None."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_cache(key: str, data):
    """Persist a JSON-serialisable object under the given key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
