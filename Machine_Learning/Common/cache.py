"""Common/cache.py - lightweight on-disk cache for expensive dataframe loads.

Keyed on the source file's resolved path + mtime + size, so edits to raw
CSVs automatically invalidate the cache without any manual bookkeeping.
Loading the full Patient_Data tree can involve hundreds of small CSVs;
this avoids re-parsing them on every pipeline run during iteration.
"""

import hashlib
import pickle
from pathlib import Path

from Config import config
from Common.utils import get_logger

logger = get_logger("cache")


def _key_for(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()


def cached_read(path, loader_fn, *args, **kwargs):
    """Return loader_fn(path, *args, **kwargs), transparently caching the
    result on disk keyed by the file's mtime/size."""
    path = Path(path)
    if not path.exists():
        return loader_fn(path, *args, **kwargs)

    cache_file = config.CACHE_DIR / f"{_key_for(path)}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # corrupted cache entry -> recompute
            logger.warning(f"Cache read failed for {path}: {exc}")

    result = loader_fn(path, *args, **kwargs)
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(result, fh)
    except Exception as exc:
        logger.warning(f"Cache write failed for {path}: {exc}")
    return result


def clear_cache():
    n = 0
    for f in config.CACHE_DIR.glob("*.pkl"):
        f.unlink()
        n += 1
    logger.info(f"Cache cleared ({n} entries removed).")
