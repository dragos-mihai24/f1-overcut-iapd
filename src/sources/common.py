"""Shared helpers: config loading and a throttled JSON file cache for HTTP sources."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "iapd-f1-undercut-student-project/0.1 (educational use)"
_last_call: dict[str, float] = {}
OFFLINE = False  # when True, only cached responses are used (no network calls)


class CacheMiss(Exception):
    """Raised in OFFLINE mode when a response is not cached yet."""


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_dir(name: str) -> Path:
    d = ROOT / load_config()["paths"]["cache"] / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_get_json(url: str, params: dict | None, cache: str, min_interval: float = 0.0,
                    allow_404: bool = False) -> dict | list:
    """GET a JSON endpoint, caching the response on disk (keyed by url+params).

    `min_interval` is the minimum delay between two *network* calls to the same
    source (cache hits are free), used to respect API rate limits.
    """
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = cache_dir(cache) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if OFFLINE:
        raise CacheMiss(url)
    for attempt in range(8):
        wait = min_interval - (time.time() - _last_call.get(cache, 0))
        if wait > 0:
            time.sleep(wait)
        _last_call[cache] = time.time()
        try:
            r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        except requests.RequestException:
            time.sleep(30)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(60 * (attempt + 1))
            continue
        if r.status_code == 404 and allow_404:
            data = []
        else:
            r.raise_for_status()
            data = r.json()
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    raise RuntimeError(f"Too many retries for {url}")
