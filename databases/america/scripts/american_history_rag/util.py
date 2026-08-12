from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests


USER_AGENT = "american-history-rag/0.1 (research corpus builder; respectful API client)"


class RateLimitError(requests.HTTPError):
    """Raised immediately on HTTP 429 so callers stop instead of adding load."""


class RateLimiter:
    def __init__(self, requests_per_minute: float, *, safety: float = 0.9):
        self.interval = 60.0 / (requests_per_minute * safety)
        self._lock = threading.Lock()
        self._next_request = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request - now)
            if delay:
                time.sleep(delay)
            self._next_request = time.monotonic() + self.interval


def session() -> requests.Session:
    client = requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    return client


def get_with_retries(client: requests.Session, url: str, *, timeout: int = 60,
                     attempts: int = 4) -> requests.Response:
    for attempt in range(attempts):
        try:
            response = client.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt + 1 >= attempts:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429:
            raise RateLimitError(
                f"Rate limited by {url}; LOC documents a one-hour block, so this run stopped."
            )
        if response.status_code not in {500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt + 1 < attempts:
            time.sleep(2 ** attempt)
    response.raise_for_status()
    return response


def cached_json(url: str, cache_dir: Path | None, *, timeout: int = 60,
                limiter: RateLimiter | None = None) -> object:
    """Fetch JSON with a persistent URL-keyed cache.

    Empty and invalid responses are never cached. Cache writes are atomic so a
    killed harvest cannot leave a valid-looking partial JSON file.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    if limiter is not None:
        limiter.wait()
    response = get_with_retries(session(), url, timeout=timeout)
    if not response.content:
        raise RuntimeError(f"Empty JSON response from {url}")
    payload = response.json()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=cache_path.parent,
                prefix=f".{cache_path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(temporary_name, cache_path)
        except Exception:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise
    return payload


def cached_text(url: str, cache_dir: Path | None, *, timeout: int = 60,
                limiter: RateLimiter | None = None) -> str:
    """Fetch UTF-8-ish source/metadata text with the same atomic cache policy."""
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
    if limiter is not None:
        limiter.wait()
    response = get_with_retries(session(), url, timeout=timeout)
    value = response.content.decode(response.encoding or "utf-8", errors="replace")
    if not value.strip():
        raise RuntimeError(f"Empty text response from {url}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=cache_path.parent,
                prefix=f".{cache_path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(value)
            os.replace(temporary_name, cache_path)
        except Exception:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise
    return value


def stable_id(*parts: str) -> str:
    value = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(value).hexdigest()[:24]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict], *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if append:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return count
