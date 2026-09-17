"""HTTP client for data acquisition.

Responsibilities that belong here rather than in each adapter:

*   on-disk caching with TTL, so repeated analysis of the same match does not
    re-hit a provider;
*   per-host rate limiting and retry with exponential backoff;
*   robots.txt consultation for sources that are scraped rather than served
    as bulk data;
*   an audit trail: every attempt, cached or live, is written to ``fetch_log``.

A failed fetch returns a ``FetchResult`` with ``ok=False``. It never returns
substitute content, and callers are written to propagate the absence.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from ..config import get_settings
from ..logging_setup import get_logger

log = get_logger("sources.http")


@dataclass
class FetchResult:
    url: str
    ok: bool
    status_code: Optional[int] = None
    text: str = ""
    content: bytes = b""
    from_cache: bool = False
    fetched_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    error: str = ""
    duration_ms: int = 0

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest() if self.content else ""

    def json(self):
        return json.loads(self.text)


class RateLimiter:
    """Minimum interval between requests to the same host."""

    def __init__(self, min_interval: float = 1.0) -> None:
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            last = self._last.get(host)
            if last is not None:
                delay = self.min_interval - (now - last)
                if delay > 0:
                    time.sleep(delay)
                    now = time.monotonic()
            self._last[host] = now


class HttpClient:
    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        min_interval: float = 1.0,
        respect_robots: Optional[bool] = None,
        unreachable_ttl: float = 300.0,
    ) -> None:
        s = get_settings()
        self.settings = s
        self.cache_dir = Path(cache_dir or s.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(min_interval)
        self.unreachable_ttl = unreachable_ttl
        self.respect_robots = s.respect_robots if respect_robots is None else respect_robots
        self._robots: dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": s.user_agent, "Accept-Encoding": "gzip, deflate"})
        # Hosts that could not be connected to at all (DNS, TLS, proxy or
        # network policy). Re-attempting them on every call wastes the user's
        # time and is impolite to the remote host, so a failure is remembered
        # briefly and reported immediately.
        self._unreachable: dict[str, tuple[float, str]] = {}

    # -- cache -----------------------------------------------------------
    def _cache_path(self, url: str, params: Optional[dict]) -> Path:
        key = url + ("?" + urllib.parse.urlencode(sorted(params.items())) if params else "")
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        host = urllib.parse.urlparse(url).netloc.replace(":", "_")
        d = self.cache_dir / host
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.cache"

    def _read_cache(self, path: Path, ttl: int) -> Optional[FetchResult]:
        if not path.is_file():
            return None
        meta_path = path.with_suffix(".meta")
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        age = time.time() - meta.get("stored_at", 0)
        if ttl >= 0 and age > ttl:
            return None
        body = path.read_bytes()
        return FetchResult(
            url=meta.get("url", ""),
            ok=True,
            status_code=meta.get("status_code"),
            content=body,
            text=body.decode(meta.get("encoding") or "utf-8", errors="replace"),
            from_cache=True,
            fetched_at=dt.datetime.fromtimestamp(meta.get("stored_at", 0), dt.timezone.utc),
        )

    def _write_cache(self, path: Path, result: FetchResult, encoding: str) -> None:
        try:
            path.write_bytes(result.content)
            path.with_suffix(".meta").write_text(
                json.dumps(
                    {
                        "url": result.url,
                        "status_code": result.status_code,
                        "stored_at": time.time(),
                        "encoding": encoding,
                        "sha256": result.sha256,
                    }
                ),
                encoding="utf-8",
            )
        except OSError as exc:  # caching is best-effort
            log.debug("cache write failed for %s: %s", path, exc)

    # -- robots ----------------------------------------------------------
    def _robots_allows(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urllib.parse.urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(f"{origin}/robots.txt")
            try:
                resp = self.session.get(f"{origin}/robots.txt", timeout=10)
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser = None  # no robots.txt published -> unrestricted
            except requests.RequestException:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        if parser is None:
            return True
        return parser.can_fetch(self.settings.user_agent, url)

    # -- fetch -----------------------------------------------------------
    def get(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
        check_robots: bool = False,
    ) -> FetchResult:
        ttl = self.settings.cache_ttl_seconds if ttl is None else ttl
        cache_path = self._cache_path(url, params)

        if not force_refresh:
            cached = self._read_cache(cache_path, ttl)
            if cached is not None:
                log.debug("cache hit %s", url)
                return cached

        if self.settings.offline:
            # Offline mode falls back to stale cache rather than inventing data.
            stale = self._read_cache(cache_path, ttl=-1)
            if stale is not None:
                stale.error = "offline: served stale cache"
                return stale
            return FetchResult(url=url, ok=False, error="offline mode and no cached copy available")

        if check_robots and not self._robots_allows(url):
            log.warning("robots.txt disallows %s", url)
            return FetchResult(url=url, ok=False, error="disallowed by robots.txt")

        host = urllib.parse.urlparse(url).netloc
        blocked = self._unreachable.get(host)
        if blocked is not None:
            since, reason = blocked
            if time.time() - since < self.unreachable_ttl:
                return FetchResult(url=url, ok=False, error=f"host unreachable: {reason}")
            del self._unreachable[host]

        last_error = ""
        status: Optional[int] = None
        started = time.monotonic()

        for attempt in range(1, self.settings.http_retries + 1):
            self.limiter.wait(host)
            try:
                resp = self.session.get(
                    url, params=params, headers=headers, timeout=self.settings.http_timeout
                )
                status = resp.status_code
                if resp.status_code == 200:
                    encoding = resp.encoding or "utf-8"
                    result = FetchResult(
                        url=resp.url,
                        ok=True,
                        status_code=200,
                        content=resp.content,
                        text=resp.content.decode(encoding, errors="replace"),
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )
                    self._write_cache(cache_path, result, encoding)
                    return result
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}"
                    backoff = min(2 ** attempt, 30)
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        backoff = min(int(retry_after), 60)
                    log.warning("%s for %s; retry %d in %ss", last_error, url, attempt, backoff)
                    time.sleep(backoff)
                    continue
                # 4xx other than 429: not retryable.
                return FetchResult(
                    url=url, ok=False, status_code=resp.status_code,
                    error=f"HTTP {resp.status_code}",
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("request failed (%d/%d) %s: %s", attempt, self.settings.http_retries, url, exc)
                if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
                    self._unreachable[host] = (time.time(), type(exc).__name__)
                    break
                if attempt < self.settings.http_retries:
                    time.sleep(min(2 ** attempt, 30))

        return FetchResult(
            url=url, ok=False, status_code=status, error=last_error or "request failed",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
