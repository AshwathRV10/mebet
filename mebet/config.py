"""Runtime configuration.

Everything that varies between machines or deployments lives here and is
driven by environment variables (optionally via a ``.env`` file), so the
application is not hard-coded around a single computer. Nothing secret is
ever sent to the frontend: the API exposes source *names* and *status*, not
keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency, no overwrite of real env vars)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- storage -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("MEBET_DATA_DIR", PROJECT_ROOT / "data")))
    database_url: str = os.environ.get("MEBET_DATABASE_URL", "")
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("MEBET_CACHE_DIR", PROJECT_ROOT / "data" / "http_cache")))
    log_dir: Path = field(default_factory=lambda: Path(os.environ.get("MEBET_LOG_DIR", PROJECT_ROOT / "logs")))

    # --- server --------------------------------------------------------
    host: str = os.environ.get("MEBET_HOST", "127.0.0.1")
    port: int = _env_int("MEBET_PORT", 8000)
    log_level: str = os.environ.get("MEBET_LOG_LEVEL", "INFO")

    # --- http acquisition ----------------------------------------------
    user_agent: str = os.environ.get(
        "MEBET_USER_AGENT",
        "mebet/0.1 (local sports analytics; contact: set MEBET_USER_AGENT)",
    )
    http_timeout: float = _env_float("MEBET_HTTP_TIMEOUT", 30.0)
    http_retries: int = _env_int("MEBET_HTTP_RETRIES", 3)
    cache_ttl_seconds: int = _env_int("MEBET_CACHE_TTL", 6 * 3600)
    respect_robots: bool = _env_bool("MEBET_RESPECT_ROBOTS", True)
    offline: bool = _env_bool("MEBET_OFFLINE", False)

    # --- source enablement / credentials -------------------------------
    # Absent key => the adapter reports itself unavailable. It never
    # substitutes invented data.
    football_data_org_key: str = os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
    api_football_key: str = os.environ.get("API_FOOTBALL_KEY", "")
    enabled_sources: str = os.environ.get("MEBET_ENABLED_SOURCES", "")

    # --- modelling ------------------------------------------------------
    # Half-life in days for time-decay weighting of historical matches.
    time_decay_half_life_days: float = _env_float("MEBET_HALF_LIFE_DAYS", 270.0)
    monte_carlo_iterations: int = _env_int("MEBET_MC_ITERATIONS", 20000)
    random_seed: int = _env_int("MEBET_RANDOM_SEED", 20240917)

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.data_dir / 'mebet.db'}"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.cache_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
