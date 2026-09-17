"""Source registry - the modular seam for adding or replacing providers."""

from __future__ import annotations

from typing import Iterable, Optional

from ..config import get_settings
from ..logging_setup import get_logger
from .base import Capability, SourceAdapter, SourceStatus

log = get_logger("sources.registry")

_REGISTRY: dict[str, type[SourceAdapter]] = {}


def register(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    """Class decorator that adds an adapter to the registry."""
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a key")
    _REGISTRY[cls.key] = cls
    return cls


def available_adapter_classes() -> dict[str, type[SourceAdapter]]:
    _load_builtin()
    return dict(_REGISTRY)


_LOADED = False


def _load_builtin() -> None:
    global _LOADED
    if _LOADED:
        return
    # Imported for their registration side effects.
    from . import (  # noqa: F401
        datahub_footballdata,
        footballdata_couk,
        fpl_api,
        fpl_mirror,
        open_meteo,
        openfootball,
    )

    _LOADED = True


def build_sources(keys: Optional[Iterable[str]] = None) -> list[SourceAdapter]:
    """Instantiate adapters, honouring MEBET_ENABLED_SOURCES when set."""
    _load_builtin()
    settings = get_settings()
    if keys is None:
        configured = [k.strip() for k in settings.enabled_sources.split(",") if k.strip()]
        keys = configured or list(_REGISTRY)
    out: list[SourceAdapter] = []
    for key in keys:
        cls = _REGISTRY.get(key)
        if cls is None:
            log.warning("unknown source key requested: %s", key)
            continue
        try:
            out.append(cls())
        except Exception as exc:  # pragma: no cover - adapter construction guard
            log.error("could not construct source %s: %s", key, exc)
    return out


def sources_for(capability: Capability, sport: str = "football",
                keys: Optional[Iterable[str]] = None) -> list[SourceAdapter]:
    """Adapters that can serve ``capability``, most reliable first."""
    sources = [s for s in build_sources(keys) if s.supports(capability, sport)]
    return sorted(sources, key=lambda s: s.reliability, reverse=True)


def status_report(keys: Optional[Iterable[str]] = None) -> list[SourceStatus]:
    return [s.status() for s in build_sources(keys)]
