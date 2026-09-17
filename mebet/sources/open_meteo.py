"""Adapter: Open-Meteo weather.

No API key, generous free tier, and - importantly for backtesting - a
historical archive endpoint, so a past match can be given the weather that
was actually recorded rather than today's forecast.
"""

from __future__ import annotations

import datetime as dt

from ..logging_setup import get_logger
from ..normalize import ConditionsRecord
from .base import Capability, SourceAdapter, SourceResponse, SourceStatus
from .registry import register

log = get_logger("sources.open_meteo")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"


@register
class OpenMeteoAdapter(SourceAdapter):
    key = "open_meteo"
    name = "Open-Meteo"
    homepage = "https://open-meteo.com"
    license = "CC BY 4.0, free for non-commercial use"
    capabilities = (Capability.WEATHER,)
    sports = ("football", "basketball", "tennis", "cricket", "baseball")
    reliability = 0.8
    min_interval = 1.0

    def status(self) -> SourceStatus:
        res = self.fetch(
            FORECAST_URL, params={"latitude": 51.5, "longitude": -0.12, "hourly": "temperature_2m"},
            ttl=3600,
        )
        return SourceStatus(
            key=self.key, available=res.ok,
            detail="reachable" if res.ok else f"unreachable: {res.error}",
        )

    def fetch_weather(self, *, latitude: float, longitude: float,
                      when: dt.datetime) -> SourceResponse:
        when_utc = when.astimezone(dt.timezone.utc) if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)
        now = dt.datetime.now(dt.timezone.utc)
        historical = when_utc < now - dt.timedelta(days=2)
        url = ARCHIVE_URL if historical else FORECAST_URL
        day = when_utc.date().isoformat()
        params = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "hourly": HOURLY,
            "start_date": day,
            "end_date": day,
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        res = self.fetch(url, params=params, ttl=3 * 3600)
        if not res.ok:
            return SourceResponse.failure(self.key, f"weather unavailable: {res.error}", [url])
        try:
            payload = res.json()
            hourly = payload.get("hourly") or {}
            times = hourly.get("time") or []
            if not times:
                return SourceResponse.failure(self.key, "weather response contained no hourly data", [url])
            target = when_utc.strftime("%Y-%m-%dT%H:00")
            idx = times.index(target) if target in times else min(
                range(len(times)),
                key=lambda i: abs(dt.datetime.fromisoformat(times[i]).replace(tzinfo=dt.timezone.utc) - when_utc),
            )

            def at(name):
                series = hourly.get(name) or []
                return series[idx] if idx < len(series) else None

            record = ConditionsRecord(
                temperature_c=at("temperature_2m"),
                humidity_pct=at("relative_humidity_2m"),
                precipitation_mm=at("precipitation"),
                wind_kph=at("wind_speed_10m"),
                description="archived observation" if historical else "forecast",
                raw={"endpoint": "archive" if historical else "forecast", "hour": times[idx]},
                source_key=self.key,
                retrieved_at=res.fetched_at,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return SourceResponse.failure(self.key, f"unexpected weather payload: {exc}", [url])

        return SourceResponse(
            source_key=self.key, ok=True, records=[record], urls=[url],
            retrieved_at=res.fetched_at, from_cache=res.from_cache,
        )
