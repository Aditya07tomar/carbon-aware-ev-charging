"""
Async HTTP clients for external carbon and weather data APIs.

Classes:
    WattTimeClient  — Marginal Operating Emissions Rate (MOER) from WattTime v3 API.
    OpenMeteoClient — Localized weather forecasts from Open-Meteo (free, no auth).

Both clients:
    • Use httpx.AsyncClient with connection pooling
    • Implement retry logic via tenacity (exponential backoff)
    • Return strictly-typed Pydantic models
    • Raise domain-specific exceptions on failure
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.schemas import (
    OpenMeteoWeatherResponse,
    WattTimeForecastResponse,
    WattTimeRealtimeResponse,
)

logger = logging.getLogger(__name__)

settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
#  Custom Exceptions
# ═══════════════════════════════════════════════════════════════════════════


class APIClientError(Exception):
    """Base exception for all API client errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class WattTimeError(APIClientError):
    """Raised when WattTime API returns an error or is unreachable."""
    pass


class OpenMeteoError(APIClientError):
    """Raised when Open-Meteo API returns an error or is unreachable."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  WattTime Client
# ═══════════════════════════════════════════════════════════════════════════


class WattTimeClient:
    """
    Async client for the WattTime v3 API.

    Authentication flow:
        1. POST /login with HTTP Basic credentials → receive a Bearer token.
        2. Use the token for subsequent requests until it expires (≈30 min).

    Endpoints used:
        • GET /v3/signal-index  — realtime MOER for a given region/location.
        • GET /v3/forecast      — 24-hour MOER forecast.

    Usage:
        async with WattTimeClient() as client:
            realtime = await client.get_realtime_signal(latitude=37.7749, longitude=-122.4194)
    """

    BASE_URL: str = "https://api.watttime.org"
    TOKEN_LIFETIME: timedelta = timedelta(minutes=29)  # refresh before expiry

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._username: str = username or settings.watttime_username
        self._password: str = password or settings.watttime_password
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    # ── Context Manager ─────────────────────────────────────────────────────

    async def __aenter__(self) -> WattTimeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Authentication ──────────────────────────────────────────────────────

    async def _ensure_token(self) -> str:
        """Authenticate or refresh the bearer token as needed."""
        now = datetime.now(timezone.utc)
        if self._token and self._token_expiry and now < self._token_expiry:
            return self._token

        logger.info("WattTime: Authenticating…")
        response = await self._client.get(
            "/login",
            auth=(self._username, self._password),
        )
        if response.status_code != 200:
            raise WattTimeError(
                f"Authentication failed: {response.text}",
                status_code=response.status_code,
            )

        data: dict[str, Any] = response.json()
        self._token = data["token"]
        self._token_expiry = now + self.TOKEN_LIFETIME
        logger.info("WattTime: Token acquired (expires %s)", self._token_expiry.isoformat())
        return self._token  # type: ignore[return-value]

    def _auth_headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    # ── Retry-Enabled Request ───────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated request with automatic retry."""
        token = await self._ensure_token()
        response = await self._client.request(
            method,
            path,
            params=params,
            headers=self._auth_headers(token),
        )
        if response.status_code == 401:
            # Token expired mid-flight — force refresh and retry once
            self._token = None
            token = await self._ensure_token()
            response = await self._client.request(
                method,
                path,
                params=params,
                headers=self._auth_headers(token),
            )

        if response.status_code != 200:
            raise WattTimeError(
                f"WattTime {path} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        result: dict[str, Any] = response.json()
        return result

    # ── Region Resolution ───────────────────────────────────────────────────

    async def get_region(
        self,
        *,
        latitude: float,
        longitude: float,
        signal_type: str = "co2_moer",
    ) -> str:
        """
        Resolve a (lat, lon) to a WattTime balancing authority region.

        The v3 API requires a region for signal-index and forecast endpoints.
        """
        try:
            data = await self._request(
                "GET",
                "/v3/region-from-loc",
                params={
                    "latitude": str(latitude),
                    "longitude": str(longitude),
                    "signal_type": signal_type,
                },
            )
            region = data.get("region", "")
            if not region:
                raise WattTimeError("No region returned for location")
            logger.info("WattTime: Resolved (%.4f, %.4f) → region=%s", latitude, longitude, region)
            return region
        except RetryError as exc:
            raise WattTimeError(f"Region lookup failed after retries: {exc}") from exc

    # ── Public Methods ──────────────────────────────────────────────────────

    async def get_realtime_signal(
        self,
        *,
        latitude: float,
        longitude: float,
        signal_type: str = "co2_moer",
        region: str | None = None,
    ) -> WattTimeRealtimeResponse:
        """
        Fetch the realtime marginal emissions signal for a location.

        Args:
            latitude:    Decimal latitude  (e.g. 37.7749).
            longitude:   Decimal longitude (e.g. -122.4194).
            signal_type: Signal type identifier (default: co2_moer).
            region:      WattTime region (auto-resolved from lat/lon if None).

        Returns:
            Parsed WattTimeRealtimeResponse with data points.
        """
        try:
            if not region:
                region = await self.get_region(
                    latitude=latitude, longitude=longitude, signal_type=signal_type
                )
            data = await self._request(
                "GET",
                "/v3/signal-index",
                params={
                    "region": region,
                    "signal_type": signal_type,
                },
            )
            return WattTimeRealtimeResponse.model_validate(data)
        except RetryError as exc:
            raise WattTimeError(f"Realtime signal fetch failed after retries: {exc}") from exc

    async def get_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        signal_type: str = "co2_moer",
        horizon_hours: int = 24,
        region: str | None = None,
    ) -> WattTimeForecastResponse:
        """
        Fetch the MOER forecast for a location.

        Args:
            latitude:      Decimal latitude.
            longitude:     Decimal longitude.
            signal_type:   Signal type identifier.
            horizon_hours: Forecast window in hours (default: 24).
            region:        WattTime region (auto-resolved from lat/lon if None).

        Returns:
            Parsed WattTimeForecastResponse with forecasted data points.
        """
        try:
            if not region:
                region = await self.get_region(
                    latitude=latitude, longitude=longitude, signal_type=signal_type
                )
            data = await self._request(
                "GET",
                "/v3/forecast",
                params={
                    "region": region,
                    "signal_type": signal_type,
                },
            )
            return WattTimeForecastResponse.model_validate(data)
        except RetryError as exc:
            raise WattTimeError(f"Forecast fetch failed after retries: {exc}") from exc

    async def get_historical(
        self,
        *,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
        signal_type: str = "co2_moer",
        region: str | None = None,
    ) -> WattTimeRealtimeResponse:
        """
        Fetch historical MOER data for a location.
        Requires WattTime API v3 /historical endpoint access.

        Args:
            latitude: Decimal latitude.
            longitude: Decimal longitude.
            start_time: Start datetime (timezone-aware).
            end_time: End datetime (timezone-aware).
            signal_type: Signal type (default: co2_moer).
            region: WattTime region (auto-resolved from lat/lon if None).
        """
        try:
            if not region:
                region = await self.get_region(
                    latitude=latitude, longitude=longitude, signal_type=signal_type
                )
            data = await self._request(
                "GET",
                "/v3/historical",
                params={
                    "region": region,
                    "signal_type": signal_type,
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
            )
            return WattTimeRealtimeResponse.model_validate(data)
        except RetryError as exc:
            raise WattTimeError(f"Historical fetch failed after retries: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════
#  Open-Meteo Client
# ═══════════════════════════════════════════════════════════════════════════


class OpenMeteoClient:
    """
    Async client for the Open-Meteo weather API (free, no auth required).

    Fetches hourly weather data relevant to solar generation estimates
    and EV battery performance (temperature, irradiance, cloud cover, wind).

    Usage:
        async with OpenMeteoClient() as client:
            weather = await client.get_weather_forecast(latitude=37.7749, longitude=-122.4194)
    """

    BASE_URL: str = "https://api.open-meteo.com"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    # ── Context Manager ─────────────────────────────────────────────────────

    async def __aenter__(self) -> OpenMeteoClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Retry-Enabled Request ───────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self,
        path: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a GET request with automatic retry."""
        response = await self._client.get(path, params=params)

        if response.status_code != 200:
            raise OpenMeteoError(
                f"Open-Meteo {path} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        result: dict[str, Any] = response.json()
        return result

    # ── Public Methods ──────────────────────────────────────────────────────

    async def get_weather_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        forecast_days: int = 2,
        timezone: str = "auto",
    ) -> OpenMeteoWeatherResponse:
        """
        Fetch hourly weather data for a location.

        Variables retrieved:
            • temperature_2m         — Air temperature at 2m (°C)
            • shortwave_radiation    — Solar irradiance (W/m²)
            • cloud_cover            — Total cloud cover (%)
            • wind_speed_10m         — Wind speed at 10m (m/s)

        Args:
            latitude:      Decimal latitude.
            longitude:     Decimal longitude.
            forecast_days: Number of forecast days (1–16, default: 2).
            timezone:      Timezone for timestamps (default: auto-detect).

        Returns:
            Parsed OpenMeteoWeatherResponse with hourly arrays.
        """
        hourly_vars: str = ",".join([
            "temperature_2m",
            "shortwave_radiation",
            "cloud_cover",
            "wind_speed_10m",
        ])

        try:
            data = await self._request(
                "/v1/forecast",
                params={
                    "latitude": str(latitude),
                    "longitude": str(longitude),
                    "hourly": hourly_vars,
                    "forecast_days": str(forecast_days),
                    "timezone": timezone,
                },
            )
            return OpenMeteoWeatherResponse.model_validate(data)
        except RetryError as exc:
            raise OpenMeteoError(f"Weather forecast fetch failed after retries: {exc}") from exc

    async def get_historical_weather(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone: str = "auto",
    ) -> OpenMeteoWeatherResponse:
        """
        Fetch historical hourly weather data for a date range.

        Args:
            latitude:   Decimal latitude.
            longitude:  Decimal longitude.
            start_date: ISO date string (YYYY-MM-DD).
            end_date:   ISO date string (YYYY-MM-DD).
            timezone:   Timezone for timestamps.

        Returns:
            Parsed OpenMeteoWeatherResponse with hourly arrays.
        """
        hourly_vars: str = ",".join([
            "temperature_2m",
            "shortwave_radiation",
            "cloud_cover",
            "wind_speed_10m",
        ])

        try:
            data = await self._request(
                "/v1/forecast",
                params={
                    "latitude": str(latitude),
                    "longitude": str(longitude),
                    "hourly": hourly_vars,
                    "start_date": start_date,
                    "end_date": end_date,
                    "timezone": timezone,
                },
            )
            return OpenMeteoWeatherResponse.model_validate(data)
        except RetryError as exc:
            raise OpenMeteoError(f"Historical weather fetch failed after retries: {exc}") from exc
