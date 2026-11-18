"""Open-Meteo Historical Weather API: hourly reanalysis weather at the circuit."""
from __future__ import annotations

import pandas as pd

from .common import cached_get_json

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m"]


def race_weather(lat: float, lon: float, start_utc: pd.Timestamp) -> dict:
    """Weather features around the race start (race lasts ~1.5-2h)."""
    day = start_utc.strftime("%Y-%m-%d")
    prev = (start_utc - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = cached_get_json(URL, {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": prev, "end_date": day,
        "hourly": ",".join(HOURLY), "timezone": "UTC",
    }, "openmeteo", min_interval=0.2)
    h = pd.DataFrame(data["hourly"])
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize("UTC")
    t0 = start_utc.floor("h")
    during = h[(h["time"] >= t0) & (h["time"] <= t0 + pd.Timedelta(hours=2))]
    before = h[(h["time"] >= t0 - pd.Timedelta(hours=3)) & (h["time"] < t0)]
    return {
        "om_temp_c": during["temperature_2m"].mean(),
        "om_humidity": during["relative_humidity_2m"].mean(),
        "om_precip_race_mm": during["precipitation"].sum(),
        "om_precip_prev3h_mm": before["precipitation"].sum(),
        "om_cloud_cover": during["cloud_cover"].mean(),
        "om_wind_kmh": during["wind_speed_10m"].mean(),
    }
