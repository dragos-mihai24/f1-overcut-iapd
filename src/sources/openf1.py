"""OpenF1 API (historical data, free, 2023+): tyre stints and race-control messages."""
from __future__ import annotations

import pandas as pd

from .common import cached_get_json

BASE = "https://api.openf1.org/v1"
INTERVAL = 2.5  # free tier: max 30 requests / minute
FIRST_SEASON = 2023


def race_sessions(season: int) -> pd.DataFrame:
    data = cached_get_json(f"{BASE}/sessions", {"year": season, "session_name": "Race"}, "openf1",
                           min_interval=INTERVAL, allow_404=True)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True)
    return df[["session_key", "date_start", "circuit_short_name", "country_name"]]


def session_key_for(season: int, start_utc: pd.Timestamp) -> int | None:
    """Match a Jolpica race to an OpenF1 race session by start date (same day)."""
    s = race_sessions(season)
    if s.empty:
        return None
    hit = s[(s["date_start"] - start_utc).abs() < pd.Timedelta(hours=12)]
    return int(hit.iloc[0]["session_key"]) if len(hit) else None


def stints(session_key: int) -> pd.DataFrame:
    data = cached_get_json(f"{BASE}/stints", {"session_key": session_key}, "openf1",
                           min_interval=INTERVAL, allow_404=True)
    return pd.DataFrame(data)


def safety_car_laps(session_key: int) -> pd.DataFrame:
    """Race-control messages of category SafetyCar (SC / VSC deployments), with lap numbers."""
    data = cached_get_json(f"{BASE}/race_control", {"session_key": session_key}, "openf1",
                           min_interval=INTERVAL, allow_404=True)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    return df[df["category"].isin(["SafetyCar"]) | df["flag"].isin(["RED"])][["lap_number", "category", "flag", "message"]]
