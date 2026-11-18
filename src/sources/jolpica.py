"""Jolpica-F1 API (successor of Ergast): calendar, circuits, results, lap times, pit stops.

Rate limit: 4 req/s burst, 500 req/hour sustained -> we keep >= 7.5 s between network calls.
"""
from __future__ import annotations

import pandas as pd

from .common import cached_get_json

BASE = "https://api.jolpi.ca/ergast/f1"
INTERVAL = 7.5
PAGE = 100


def _get(path: str, params: dict | None = None) -> dict:
    return cached_get_json(f"{BASE}/{path}", params, "jolpica", min_interval=INTERVAL)["MRData"]


def _to_seconds(s: str) -> float:
    """'1:37.284' -> 97.284"""
    parts = s.split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def get_schedule(season: int) -> pd.DataFrame:
    """One row per Grand Prix: round, name, circuit id/name/coords, UTC start time."""
    data = _get(f"{season}/races.json", {"limit": 100})
    rows = []
    for race in data["RaceTable"]["Races"]:
        c = race["Circuit"]
        start = pd.Timestamp(f"{race['date']}T{race.get('time', '12:00:00Z')}")
        rows.append({
            "season": season,
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "circuit_id": c["circuitId"],
            "circuit_name": c["circuitName"],
            "circuit_url": c.get("url", ""),
            "lat": float(c["Location"]["lat"]),
            "lon": float(c["Location"]["long"]),
            "country": c["Location"]["country"],
            "start_utc": start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC"),
        })
    return pd.DataFrame(rows)


def get_schedules(seasons: list[int]) -> pd.DataFrame:
    return pd.concat([get_schedule(s) for s in seasons], ignore_index=True)


def get_results(season: int, rnd: int) -> pd.DataFrame:
    data = _get(f"{season}/{rnd}/results.json", {"limit": 100})
    rows = []
    for x in data["RaceTable"]["Races"][0]["Results"]:
        rows.append({
            "driver": x["Driver"]["driverId"],
            "code": x["Driver"].get("code", x["Driver"]["driverId"][:3].upper()),
            "number": int(x["number"]),
            "team": x["Constructor"]["constructorId"],
            "grid": int(x["grid"]),
            "finish_position": int(x["position"]),
            "laps_completed": int(x["laps"]),
            "status": x["status"],
        })
    return pd.DataFrame(rows)


def get_laps(season: int, rnd: int) -> pd.DataFrame:
    """Lap times of every driver: one row per (driver, lap) with lap time (s) and position."""
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        data = _get(f"{season}/{rnd}/laps.json", {"limit": PAGE, "offset": offset})
        total = int(data["total"])
        races = data["RaceTable"]["Races"]
        for lap in (races[0]["Laps"] if races else []):
            for t in lap["Timings"]:
                if "time" not in t:
                    continue
                pos = t.get("position")
                rows.append({"driver": t["driverId"], "lap": int(lap["number"]),
                             "position": int(pos) if pos else float("nan"),
                             "lap_time": _to_seconds(t["time"])})
        offset += PAGE
    return pd.DataFrame(rows)


def get_pitstops(season: int, rnd: int) -> pd.DataFrame:
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        data = _get(f"{season}/{rnd}/pitstops.json", {"limit": PAGE, "offset": offset})
        total = int(data["total"])
        races = data["RaceTable"]["Races"]
        for p in (races[0].get("PitStops", []) if races else []):
            try:
                dur = _to_seconds(p["duration"])
            except (ValueError, KeyError):
                dur = float("nan")
            rows.append({"driver": p["driverId"], "lap": int(p["lap"]), "stop": int(p["stop"]),
                         "pit_lane_s": dur})
        offset += PAGE
    return pd.DataFrame(rows, columns=["driver", "lap", "stop", "pit_lane_s"])
