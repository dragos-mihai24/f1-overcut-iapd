"""Download (and cache) all raw data needed for the dataset.

Usage:  python -m src.fetch_raw
The Jolpica download is slow (~2,300 requests at <= 480/hour) but resumable:
every response is cached under data/cache/, so re-running skips what is done.
"""
from __future__ import annotations

import sys
import time

from .sources import jolpica, openf1
from .sources.common import load_config
from .sources.openmeteo import race_weather


def main() -> None:
    cfg = load_config()
    seasons = cfg["seasons"]
    # most recent seasons first: they have OpenF1 tyre data and are used for testing
    seasons = sorted(seasons, reverse=True)
    sched = jolpica.get_schedules(seasons)
    t0 = time.time()
    for i, r in enumerate(sched.itertuples(), 1):
        try:
            race_weather(r.lat, r.lon, r.start_utc)
            if r.season >= openf1.FIRST_SEASON:
                sk = openf1.session_key_for(r.season, r.start_utc)
                if sk:
                    openf1.stints(sk)
                    openf1.safety_car_laps(sk)
            jolpica.get_results(r.season, r.round)
            jolpica.get_pitstops(r.season, r.round)
            laps = jolpica.get_laps(r.season, r.round)
            print(f"[{i}/{len(sched)}] {r.season} R{r.round:02d} {r.race_name}: {len(laps)} lap rows "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        except Exception as exc:  # keep going; a re-run retries failures
            print(f"[{i}/{len(sched)}] {r.season} R{r.round:02d} FAILED: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
