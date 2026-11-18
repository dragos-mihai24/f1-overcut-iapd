"""Build the strategy-duel dataset from the cached raw data.

Usage:  python -m src.build_dataset [--offline]
  --offline  use only races already downloaded by src.fetch_raw (no network calls)
Outputs:  data/processed/duels.csv  (one row per duel)
          data/processed/races.csv  (one row per race, race-level statistics)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .duels import extract_duels
from .sources import common, jolpica, openf1
from .sources.common import ROOT, load_config
from .sources.openmeteo import race_weather
from .sources.wikipedia_circuits import match_circuits

HARDNESS = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}


def leave_one_season_out(races: pd.DataFrame, col: str) -> pd.Series:
    """Mean of `col` at the same circuit in *other* seasons (fallback: the race itself).
    Avoids using the race's own outcome-related statistics as a feature."""
    out = []
    for r in races.itertuples():
        same = races[(races["circuit_id"] == r.circuit_id) & (races["season"] != r.season)][col].dropna()
        out.append(same.mean() if len(same) else getattr(r, col))
    return pd.Series(out, index=races.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    common.OFFLINE = args.offline
    cfg = load_config()

    sched = jolpica.get_schedules(cfg["seasons"])
    circuits = match_circuits(sched)
    duels, races = [], []
    for r in sched.itertuples():
        try:
            laps = jolpica.get_laps(r.season, r.round)
            pits = jolpica.get_pitstops(r.season, r.round)
            results = jolpica.get_results(r.season, r.round)
            stints = None
            if r.season >= openf1.FIRST_SEASON:
                sk = openf1.session_key_for(r.season, r.start_utc)
                stints = openf1.stints(sk) if sk else None
            weather = race_weather(r.lat, r.lon, r.start_utc)
        except common.CacheMiss:
            continue
        if laps.empty:
            print(f"{r.season} R{r.round:02d}: no lap data", file=sys.stderr)
            continue
        df, info = extract_duels(laps, pits, results, stints, cfg)
        meta = {"season": r.season, "round": r.round, "race_name": r.race_name, "circuit_id": r.circuit_id}
        races.append({**meta, **info, **weather, "n_duels": len(df)})
        if len(df):
            duels.append(df.assign(**meta, **weather))
        print(f"{r.season} R{r.round:02d} {r.race_name:<28} duels={len(df):3d}  "
              f"pit_loss={info['pit_loss_race']:.1f}s  neutral_laps={info['n_neutral_laps']}", flush=True)

    races = pd.DataFrame(races)
    races["pit_loss_circuit"] = leave_one_season_out(races, "pit_loss_race")
    races["overtakes_per_lap_circuit"] = leave_one_season_out(races, "overtakes_per_lap_race")
    d = pd.concat(duels, ignore_index=True)
    d = d.merge(races[["season", "round", "pit_loss_circuit", "overtakes_per_lap_circuit", "total_laps",
                       "race_wet_tyres"]], on=["season", "round"], how="left")
    d = d.merge(circuits.drop(columns=["wiki_match"]), on="circuit_id", how="left")

    # derived features
    d["stint_age_delta"] = d["stint_age_A"] - d["stint_age_B"]
    d["tyre_age_delta"] = d["tyre_age_A"] - d["tyre_age_B"]
    for c in ["cmp_A_old", "cmp_B_old", "cmp_A_new", "cmp_B_new"]:
        d[c.replace("cmp_", "hard_")] = d[c].map(HARDNESS)
    d["rain_race"] = (d["om_precip_race_mm"] > 0.2).astype(int)
    d["era_ground_effect"] = (d["season"] >= 2022).astype(int)
    d["duel_id"] = (d["season"].astype(str) + "_" + d["round"].astype(str).str.zfill(2) + "_" + d["driver_A"]
                    + "_" + d["lap_A"].astype(str) + "_" + d["driver_B"])

    first = ["duel_id", "season", "round", "race_name", "circuit_id", "driver_A", "driver_B", "team_A",
             "team_B", "lap_A", "lap_B"]
    d = d[first + [c for c in d.columns if c not in first]]
    out = ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "duels.csv", index=False)
    races.to_csv(out / "races.csv", index=False)
    print(f"\n{len(d)} duels from {races['n_duels'].gt(0).sum()} races "
          f"({len(races)} races processed) -> {out / 'duels.csv'}")
    print("success rate:", round(d["undercut_success"].mean(), 3),
          "| neutralized:", int(d["neutralized"].sum()))


if __name__ == "__main__":
    main()
