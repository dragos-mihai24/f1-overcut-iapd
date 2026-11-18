"""Extraction of strategy duels (undercut vs overcut) from one race.

Inputs per race (see src/sources): Jolpica lap times + positions, Jolpica pit stops,
Jolpica results, and (2023+) OpenF1 tyre stints.

A duel (A, B):
  * driver A pits at the end of lap L_A (in-lap); B was within `max_gap_s` of A at the end
    of lap r = L_A - 1
  * B does not pit on L_A, has not pitted in the previous `min_laps_since_b_stop` laps,
    and pits on lap L_B with L_A < L_B <= L_A + max_lap_delay
  * A does not pit again before L_B + 1, and both complete lap e = L_B + 1
Label: undercut_success = 1 if A is ahead of B at the end of lap e (cumulative race time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


class Race:
    """(driver, lap) lookups over one race."""

    def __init__(self, laps: pd.DataFrame, pits: pd.DataFrame, results: pd.DataFrame,
                 stints: pd.DataFrame | None, neutral_ratio: float):
        laps = laps.sort_values(["driver", "lap"]).copy()
        # cumulative time is only valid while laps are contiguous: cut each driver at the first gap
        contiguous = laps["lap"] == laps.groupby("driver").cumcount() + 1
        self.n_gap_rows = int((~contiguous).sum())
        laps = laps[contiguous.groupby(laps["driver"]).cummin()]
        laps["t"] = laps.groupby("driver")["lap_time"].cumsum()  # race time at end of lap
        self.laps = laps
        self.T = {(d, l): t for d, l, t in zip(laps["driver"], laps["lap"], laps["t"])}
        self.LT = {(d, l): t for d, l, t in zip(laps["driver"], laps["lap"], laps["lap_time"])}
        self.P = {(d, l): p for d, l, p in zip(laps["driver"], laps["lap"], laps["position"])}
        self.drivers = sorted(laps["driver"].unique())
        self.crossings = {d: g["t"].to_numpy() for d, g in laps.groupby("driver")}
        self.total_laps = int(results["laps_completed"].max())
        self.team = dict(zip(results["driver"], results["team"]))
        # pit stops: in-lap numbers; ignore "stops" on the final lap (retirements into the pits)
        self.stops = {d: [] for d in self.drivers}
        self.pit_lane = {}
        for p in pits.itertuples():
            if p.driver in self.stops and (p.driver, p.lap + 1) in self.T:
                self.stops[p.driver].append(p.lap)
                self.pit_lane[(p.driver, p.lap)] = p.pit_lane_s
        for d in self.stops:
            self.stops[d] = sorted(set(self.stops[d]))
        self.pit_laps = {(d, l) for d, ls in self.stops.items() for L in ls for l in (L, L + 1)}
        self._detect_neutralized(neutral_ratio)
        self._tyres(results, stints)

    # --- safety car / VSC detection from lap times ---------------------------------
    def _detect_neutralized(self, ratio: float) -> None:
        """A lap is neutralized (SC / VSC / red flag) if the median lap time of the field,
        excluding in/out laps, is more than `ratio` x the race's typical green lap."""
        clean = self.laps[[(d, l) not in self.pit_laps for d, l in zip(self.laps["driver"], self.laps["lap"])]]
        med = clean.groupby("lap")["lap_time"].median()
        # laps where the whole field pitted (e.g. SC through the pit lane): use all cars
        med = med.combine_first(self.laps.groupby("lap")["lap_time"].median()).sort_index()
        ref = med.loc[med.index > 1].quantile(0.25)  # green-flag pace (robust to many SC laps)
        self.lap_median = med
        self.neutral_laps = set(med.index[(med > ratio * ref) & (med.index > 1)])

    # --- tyres --------------------------------------------------------------------
    def _tyres(self, results: pd.DataFrame, stints: pd.DataFrame | None) -> None:
        """Compound + true tyre age from OpenF1 stints (2023+), else unknown."""
        self.tyre = {}
        if stints is None or stints.empty:
            return
        num2drv = dict(zip(results["number"], results["driver"]))
        for s in stints.itertuples():
            d = num2drv.get(s.driver_number)
            if d is None or pd.isna(s.lap_start) or pd.isna(s.lap_end):
                continue
            age0 = 0 if pd.isna(s.tyre_age_at_start) else int(s.tyre_age_at_start)
            for lap in range(int(s.lap_start), int(s.lap_end) + 1):
                self.tyre[(d, lap)] = (s.compound, age0 + lap - int(s.lap_start) + 1)

    def compound(self, d: str, lap: int):
        return self.tyre.get((d, lap), (np.nan, np.nan))[0]

    def true_tyre_age(self, d: str, lap: int):
        return self.tyre.get((d, lap), (np.nan, np.nan))[1]

    def stint_age(self, d: str, lap: int) -> int:
        """Laps driven on the current set (since the last stop or the start)."""
        prev = [s for s in self.stops[d] if s < lap]
        return lap - (prev[-1] if prev else 0)

    # --- helpers --------------------------------------------------------------------
    def t(self, d: str, lap: int) -> float:
        return self.T.get((d, lap), np.nan)

    def clean_laptimes(self, d: str, laps) -> list[float]:
        return [self.LT[(d, l)] for l in laps
                if l > 1 and (d, l) in self.LT and (d, l) not in self.pit_laps and l not in self.neutral_laps]


def race_pit_loss(rc: Race) -> float:
    """Median time lost by a green-flag stop: (in-lap + out-lap) - 2 x typical clean lap."""
    losses = []
    for d, stops in rc.stops.items():
        for L in stops:
            if L in rc.neutral_laps or (L + 1) in rc.neutral_laps:
                continue
            t0, t2 = rc.t(d, L - 1), rc.t(d, L + 1)
            ref = rc.clean_laptimes(d, range(L - 5, L)) + rc.clean_laptimes(d, range(L + 2, L + 7))
            if not np.isnan(t0) and not np.isnan(t2) and len(ref) >= 3:
                losses.append((t2 - t0) - 2 * np.median(ref))
    losses = [x for x in losses if 5 < x < 60]  # drop penalties / broken stops
    return float(np.median(losses)) if losses else np.nan


def race_overtakes_per_lap(rc: Race) -> float:
    """On-track position swaps per green lap between cars that did not pit (overtaking ease)."""
    pos = rc.laps.pivot_table(index="lap", columns="driver", values="position")
    swaps, n = 0, 0
    for lap in pos.index[1:]:
        if lap + 1 not in pos.index or (lap + 1) in rc.neutral_laps:
            continue
        a, b = pos.loc[lap], pos.loc[lap + 1]
        ok = [d for d in pos.columns if pd.notna(a[d]) and pd.notna(b[d])
              and (d, lap) not in rc.pit_laps and (d, lap + 1) not in rc.pit_laps]
        if len(ok) < 5:
            continue
        av, bv = a[ok].to_numpy(), b[ok].to_numpy()
        flips = np.sign(av[:, None] - av[None, :]) * np.sign(bv[:, None] - bv[None, :]) < 0
        swaps += int(flips.sum() // 2)
        n += 1
    return swaps / n if n else np.nan


def extract_duels(laps, pits, results, stints, cfg) -> tuple[pd.DataFrame, dict]:
    dc, tc = cfg["duel"], cfg["traffic"]
    rc = Race(laps, pits, results, stints, dc["neutral_ratio"])
    pit_loss = race_pit_loss(rc)
    race_info = {
        "pit_loss_race": pit_loss,
        "overtakes_per_lap_race": race_overtakes_per_lap(rc),
        "total_laps": rc.total_laps,
        "n_neutral_laps": len(rc.neutral_laps),
        "n_lap_rows_dropped": rc.n_gap_rows,
        "n_stops_race": sum(len(s) for s in rc.stops.values()),
        "race_wet_tyres": int(any(c in WET_COMPOUNDS for c, _ in rc.tyre.values())) if rc.tyre else np.nan,
    }
    rows = []
    for A in rc.drivers:
        for LA in rc.stops[A]:
            r = LA - 1
            tA = rc.t(A, r)
            if r < 1 or np.isnan(tA):
                continue
            # traffic at A's rejoin: g = how far behind A each other car crosses the line next
            g = []
            for C in rc.drivers:
                if C == A or LA in rc.stops[C]:
                    continue
                later = rc.crossings[C][rc.crossings[C] >= tA]
                if len(later):
                    g.append(later.min() - tA)
            g = np.array(g)
            traffic = rejoin_gap = np.nan
            if not np.isnan(pit_loss) and len(g):
                traffic = int(((g >= pit_loss - tc["window_ahead_s"]) & (g <= pit_loss + tc["window_behind_s"])).sum())
                ahead = pit_loss - g
                ahead = ahead[ahead >= 0]
                rejoin_gap = float(ahead.min()) if len(ahead) else np.nan

            for B in rc.drivers:
                if B == A:
                    continue
                tB = rc.t(B, r)
                if np.isnan(tB):
                    continue
                gap = tB - tA  # > 0: B behind A
                if abs(gap) > dc["max_gap_s"] or LA in rc.stops[B]:
                    continue
                later = [s for s in rc.stops[B] if s > LA]
                if not later or later[0] - LA > dc["max_lap_delay"]:
                    continue
                LB = later[0]
                if any(LA - dc["min_laps_since_b_stop"] <= s < LA for s in rc.stops[B]):
                    continue
                if any(LA < s <= LB + 1 for s in rc.stops[A]):
                    continue
                e = LB + 1
                tAe, tBe = rc.t(A, e), rc.t(B, e)
                if np.isnan(tAe) or np.isnan(tBe):
                    continue
                win = range(r - dc["pace_window"] + 1, r + 1)
                pa, pb = rc.clean_laptimes(A, win), rc.clean_laptimes(B, win)
                rows.append({
                    "driver_A": A, "driver_B": B, "team_A": rc.team.get(A), "team_B": rc.team.get(B),
                    "lap_A": LA, "lap_B": LB,
                    # --- pre-decision features (known when A pits) ---
                    "gap_s": abs(gap),
                    "a_was_ahead": int(gap > 0),
                    "position_A": rc.P.get((A, r)),
                    "same_team": int(rc.team.get(A) == rc.team.get(B)),
                    "stop_number_A": rc.stops[A].index(LA) + 1,
                    "stint_age_A": rc.stint_age(A, r), "stint_age_B": rc.stint_age(B, r),
                    "tyre_age_A": rc.true_tyre_age(A, r), "tyre_age_B": rc.true_tyre_age(B, r),
                    "cmp_A_old": rc.compound(A, r), "cmp_B_old": rc.compound(B, r),
                    "cmp_A_new": rc.compound(A, LA + 1),
                    "pace_A": np.median(pa) if len(pa) >= 2 else np.nan,
                    "pace_B": np.median(pb) if len(pb) >= 2 else np.nan,
                    "traffic_at_exit": traffic,
                    "rejoin_gap_ahead": rejoin_gap,
                    "laps_remaining": rc.total_laps - LA,
                    "race_progress": LA / rc.total_laps,
                    "neutral_before": int(any(l in rc.neutral_laps for l in range(r - 2, r + 1))),
                    # --- post-hoc information (not known when A pits) ---
                    "lap_delay_B": LB - LA,
                    "pit_lane_A": rc.pit_lane.get((A, LA), np.nan),
                    "pit_lane_B": rc.pit_lane.get((B, LB), np.nan),
                    "cmp_B_new": rc.compound(B, LB + 1),
                    "neutralized": int(any(l in rc.neutral_laps for l in range(LA, e + 1))),
                    # --- label ---
                    "gap_after_s": tBe - tAe,
                    "undercut_success": int(tAe < tBe),
                })
    df = pd.DataFrame(rows)
    if len(df):
        df["pace_delta"] = df["pace_A"] - df["pace_B"]
    return df, race_info
