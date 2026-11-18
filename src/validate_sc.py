"""Validate the lap-time based SC/VSC detector against OpenF1 race-control messages (2023+).

Usage:  python -m src.validate_sc
For every 2023+ race: official neutralized laps = from 'SAFETY CAR / VIRTUAL SAFETY CAR DEPLOYED'
to 'SAFETY CAR IN THIS LAP' / 'VSC ENDING' (+ red-flag laps). Reports lap-level precision/recall.
"""
from __future__ import annotations

import pandas as pd

from .duels import Race
from .sources import common, jolpica, openf1
from .sources.common import ROOT, load_config


def official_neutral_laps(msgs: pd.DataFrame) -> set[int]:
    laps, start = set(), None
    for m in msgs.sort_values("lap_number").itertuples():
        text = str(m.message).upper()
        if pd.isna(m.lap_number):
            continue
        lap = int(m.lap_number)
        if "DEPLOYED" in text:
            start = lap if start is None else start
        elif ("IN THIS LAP" in text or "ENDING" in text) and start is not None:
            laps.update(range(start, lap + 1))
            start = None
        elif m.flag == "RED":
            laps.add(lap)
    return laps


def main() -> None:
    common.OFFLINE = True
    cfg = load_config()
    seasons = [s for s in cfg["seasons"] if s >= openf1.FIRST_SEASON]
    sched = jolpica.get_schedules(seasons)
    tp = fp = fn = tn = 0
    for r in sched.itertuples():
        try:
            laps = jolpica.get_laps(r.season, r.round)
            pits = jolpica.get_pitstops(r.season, r.round)
            res = jolpica.get_results(r.season, r.round)
            sk = openf1.session_key_for(r.season, r.start_utc)
            msgs = openf1.safety_car_laps(sk) if sk else pd.DataFrame()
        except common.CacheMiss:
            continue
        if laps.empty or msgs.empty and sk is None:
            continue
        rc = Race(laps, pits, res, None, cfg["duel"]["neutral_ratio"])
        official = official_neutral_laps(msgs) if not msgs.empty else set()
        for lap in range(2, rc.total_laps + 1):
            pred, true = lap in rc.neutral_laps, lap in official
            tp += pred and true
            fp += pred and not true
            fn += true and not pred
            tn += not pred and not true
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    print(f"laps: {tp + tn + fp + fn}  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={prec:.3f} recall={rec:.3f} accuracy={acc:.3f}")
    out = ROOT / "reports" / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"laps": tp + tn + fp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                   "precision": prec, "recall": rec, "accuracy": acc}]).round(3) \
        .to_csv(out / "sc_detector_validation.csv", index=False)


if __name__ == "__main__":
    main()
