"""Phase 1 exploratory data analysis: descriptive statistics + figures.

Usage:  python -m src.eda
Writes reports/phase1/figures/*.png and reports/phase1/descriptive_stats.csv
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .features import POST_HOC, PRE_DECISION_NUMERIC, TARGET, TYRE_2023  # noqa: E402
from .sources.common import ROOT, load_config  # noqa: E402

BLUE, ORANGE, GREY = "#2a6fdb", "#e8702a", "#8a8f98"


def rate_by_bins(d: pd.DataFrame, col: str, bins) -> pd.DataFrame:
    b = pd.cut(d[col], bins)
    g = d.groupby(b, observed=True)[TARGET].agg(["mean", "size"])
    return g


def main() -> None:
    cfg = load_config()
    out = ROOT / "reports" / "phase1"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)

    all_d = pd.read_csv(ROOT / cfg["paths"]["processed"] / "duels.csv")
    d = all_d[all_d["neutralized"] == 0]
    print(f"duels: {len(all_d)} total, {len(d)} green-flag; features: "
          f"{len(PRE_DECISION_NUMERIC)} pre-decision + {len(TYRE_2023)} tyre (2023+) + {len(POST_HOC)} post-hoc")

    # 1. descriptive statistics (central tendency + dispersion, course 02)
    num = PRE_DECISION_NUMERIC + TYRE_2023 + POST_HOC + ["gap_after_s"]
    desc = all_d[num].describe().T
    desc["median"] = all_d[num].median()
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc["missing_%"] = all_d[num].isna().mean() * 100
    desc["skew"] = all_d[num].skew()
    desc.round(3).to_csv(out / "descriptive_stats.csv")

    # 2. duels per season + class balance
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    s = all_d.groupby(["season", "neutralized"]).size().unstack(fill_value=0)
    s.plot.bar(stacked=True, ax=ax[0], color=[BLUE, GREY], width=0.75)
    ax[0].legend(["green flag", "SC / VSC"], frameon=False)
    ax[0].set(title="Duels per season", xlabel="", ylabel="duels")
    cb = d.groupby("a_was_ahead")[TARGET].mean()
    ax[1].bar(["A behind B\n(undercut attack)", "A ahead of B\n(cover stop)"], cb.values, color=[ORANGE, BLUE])
    ax[1].axhline(d[TARGET].mean(), color=GREY, ls="--", lw=1)
    for i, v in enumerate(cb.values):
        ax[1].text(i, v + 0.02, f"{v:.0%}", ha="center")
    ax[1].set(title="First stopper ends up ahead", ylim=(0, 1), ylabel="success rate")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_volume_and_balance.png", dpi=150)

    # 3. success rate vs key numeric drivers (attack duels only: A behind B)
    att = d[d["a_was_ahead"] == 0]
    specs = [("gap_s", [0, 1, 2, 3, 4, 5, 6], "gap before stop (s)"),
             ("stint_age_delta", [-60, -10, -5, 0, 5, 10, 60], "stint age A − B (laps)"),
             ("pace_delta", [-5, -0.5, -0.2, 0, 0.2, 0.5, 5], "pace A − B (s/lap)"),
             ("lap_delay_B", [0, 1, 2, 3, 5, 10], "laps B stays out (post-hoc)")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, (col, bins, label) in zip(axs, specs):
        g = rate_by_bins(att, col, bins)
        ax.bar(range(len(g)), g["mean"], color=ORANGE)
        ax.set_xticks(range(len(g)), [str(i) for i in g.index], rotation=35, fontsize=7)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=7)
        ax.set(title=label, ylim=(0, 1))
    axs[0].set_ylabel("undercut success rate")
    fig.suptitle("Undercut attacks (A starts behind B): success rate by feature", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_success_by_feature.png", dpi=150, bbox_inches="tight")

    # 4. by circuit
    c = att.groupby("circuit_id")[TARGET].agg(["mean", "size"]).query("size >= 10").sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(c))))
    ax.barh(c.index, c["mean"], color=ORANGE)
    ax.axvline(att[TARGET].mean(), color=GREY, ls="--", lw=1)
    ax.set(title="Undercut success rate by circuit (≥10 attacks)", xlabel="success rate", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "03_success_by_circuit.png", dpi=150)

    # 5. missing values
    miss = all_d[num].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(miss) + 1))
        ax.barh(miss.index, miss.values * 100, color=GREY)
        ax.set(title="Missing values (%)", xlabel="% of duels")
        fig.tight_layout()
        fig.savefig(fig_dir / "04_missing_values.png", dpi=150)

    # 6. correlation (redundancy check)
    corr = d[PRE_DECISION_NUMERIC + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.6},
                xticklabels=True, yticklabels=True)
    ax.set_title("Pearson correlation, pre-decision features + target")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_correlation.png", dpi=150)

    # 7. outliers in pit-lane time (IQR rule)
    pl = all_d["pit_lane_A"].dropna()
    q1, q3 = pl.quantile([0.25, 0.75])
    hi = q3 + 1.5 * (q3 - q1)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    sns.boxplot(x=pl, ax=ax, color=BLUE, fliersize=2)
    ax.set(title=f"Pit-lane time of A (IQR outliers > {hi:.1f}s: {(pl > hi).mean():.1%})", xlabel="s")
    fig.tight_layout()
    fig.savefig(fig_dir / "06_pit_lane_outliers.png", dpi=150)

    print("success rate overall:", round(d[TARGET].mean(), 3),
          "| attacks:", round(att[TARGET].mean(), 3), f"(n={len(att)})",
          "| covers:", round(d.loc[d["a_was_ahead"] == 1, TARGET].mean(), 3))
    print("top |corr| with target:")
    print(corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(8).round(3).to_string())
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
