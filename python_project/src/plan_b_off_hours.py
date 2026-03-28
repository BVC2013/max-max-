"""
plan_b_off_hours.py
Plan B — Off-Hours Throughput Strain
ED visit rate and encounter duration by hour-of-arrival and day-of-week.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


HOUR_BUCKET_COLORS = {
    "Night (0–6)":    "#d73027",
    "Day (7–16)":     "#4575b4",
    "Evening (17–23)":"#f46d43",
}


def _hour_bucket(h: float) -> str:
    if h < 7:
        return "Night (0–6)"
    if h < 17:
        return "Day (7–16)"
    return "Evening (17–23)"


def run(data_dir: Path, output_dir: Path, processed_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading engineered data …")
    enc = pd.read_parquet(processed_dir / "encounters_engineered.parquet")

    hourly = enc.dropna(subset=["AdmitHour_num"]).copy()
    hourly["HourBucket"] = hourly["AdmitHour_num"].apply(_hour_bucket)
    hourly["HourBucket"] = pd.Categorical(
        hourly["HourBucket"],
        categories=["Night (0–6)", "Day (7–16)", "Evening (17–23)"],
        ordered=True,
    )

    print(f"  Plan B — rows with admit hour: {len(hourly):,}")

    # =========================================================================
    # Plot 1: Dual-axis — ED rate (bars) + avg duration (line) by hour
    # =========================================================================
    hourly_agg = (
        hourly.groupby("AdmitHour_num")
        .agg(
            n         =("EncounterKey", "count"),
            ed_rate   =("IsEdVisit_Flag", "mean"),
            avg_dur   =("Encounter_Duration_Hours", "mean"),
            is_off    =("Is_Off_Hours", "first"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(14, 6))
    colors = ["#d73027" if off else "#4575b4" for off in hourly_agg["is_off"]]
    bars = ax1.bar(hourly_agg["AdmitHour_num"], hourly_agg["ed_rate"],
                   color=colors, alpha=0.8, width=0.8)

    ax1.add_patch(plt.Rectangle((-0.5, 0), 7, 1, color="#d73027", alpha=0.08, zorder=0))
    ax1.add_patch(plt.Rectangle((16.5, 0), 7, 1, color="#d73027", alpha=0.08, zorder=0))

    ax1.set_xlabel("Admit Hour (0–23)", fontsize=12)
    ax1.set_ylabel("ED Visit Rate", fontsize=12, color="#4575b4")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xticks(range(24))

    ax2 = ax1.twinx()
    ax2.plot(hourly_agg["AdmitHour_num"], hourly_agg["avg_dur"],
             color="#1a9641", linewidth=2.2, marker="o", markersize=5, label="Avg Duration")
    ax2.set_ylabel("Avg Encounter Duration (hours)", fontsize=12, color="#1a9641")
    ax2.tick_params(axis="y", colors="#1a9641")

    patch_off = mpatches.Patch(color="#d73027", alpha=0.6, label="Off-Hours")
    patch_reg = mpatches.Patch(color="#4575b4", alpha=0.6, label="Regular Hours")
    line_dur  = plt.Line2D([0], [0], color="#1a9641", linewidth=2, label="Avg Duration")
    ax1.legend(handles=[patch_off, patch_reg, line_dur], loc="upper left", fontsize=10)

    ax1.set_title("Plan B — ED Rate & Encounter Duration by Hour of Arrival\n"
                  "Red shading = off-hours (0–6 | 17–23)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_b_hourly_dual_axis.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 2: Violin / box — duration by hour bucket
    # =========================================================================
    dur_df = hourly.dropna(subset=["Encounter_Duration_Hours"])
    dur_df = dur_df[dur_df["Encounter_Duration_Hours"] > 0]
    clip_99 = dur_df["Encounter_Duration_Hours"].quantile(0.99)
    dur_df  = dur_df[dur_df["Encounter_Duration_Hours"] <= clip_99]

    buckets      = ["Night (0–6)", "Day (7–16)", "Evening (17–23)"]
    bucket_data  = [dur_df[dur_df["HourBucket"] == b]["Encounter_Duration_Hours"].values
                    for b in buckets]
    bucket_means = [d.mean() for d in bucket_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(bucket_data, positions=[1, 2, 3], showmedians=False,
                          showextrema=False)
    for i, (pc, bkt) in enumerate(zip(parts["bodies"], buckets)):
        pc.set_facecolor(HOUR_BUCKET_COLORS[bkt])
        pc.set_alpha(0.65)

    ax.boxplot(bucket_data, positions=[1, 2, 3], widths=0.12, patch_artist=True,
               boxprops=dict(facecolor="white", alpha=0.8),
               medianprops=dict(color="black", linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5),
               flierprops=dict(markersize=0))

    for i, (pos, mean) in enumerate(zip([1, 2, 3], bucket_means)):
        ax.plot(pos, mean, "D", color="black", markersize=7, zorder=5)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(buckets, fontsize=12)
    ax.set_ylabel("Encounter Duration (hours)", fontsize=12)
    ax.set_title("Plan B — Duration Distribution by Arrival Time Bucket\n"
                 "Diamond = mean | Box = IQR | Clipped at 99th percentile",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_b_duration_violin.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 3: Heat-map — day-of-week × hour → ED rate
    # =========================================================================
    hourly["DayOfWeek"] = pd.to_datetime(
        hourly["Date"], errors="coerce"
    ).dt.day_name()

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_hour_agg = (
        hourly.groupby(["DayOfWeek", "AdmitHour_num"])["IsEdVisit_Flag"]
        .mean()
        .unstack(fill_value=0)
    )
    dow_hour_agg = dow_hour_agg.reindex([d for d in dow_order if d in dow_hour_agg.index])

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(dow_hour_agg.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=dow_hour_agg.values.max())
    plt.colorbar(im, ax=ax, label="ED Visit Rate", format="%.0%")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=9)
    ax.set_yticks(range(len(dow_hour_agg.index)))
    ax.set_yticklabels(dow_hour_agg.index, fontsize=10)
    ax.set_xlabel("Admit Hour", fontsize=12)
    ax.set_title("Plan B — ED Rate Heat-Map by Day-of-Week & Hour\n"
                 "Darker = higher proportion of ED encounters", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_b_dow_hour_heatmap.png", dpi=150)
    plt.close(fig)

    print("  Plan B plots saved.")
