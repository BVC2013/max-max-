"""
plan_c_continuity.py
Plan C — Continuity Breakdown (Provider Fragmentation)
Cumulative care transfers → ED escalation + longer encounter durations.
Kaplan-Meier time-to-ED-escalation by transfer stratum.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


TRANSFER_COLORS = {
    "0":  "#4575b4",
    "1":  "#fee090",
    "2":  "#f46d43",
    "3+": "#d73027",
}


def _transfer_bin(n: float) -> str:
    if pd.isna(n):
        return "0"
    n = int(n)
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def run(data_dir: Path, output_dir: Path, processed_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading engineered data …")
    enc = pd.read_parquet(processed_dir / "encounters_engineered.parquet")

    continuity = enc.dropna(subset=["Cumulative_Care_Transfers"]).copy()
    continuity["TransferBin"] = continuity["Cumulative_Care_Transfers"].apply(_transfer_bin)
    continuity["TransferBin"] = pd.Categorical(
        continuity["TransferBin"], categories=["0", "1", "2", "3+"], ordered=True
    )
    continuity["ed_num"] = continuity["IsEdVisit_Flag"].astype(int)

    print(f"  Plan C — rows: {len(continuity):,}")

    # =========================================================================
    # Plot 1: Dual-axis — ED rate + avg duration by transfer count
    # =========================================================================
    transfer_agg = (
        continuity.groupby("TransferBin", observed=True)
        .agg(
            n       =("EncounterKey", "count"),
            ed_rate =("ed_num", "mean"),
            avg_dur =("Encounter_Duration_Hours", "mean"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(9, 6))
    x = np.arange(len(transfer_agg))
    bars = ax1.bar(x, transfer_agg["ed_rate"],
                   color=[TRANSFER_COLORS[b] for b in transfer_agg["TransferBin"]],
                   alpha=0.85, width=0.55)
    for bar, rate, n in zip(bars, transfer_agg["ed_rate"], transfer_agg["n"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{rate:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(transfer_agg["TransferBin"], fontsize=12)
    ax1.set_xlabel("Cumulative Care Transfers", fontsize=12)
    ax1.set_ylabel("ED Escalation Rate", fontsize=12, color="#d73027")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.tick_params(axis="y", colors="#d73027")

    ax2 = ax1.twinx()
    ax2.plot(x, transfer_agg["avg_dur"], color="#1a9641", linewidth=2.2,
             marker="D", markersize=8, linestyle="--", label="Avg Duration")
    ax2.set_ylabel("Avg Encounter Duration (hours)", fontsize=12, color="#1a9641")
    ax2.tick_params(axis="y", colors="#1a9641")

    ax1.set_title("Plan C — Care Transfers vs ED Escalation & Duration",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_c_transfer_escalation.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 2: Ridge-style density — duration by transfer bin
    # (stacked KDE using filled plots)
    # =========================================================================
    from scipy.stats import gaussian_kde

    dur_clip = continuity["Encounter_Duration_Hours"].quantile(0.97)
    bins = ["0", "1", "2", "3+"]
    fig, ax = plt.subplots(figsize=(11, 6))

    x_range = np.linspace(0, dur_clip, 500)
    offset = 0
    for i, bkt in enumerate(reversed(bins)):
        subset = continuity[
            (continuity["TransferBin"] == bkt) &
            continuity["Encounter_Duration_Hours"].between(0, dur_clip)
        ]["Encounter_Duration_Hours"].dropna()
        if len(subset) < 20:
            continue
        kde = gaussian_kde(subset)
        density = kde(x_range)
        density = density / density.max() * 0.9
        ax.fill_between(x_range, offset, offset + density,
                        alpha=0.7, color=TRANSFER_COLORS[bkt], label=f"{bkt} transfers")
        ax.plot(x_range, offset + density, color=TRANSFER_COLORS[bkt], linewidth=1.5)
        ax.text(dur_clip * 1.01, offset + density.max() / 2, f"{bkt}",
                va="center", fontsize=11, color=TRANSFER_COLORS[bkt], fontweight="bold")
        offset += 1.0

    ax.set_xlabel("Encounter Duration (hours)", fontsize=12)
    ax.set_ylabel("Transfer Count Group", fontsize=12)
    ax.set_yticks([])
    ax.set_title("Plan C — Duration Distribution by Care Transfer Count\n"
                 "Higher transfers → rightward shift", fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_c_duration_ridges.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 3: Kaplan-Meier — time to ED escalation by transfer stratum
    # =========================================================================
    try:
        from lifelines import KaplanMeierFitter

        km_data = (
            enc.dropna(subset=["DiagnosisValue", "Cumulative_Days_In_Journey"])
            .groupby(["PatientDurableKey", "DiagnosisValue"])
            .apply(lambda g: pd.Series({
                "max_transfers":  g["Cumulative_Care_Transfers"].max(),
                "had_ed":         g["IsEdVisit_Flag"].any(),
                "days_to_event":  (
                    g.loc[g["IsEdVisit_Flag"], "Cumulative_Days_In_Journey"].min()
                    if g["IsEdVisit_Flag"].any()
                    else g["Cumulative_Days_In_Journey"].max()
                ),
            }))
            .reset_index()
        )

        km_data = km_data.dropna(subset=["days_to_event"])
        km_data = km_data[km_data["days_to_event"] >= 0]

        def _transfer_stratum(n):
            if pd.isna(n) or n == 0:
                return "0 transfers"
            if n == 1:
                return "1 transfer"
            return "2+ transfers"

        km_data["stratum"] = km_data["max_transfers"].apply(_transfer_stratum)

        stratum_colors = {
            "0 transfers":  "#4575b4",
            "1 transfer":   "#f46d43",
            "2+ transfers": "#d73027",
        }

        fig, ax = plt.subplots(figsize=(11, 7))
        for stratum, color in stratum_colors.items():
            sub = km_data[km_data["stratum"] == stratum]
            if len(sub) < 10:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["days_to_event"], event_observed=sub["had_ed"], label=stratum)
            kmf.plot_cumulative_density(ax=ax, color=color, linewidth=2.2, ci_show=True)

        ax.set_xlabel("Days in Journey", fontsize=12)
        ax.set_ylabel("Cumulative Probability of ED Escalation", fontsize=12)
        ax.set_title("Plan C — Cumulative ED Escalation by Transfer Stratum\n"
                     "Kaplan-Meier estimator", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_dir / "plan_c_km_ed_escalation.png", dpi=150)
        plt.close(fig)

    except Exception as exc:
        print(f"  [Plan C] KM plot skipped: {exc}")

    print("  Plan C plots saved.")
