"""
plan_d_geography.py
Plan D — Local Care Mismatch Map (Geography + Digital Engagement)
Block-group level dot-map of ED rate, abandonment risk, and MyChart inactivity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run(data_dir: Path, output_dir: Path, processed_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading engineered data …")
    enc = pd.read_parquet(processed_dir / "encounters_engineered.parquet")

    # =========================================================================
    # Block-group aggregation
    # =========================================================================
    print("  Aggregating block-group statistics …")

    has_geo = enc[
        enc["Geo_Data_Available"].fillna(False) &
        enc["CENTLAT"].notna() &
        enc["CENTLON"].notna()
    ].copy()

    block_agg = (
        has_geo.groupby(
            ["CensusBlockFipsCode_Clean", "CENTLAT", "CENTLON",
             "PopulationValue", "Is_Rural_Census_Block"],
            dropna=True,
        )
        .agg(
            n_encounters     =("EncounterKey", "count"),
            ed_rate          =("IsEdVisit_Flag", "mean"),
            abandon_rate     =("Is_True_Abandonment_Risk", "mean"),
            mychart_inactive =(
                "MyChart_Active",
                lambda s: 1 - s.fillna(False).mean()
            ),
        )
        .reset_index()
    )

    block_agg = block_agg[block_agg["n_encounters"] >= 5]
    print(f"  Block groups with sufficient data: {len(block_agg):,}")

    scaler = MinMaxScaler()
    block_agg["risk_score"] = scaler.fit_transform(
        block_agg[["ed_rate", "abandon_rate", "mychart_inactive"]]
    ).mean(axis=1)

    # =========================================================================
    # Plot 1: Kansas dot map
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f0f0f0")

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(
        vmin=block_agg["ed_rate"].min(),
        vmax=block_agg["ed_rate"].max(),
    )
    size_scale = 300 * (block_agg["abandon_rate"] + 0.01)
    alpha_vals = np.where(block_agg["Is_Rural_Census_Block"].fillna(False), 0.85, 0.4)

    scatter = ax.scatter(
        block_agg["CENTLON"],
        block_agg["CENTLAT"],
        c=block_agg["ed_rate"],
        s=size_scale,
        alpha=alpha_vals,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidths=0.3,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("ED Visit Rate", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(
        "Plan D — Kansas Block-Group Risk Map\n"
        "Color = ED rate | Size = abandonment risk | Opacity: Rural (opaque) vs Urban (translucent)",
        fontsize=13, fontweight="bold",
    )
    ax.set_facecolor("#dce9f5")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_d_kansas_dot_map.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 2: Composite risk — rural vs urban violin
    # =========================================================================
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 6))

    rural_scores = block_agg[block_agg["Is_Rural_Census_Block"].fillna(False)]["risk_score"].dropna()
    urban_scores = block_agg[~block_agg["Is_Rural_Census_Block"].fillna(False)]["risk_score"].dropna()

    x_range = np.linspace(0, 1, 300)
    for scores, color, label, pos in [
        (rural_scores, "#d73027", "Rural", 1),
        (urban_scores, "#4575b4", "Urban", 2),
    ]:
        if len(scores) < 5:
            continue
        kde = gaussian_kde(scores)
        density = kde(x_range)
        density = density / density.max() * 0.4
        ax.fill_betweenx(x_range, pos - density, pos + density,
                         alpha=0.65, color=color, label=label)
        ax.plot([pos - density, pos + density], [x_range, x_range], color=color, alpha=0.3)
        ax.plot([pos, pos], [np.percentile(scores, 25), np.percentile(scores, 75)],
                color="white", linewidth=4, solid_capstyle="round")
        ax.plot(pos, scores.mean(), "D", color="white", markersize=9, zorder=5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Rural", "Urban"], fontsize=13)
    ax.set_ylabel("Composite Risk Score (0–1)", fontsize=12)
    ax.set_title("Plan D — Composite Risk Score: Rural vs Urban\n"
                 "(avg of normalised ED rate, abandonment rate, MyChart inactivity)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_d_rural_urban_risk.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 3: Scatter — MyChart inactivity vs ED rate
    # =========================================================================

    fig, ax = plt.subplots(figsize=(11, 7))

    rural_mask = block_agg["Is_Rural_Census_Block"].fillna(False)
    colors = np.where(rural_mask, "#d73027", "#4575b4")
    sizes  = np.clip(block_agg["n_encounters"] / 10, 20, 400)

    sc = ax.scatter(
        block_agg["mychart_inactive"],
        block_agg["ed_rate"],
        c=colors,
        s=sizes,
        alpha=0.65,
        edgecolors="white",
        linewidths=0.4,
    )

    # OLS trendline
    valid = block_agg[["mychart_inactive", "ed_rate"]].dropna()
    if len(valid) > 5:
        coefs = np.polyfit(valid["mychart_inactive"], valid["ed_rate"], 1)
        x_line = np.linspace(valid["mychart_inactive"].min(), valid["mychart_inactive"].max(), 100)
        ax.plot(x_line, np.polyval(coefs, x_line), color="black", linewidth=1.8,
                linestyle="--", label="OLS trend")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d73027",
               markersize=10, label="Rural"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4575b4",
               markersize=10, label="Urban"),
        Line2D([0], [0], linestyle="--", color="black", label="OLS trend"),
    ]
    ax.legend(handles=legend_handles, fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("MyChart Inactivity Rate", fontsize=12)
    ax.set_ylabel("ED Visit Rate", fontsize=12)
    ax.set_title("Plan D — MyChart Inactivity vs ED Rate by Block Group\n"
                 "Dot size ∝ encounter volume", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_d_mychart_vs_ed_scatter.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 4: Top-10 highest-risk block groups — horizontal bar
    # =========================================================================
    top10 = block_agg.nlargest(10, "risk_score").copy()
    top10["label"] = (
        top10["CENTLAT"].round(3).astype(str) + "°N, " +
        top10["CENTLON"].abs().round(3).astype(str) + "°W"
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = ["#d73027" if r else "#4575b4"
                  for r in top10["Is_Rural_Census_Block"].fillna(False)]
    bars = ax.barh(top10["label"], top10["risk_score"], color=bar_colors, alpha=0.85)
    for bar, n in zip(bars, top10["n_encounters"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={n:,}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d73027", alpha=0.85, label="Rural"),
        Patch(facecolor="#4575b4", alpha=0.85, label="Urban"),
    ]
    ax.legend(handles=legend_handles, fontsize=11)
    ax.set_xlabel("Composite Risk Score (0–1)", fontsize=12)
    ax.set_title("Plan D — Top 10 Highest-Risk Block Groups",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_d_top10_clusters.png", dpi=150)
    plt.close(fig)

    print("  Plan D plots saved.")
