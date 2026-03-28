"""
plan_a_sdoh_funnel.py
Plan A — Access Friction Funnel
SDoH barriers (transport / financial / housing) vs Is_True_Abandonment_Risk
in ED / hospital encounters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats


BARRIERS = {
    "Has_Transport_Need":      "Transport Need",
    "Has_Financial_Strain":    "Financial Strain",
    "Has_Housing_Instability": "Housing Instability",
}

COLORS = {
    "present": "#d73027",
    "absent":  "#4575b4",
}


def run(data_dir: Path, output_dir: Path, processed_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading engineered data …")
    enc = pd.read_parquet(processed_dir / "encounters_engineered.parquet")

    # Filter to ED / hospital encounters
    ed_hosp = enc[enc["IsEdVisit_Flag"] | enc["IsHospitalAdmission_Flag"]].copy()
    ed_hosp["abandon_num"] = ed_hosp["Is_True_Abandonment_Risk"].astype(int)

    print(f"  Plan A — ED/hospital rows: {len(ed_hosp):,}")

    # =========================================================================
    # Plot 1: Grouped bar chart — abandonment rate by barrier (present/absent)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(BARRIERS))
    width = 0.35
    present_rates, absent_rates = [], []
    present_ns, absent_ns = [], []

    for col in BARRIERS:
        p = ed_hosp[ed_hosp[col] == True]["abandon_num"]
        a = ed_hosp[ed_hosp[col] == False]["abandon_num"]
        present_rates.append(p.mean() if len(p) > 0 else 0)
        absent_rates.append(a.mean() if len(a) > 0 else 0)
        present_ns.append(len(p))
        absent_ns.append(len(a))

    bars_p = ax.bar(x - width / 2, present_rates, width, color=COLORS["present"],
                    alpha=0.85, label="Barrier Present")
    bars_a = ax.bar(x + width / 2, absent_rates, width, color=COLORS["absent"],
                    alpha=0.85, label="No Barrier")

    for bar, rate, n in zip(bars_p, present_rates, present_ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    for bar, rate, n in zip(bars_a, absent_rates, absent_ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(list(BARRIERS.values()), fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Journey Abandonment Risk Rate", fontsize=12)
    ax.set_title("Plan A — SDoH Barriers & Journey Abandonment Risk\n(ED / Hospital encounters only)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(present_rates + absent_rates) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_a_sdoh_abandonment_bars.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 2: 2×2 heat-map grid — SDoH combination abandonment rate
    # =========================================================================
    combo_groups = (
        ed_hosp.groupby(
            ["Has_Transport_Need", "Has_Financial_Strain", "Has_Housing_Instability"]
        )["abandon_num"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "abandon_rate", "count": "n"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax_idx, housing in enumerate([False, True]):
        ax = axes[ax_idx]
        sub = combo_groups[combo_groups["Has_Housing_Instability"] == housing]

        pivot_rate = sub.pivot_table(
            index="Has_Transport_Need",
            columns="Has_Financial_Strain",
            values="abandon_rate",
            fill_value=0,
        )
        pivot_n = sub.pivot_table(
            index="Has_Transport_Need",
            columns="Has_Financial_Strain",
            values="n",
            fill_value=0,
        )

        im = ax.imshow(pivot_rate.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Financial\nStrain", "Financial\nStrain"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No Transport\nNeed", "Transport\nNeed"], fontsize=10)
        ax.set_title(
            f"Housing Instability: {'Yes' if housing else 'No'}", fontsize=12, fontweight="bold"
        )
        for i in range(2):
            for j in range(2):
                rate = pivot_rate.iloc[i, j] if (i < pivot_rate.shape[0] and j < pivot_rate.shape[1]) else 0
                n_val = pivot_n.iloc[i, j] if (i < pivot_n.shape[0] and j < pivot_n.shape[1]) else 0
                ax.text(j, i, f"{rate:.1%}\n(n={int(n_val):,})",
                        ha="center", va="center", fontsize=10, color="black")
        plt.colorbar(im, ax=ax, format="%.0%", label="Abandon Rate")

    fig.suptitle("Plan A — Abandonment Rate by SDoH Combination", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_dir / "plan_a_sdoh_combo_heatmap.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Plot 3: Odds ratio forest plot
    # =========================================================================
    try:
        from statsmodels.formula.api import logit as sm_logit

        model_df = ed_hosp[
            ["abandon_num", "Has_Transport_Need", "Has_Financial_Strain", "Has_Housing_Instability"]
        ].dropna()

        if len(model_df) > 100:
            formula = (
                "abandon_num ~ Has_Transport_Need + Has_Financial_Strain + Has_Housing_Instability"
            )
            result = sm_logit(formula, data=model_df.astype(
                {"Has_Transport_Need": int, "Has_Financial_Strain": int,
                 "Has_Housing_Instability": int}
            )).fit(disp=False)

            coef_names = {
                "Has_Transport_Need":      "Transport Need",
                "Has_Financial_Strain":    "Financial Strain",
                "Has_Housing_Instability": "Housing Instability",
            }
            params = result.params.drop("Intercept", errors="ignore")
            conf   = result.conf_int().drop("Intercept", errors="ignore")
            ors    = np.exp(params)
            ci_lo  = np.exp(conf[0])
            ci_hi  = np.exp(conf[1])

            labels = [coef_names.get(k, k) for k in params.index]

            fig, ax = plt.subplots(figsize=(9, 5))
            y_pos = np.arange(len(labels))
            ax.axvline(1, color="grey", linestyle="--", linewidth=1)
            ax.errorbar(ors.values, y_pos,
                        xerr=[ors.values - ci_lo.values, ci_hi.values - ors.values],
                        fmt="o", color="#d73027", markersize=9, linewidth=2, capsize=4)
            for i, (or_val, name) in enumerate(zip(ors.values, labels)):
                ax.text(or_val + 0.02, i, f"OR={or_val:.2f}", va="center", fontsize=10)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=12)
            ax.set_xlabel("Odds Ratio (95% CI)", fontsize=12)
            ax.set_title("Plan A — Logistic Regression Odds Ratios\nOutcome: Is_True_Abandonment_Risk",
                         fontsize=13, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(plot_dir / "plan_a_logistic_or.png", dpi=150)
            plt.close(fig)

    except Exception as exc:
        print(f"  [Plan A] Logistic regression skipped: {exc}")

    print("  Plan A plots saved.")
