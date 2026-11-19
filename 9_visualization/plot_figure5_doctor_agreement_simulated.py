from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

DATA_FILE = (
    ROOT /
    "9_results/AAA_morphology_dataset_v2.csv"
)

OUT_DIR = (
    ROOT /
    "9_results/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SIM_DATA_OUT = (
    ROOT /
    "9_results/Figure5_simulated_doctor_measurements.csv"
)


# ============================================================
# Publication style
# ============================================================

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Times",
    "DejaVu Serif"
]

plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.8

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# Load AI measurements
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


# Only AAA cases for clinical aneurysm diameter agreement
df = (
    df[
        df["class"] == "AAA"
    ]
    .copy()
    .sort_values("case_id")
    .reset_index(drop=True)
)


ai = df[
    "diameter_mm"
].to_numpy(
    dtype=float
)


# ============================================================
# Simulate two blinded radiologists
# ============================================================
# IMPORTANT:
# These values are ONLY for visual/layout preview.
# They must be replaced by actual manual measurements
# before manuscript submission.
# ============================================================

rng = np.random.default_rng(
    20260816
)


doctor1 = (
    ai
    +
    0.8
    +
    rng.normal(
        loc=0.0,
        scale=2.1,
        size=len(ai)
    )
)


doctor2 = (
    ai
    -
    0.4
    +
    rng.normal(
        loc=0.0,
        scale=2.3,
        size=len(ai)
    )
)


manual_mean = (
    doctor1 + doctor2
) / 2.0


# ============================================================
# Save simulated table
# ============================================================

sim_df = pd.DataFrame(
    {
        "case_id":
            df["case_id"],

        "AI_diameter_mm":
            ai,

        "Doctor1_diameter_mm":
            doctor1,

        "Doctor2_diameter_mm":
            doctor2,

        "Manual_mean_mm":
            manual_mean
    }
)


sim_df.to_csv(
    SIM_DATA_OUT,
    index=False
)


print("\nSIMULATED DATA:")
print(
    sim_df.round(2).to_string(
        index=False
    )
)


# ============================================================
# Statistics
# ============================================================

def agreement_stats(
        reference,
        comparison
):

    r, p = pearsonr(
        reference,
        comparison
    )

    diff = (
        reference -
        comparison
    )

    mae = np.mean(
        np.abs(diff)
    )

    rmse = np.sqrt(
        np.mean(
            diff ** 2
        )
    )

    bias = np.mean(
        diff
    )

    sd = np.std(
        diff,
        ddof=1
    )

    loa_low = (
        bias - 1.96 * sd
    )

    loa_high = (
        bias + 1.96 * sd
    )

    return {
        "r": r,
        "p": p,
        "MAE": mae,
        "RMSE": rmse,
        "bias": bias,
        "LoA_low": loa_low,
        "LoA_high": loa_high
    }


stats_d1 = agreement_stats(
    ai,
    doctor1
)

stats_d2 = agreement_stats(
    ai,
    doctor2
)

stats_mean = agreement_stats(
    ai,
    manual_mean
)


# ============================================================
# Scatter panel
# ============================================================

def draw_scatter(
        ax,
        x,
        y,
        xlabel,
        ylabel,
        stats,
        panel_label
):

    ax.scatter(
        x,
        y,
        s=42,
        alpha=0.78,
        edgecolor="black",
        linewidth=0.45
    )


    # identity line
    low = min(
        np.min(x),
        np.min(y)
    ) - 3

    high = max(
        np.max(x),
        np.max(y)
    ) + 3


    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1.0,
        color="0.35",
        label="Identity line"
    )


    # regression line
    coef = np.polyfit(
        x,
        y,
        1
    )

    xx = np.linspace(
        low,
        high,
        100
    )

    yy = (
        coef[0] * xx
        +
        coef[1]
    )


    ax.plot(
        xx,
        yy,
        linewidth=1.3,
        color="#B9473D",
        label="Linear fit"
    )


    ax.set_xlim(
        low,
        high
    )

    ax.set_ylim(
        low,
        high
    )


    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )


    ax.grid(
        linestyle="--",
        linewidth=0.4,
        alpha=0.30
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    ax.text(
        0.04,
        0.95,
        (
            f"r = {stats['r']:.3f}\n"
            f"MAE = {stats['MAE']:.2f} mm\n"
            f"RMSE = {stats['RMSE']:.2f} mm"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5
    )


    ax.text(
        -0.13,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )


# ============================================================
# Bland-Altman panel
# ============================================================

def draw_bland_altman(
        ax,
        ai_measurement,
        manual_measurement,
        stats,
        panel_label
):

    means = (
        ai_measurement
        +
        manual_measurement
    ) / 2.0


    differences = (
        ai_measurement
        -
        manual_measurement
    )


    ax.scatter(
        means,
        differences,
        s=42,
        alpha=0.78,
        edgecolor="black",
        linewidth=0.45
    )


    bias = stats[
        "bias"
    ]

    loa_low = stats[
        "LoA_low"
    ]

    loa_high = stats[
        "LoA_high"
    ]


    ax.axhline(
        bias,
        linewidth=1.2,
        color="#B9473D"
    )

    ax.axhline(
        loa_high,
        linewidth=1.0,
        linestyle="--",
        color="0.35"
    )

    ax.axhline(
        loa_low,
        linewidth=1.0,
        linestyle="--",
        color="0.35"
    )


    x_right = (
        np.max(means)
        +
        0.5
    )


    ax.text(
        x_right,
        bias,
        f"Bias = {bias:.2f}",
        fontsize=8,
        va="bottom"
    )

    ax.text(
        x_right,
        loa_high,
        f"+1.96 SD = {loa_high:.2f}",
        fontsize=8,
        va="bottom"
    )

    ax.text(
        x_right,
        loa_low,
        f"-1.96 SD = {loa_low:.2f}",
        fontsize=8,
        va="top"
    )


    ax.set_xlabel(
        "Mean of AI and manual diameter (mm)"
    )

    ax.set_ylabel(
        "AI - manual diameter (mm)"
    )


    ax.grid(
        linestyle="--",
        linewidth=0.4,
        alpha=0.30
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    ax.text(
        -0.13,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        9.5,
        7.5
    ),
    dpi=300
)


# A
draw_scatter(
    axes[0, 0],
    ai,
    doctor1,
    "AI-derived maximum diameter (mm)",
    "Doctor 1 maximum diameter (mm)",
    stats_d1,
    "(A)"
)


# B
draw_scatter(
    axes[0, 1],
    ai,
    doctor2,
    "AI-derived maximum diameter (mm)",
    "Doctor 2 maximum diameter (mm)",
    stats_d2,
    "(B)"
)


# C
draw_scatter(
    axes[1, 0],
    ai,
    manual_mean,
    "AI-derived maximum diameter (mm)",
    "Mean manual maximum diameter (mm)",
    stats_mean,
    "(C)"
)


# D
draw_bland_altman(
    axes[1, 1],
    ai,
    manual_mean,
    stats_mean,
    "(D)"
)


# ============================================================
# Watermark for simulated version
# ============================================================

fig.text(
    0.5,
    0.505,
    "SIMULATED DATA — LAYOUT PREVIEW",
    ha="center",
    va="center",
    fontsize=21,
    color="0.72",
    alpha=0.40,
    rotation=27,
    fontweight="bold"
)


plt.subplots_adjust(
    left=0.10,
    right=0.965,
    top=0.955,
    bottom=0.09,
    wspace=0.30,
    hspace=0.32
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Figure5_doctor_agreement_SIMULATED"
)


PNG = BASE.with_suffix(
    ".png"
)

SVG = BASE.with_suffix(
    ".svg"
)

PDF = BASE.with_suffix(
    ".pdf"
)


fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.04
)


fig.savefig(
    SVG,
    format="svg",
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.04
)


fig.savefig(
    PDF,
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.04
)


print("\n============================================")
print("SIMULATED Figure 5 completed")
print("============================================")

print(PNG)
print(SVG)
print(PDF)

print("\nSimulated table:")
print(SIM_DATA_OUT)

print("\nAI vs mean manual:")
print(
    f"r     = {stats_mean['r']:.4f}"
)

print(
    f"MAE   = {stats_mean['MAE']:.4f} mm"
)

print(
    f"RMSE  = {stats_mean['RMSE']:.4f} mm"
)

print(
    f"Bias  = {stats_mean['bias']:.4f} mm"
)

print(
    f"95% LoA = "
    f"[{stats_mean['LoA_low']:.4f}, "
    f"{stats_mean['LoA_high']:.4f}] mm"
)


plt.close(
    fig
)

