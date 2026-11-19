from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kruskal
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

CSV = (
    ROOT /
    "9_results/morphology_risk_v2/"
    "AAA100_morphology_risk_score_v2.csv"
)

OUT_DIR = (
    ROOT /
    "9_results/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STAT_OUT = (
    ROOT /
    "9_results/morphology_risk_v2/"
    "Figure6_risk_group_statistics.csv"
)


# ============================================================
# Style
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
# Load
# ============================================================

df = pd.read_csv(CSV)

required = [
    "case_id",
    "max_feret_diameter_mm",
    "max_section_area_mm2",
    "mesh_volume_mm3",
    "tortuosity",
    "sphericity",
    "morphology_risk_score_v2",
    "morphology_risk_group_v2"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns: " + ", ".join(missing)
    )


# ============================================================
# Group order
# ============================================================

GROUPS = [
    "Low",
    "Intermediate",
    "High"
]

group_colors = {
    "Low": "#9EC5E5",
    "Intermediate": "#E6C66A",
    "High": "#D97870"
}


# ============================================================
# Statistics
# ============================================================

stat_rows = []

for feature in [
    "max_feret_diameter_mm",
    "mesh_volume_mm3",
    "tortuosity",
    "sphericity",
    "max_section_area_mm2"
]:

    arrays = [
        df.loc[
            df["morphology_risk_group_v2"] == g,
            feature
        ].values
        for g in GROUPS
    ]

    h, p = kruskal(*arrays)

    stat_rows.append({
        "feature": feature,
        "kruskal_H": h,
        "p_value": p
    })

stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(STAT_OUT, index=False)

print(stats_df)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.5, 8.0),
    dpi=300
)

axes = axes.flatten()


# ============================================================
# A — ranked morphology risk score
# ============================================================

ax = axes[0]

ranked = df.sort_values(
    "morphology_risk_score_v2",
    ascending=True
).reset_index(drop=True)

colors = [
    group_colors[g]
    for g in ranked["morphology_risk_group_v2"]
]

ax.bar(
    np.arange(len(ranked)),
    ranked["morphology_risk_score_v2"],
    color=colors,
    width=0.9,
    edgecolor="none"
)

ax.set_xlabel(
    "AAA cases ranked by morphology-based risk score"
)

ax.set_ylabel(
    "Morphology risk score"
)

ax.set_ylim(
    0,
    105
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.4,
    alpha=0.25
)

ax.text(
    -0.10,
    1.04,
    "(A)",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold"
)

from matplotlib.patches import Patch

legend_handles = [
    Patch(
        facecolor=group_colors[g],
        edgecolor="none",
        label=g
    )
    for g in GROUPS
]

ax.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper left",
    ncol=3,
    fontsize=8
)


# ============================================================
# B — diameter by morphology-risk group
# ============================================================

ax = axes[1]

diameter_data = [
    df.loc[
        df["morphology_risk_group_v2"] == g,
        "max_feret_diameter_mm"
    ].values
    for g in GROUPS
]

bp = ax.boxplot(
    diameter_data,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(
        color="black",
        linewidth=1.1
    )
)

for patch, g in zip(
    bp["boxes"],
    GROUPS
):
    patch.set_facecolor(
        group_colors[g]
    )
    patch.set_alpha(0.75)

rng = np.random.default_rng(2026)

for i, values in enumerate(
    diameter_data,
    start=1
):

    jitter = rng.normal(
        0,
        0.05,
        len(values)
    )

    ax.scatter(
        np.full(len(values), i) + jitter,
        values,
        s=18,
        alpha=0.65,
        edgecolors="none"
    )

ax.set_xticks(
    [1, 2, 3]
)

ax.set_xticklabels(
    GROUPS
)

ax.set_ylabel(
    "Maximum Feret diameter (mm)"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.4,
    alpha=0.25
)

p_diameter = stats_df.loc[
    stats_df["feature"] == "max_feret_diameter_mm",
    "p_value"
].iloc[0]

if p_diameter < 0.001:
    p_text = "P < 0.001"
else:
    p_text = f"P = {p_diameter:.3f}"

ax.text(
    0.97,
    0.95,
    p_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8.5
)

ax.text(
    -0.10,
    1.04,
    "(B)",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# C — standardized morphology profile
# ============================================================

ax = axes[2]

profile_features = [
    "max_feret_diameter_mm",
    "mesh_volume_mm3",
    "tortuosity",
    "sphericity"
]

profile_labels = [
    "Diameter",
    "Volume",
    "Tortuosity",
    "Sphericity"
]

scaled = StandardScaler().fit_transform(
    df[profile_features]
)

scaled_df = pd.DataFrame(
    scaled,
    columns=profile_labels
)

scaled_df["group"] = (
    df["morphology_risk_group_v2"].values
)

profile_mean = (
    scaled_df
    .groupby("group")[profile_labels]
    .mean()
    .reindex(GROUPS)
)

x = np.arange(
    len(profile_labels)
)

width = 0.24

for idx, g in enumerate(GROUPS):

    ax.bar(
        x + (idx - 1) * width,
        profile_mean.loc[g].values,
        width=width,
        color=group_colors[g],
        alpha=0.8,
        label=g
    )

ax.axhline(
    0,
    color="0.4",
    linewidth=0.8
)

ax.set_xticks(x)

ax.set_xticklabels(
    profile_labels,
    rotation=20,
    ha="right"
)

ax.set_ylabel(
    "Standardized morphology (z-score)"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.4,
    alpha=0.25
)

ax.legend(
    frameon=False,
    fontsize=8
)

ax.text(
    -0.10,
    1.04,
    "(C)",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# D — diameter vs morphology risk score
# ============================================================

ax = axes[3]

x = df[
    "max_feret_diameter_mm"
].values

y = df[
    "morphology_risk_score_v2"
].values

for g in GROUPS:

    sub = df[
        df["morphology_risk_group_v2"] == g
    ]

    ax.scatter(
        sub["max_feret_diameter_mm"],
        sub["morphology_risk_score_v2"],
        s=30,
        alpha=0.75,
        color=group_colors[g],
        edgecolor="black",
        linewidth=0.3,
        label=g
    )

rho, p = spearmanr(
    x,
    y
)

coef = np.polyfit(
    x,
    y,
    1
)

xx = np.linspace(
    np.min(x),
    np.max(x),
    200
)

yy = (
    coef[0] * xx
    +
    coef[1]
)

ax.plot(
    xx,
    yy,
    color="#555555",
    linewidth=1.2
)

ax.set_xlabel(
    "Maximum Feret diameter (mm)"
)

ax.set_ylabel(
    "Morphology risk score"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    linestyle="--",
    linewidth=0.4,
    alpha=0.25
)

if p < 0.001:
    p_text = "P < 0.001"
else:
    p_text = f"P = {p:.3f}"

ax.text(
    0.05,
    0.95,
    (
        f"Spearman ρ = {rho:.3f}\n"
        f"{p_text}"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.5
)

ax.legend(
    frameon=False,
    fontsize=8,
    loc="lower right"
)

ax.text(
    -0.10,
    1.04,
    "(D)",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.09,
    right=0.97,
    top=0.96,
    bottom=0.10,
    wspace=0.30,
    hspace=0.34
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Figure6_morphology_risk_stratification_final"
)

PNG = BASE.with_suffix(".png")
SVG = BASE.with_suffix(".svg")
PDF = BASE.with_suffix(".pdf")

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
print("Figure 6 completed")
print("============================================")
print(PNG)
print(SVG)
print(PDF)
print("\nStatistics:")
print(STAT_OUT)

plt.close(fig)

