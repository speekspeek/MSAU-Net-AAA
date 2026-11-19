from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

CSV = (
    ROOT /
    "9_results/true_centerline_diameter/"
    "AAA_true_centerline_diameter_v3.csv"
)

OUT_DIR = (
    ROOT /
    "9_results/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CORR_OUT = (
    ROOT /
    "9_results/true_centerline_diameter/"
    "Figure5_morphology_spearman_correlation.csv"
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
# Load
# ============================================================

df = pd.read_csv(
    CSV
)

print("=" * 80)
print("FIGURE 5 — AUTOMATED MORPHOLOGICAL ASSESSMENT")
print("=" * 80)

print(
    "Cases:",
    len(df)
)


# ============================================================
# Required columns
# ============================================================

required = [
    "case_id",
    "max_feret_diameter_mm",
    "max_section_area_mm2",
    "max_section_equivalent_diameter_mm",
    "centerline_length_mm",
    "tortuosity",
    "mesh_volume_mm3",
    "sphericity"
]


missing = [
    x
    for x in required
    if x not in df.columns
]


if missing:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing)
    )


# ============================================================
# Derived units
# ============================================================

df["mesh_volume_ml"] = (
    df["mesh_volume_mm3"]
    /
    1000.0
)


# ============================================================
# Summary print
# ============================================================

summary_cols = [
    "max_feret_diameter_mm",
    "max_section_area_mm2",
    "max_section_equivalent_diameter_mm",
    "centerline_length_mm",
    "tortuosity",
    "mesh_volume_ml",
    "sphericity"
]


print("\nSummary:")
print(
    df[
        summary_cols
    ]
    .describe()
    .T
    .round(3)
)


# ============================================================
# Helper plotting functions
# ============================================================

def style_axis(ax):

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

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.4,
        alpha=0.25
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )


def panel_label(
        ax,
        label
):

    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )


def draw_distribution(
        ax,
        values,
        xlabel,
        panel,
        bins=12
):

    values = np.asarray(
        values,
        dtype=float
    )

    ax.hist(
        values,
        bins=bins,
        alpha=0.72,
        edgecolor="black",
        linewidth=0.55
    )

    mean = np.mean(
        values
    )

    median = np.median(
        values
    )

    ax.axvline(
        mean,
        linestyle="-",
        linewidth=1.3,
        color="#B7473C"
    )

    ax.axvline(
        median,
        linestyle="--",
        linewidth=1.15,
        color="#3F6FA8"
    )

    ax.set_xlabel(
        xlabel,
        fontsize=9
    )

    ax.set_ylabel(
        "Number of cases",
        fontsize=9
    )

    ax.text(
        0.97,
        0.94,
        (
            f"Mean = {mean:.2f}\n"
            f"Median = {median:.2f}\n"
            f"N = {len(values)}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8
    )

    style_axis(
        ax
    )

    panel_label(
        ax,
        panel
    )


# ============================================================
# Spearman correlation matrix
# ============================================================

corr_features = {
    "Diameter":
        "max_feret_diameter_mm",

    "Area":
        "max_section_area_mm2",

    "Volume":
        "mesh_volume_mm3",

    "Length":
        "centerline_length_mm",

    "Tortuosity":
        "tortuosity",

    "Sphericity":
        "sphericity"
}


corr_labels = list(
    corr_features.keys()
)

corr_cols = list(
    corr_features.values()
)


corr_matrix = np.zeros(
    (
        len(corr_cols),
        len(corr_cols)
    ),
    dtype=float
)


p_matrix = np.zeros_like(
    corr_matrix
)


for i, col_i in enumerate(
        corr_cols
):

    for j, col_j in enumerate(
        corr_cols
    ):

        rho, p = spearmanr(
            df[col_i],
            df[col_j]
        )

        corr_matrix[
            i,
            j
        ] = rho

        p_matrix[
            i,
            j
        ] = p


corr_df = pd.DataFrame(
    corr_matrix,
    index=corr_labels,
    columns=corr_labels
)


corr_df.to_csv(
    CORR_OUT
)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        11.5,
        7.4
    ),
    dpi=300
)


axes = axes.flatten()


# ============================================================
# A — Maximum Feret diameter
# ============================================================

draw_distribution(
    axes[0],
    df[
        "max_feret_diameter_mm"
    ],
    "Maximum Feret diameter (mm)",
    "(A)",
    bins=12
)


# ============================================================
# B — Volume
# ============================================================

draw_distribution(
    axes[1],
    df[
        "mesh_volume_ml"
    ],
    "AAA volume (mL)",
    "(B)",
    bins=12
)


# ============================================================
# C — Tortuosity
# ============================================================

draw_distribution(
    axes[2],
    df[
        "tortuosity"
    ],
    "Centerline tortuosity",
    "(C)",
    bins=12
)


# ============================================================
# D — Sphericity
# ============================================================

draw_distribution(
    axes[3],
    df[
        "sphericity"
    ],
    "Sphericity",
    "(D)",
    bins=12
)


# ============================================================
# E — Diameter vs Volume
# ============================================================

ax = axes[4]


x = df[
    "max_feret_diameter_mm"
].to_numpy(
    dtype=float
)

y = df[
    "mesh_volume_ml"
].to_numpy(
    dtype=float
)


rho, p = spearmanr(
    x,
    y
)


ax.scatter(
    x,
    y,
    s=30,
    alpha=0.72,
    edgecolor="black",
    linewidth=0.35
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
    linewidth=1.3,
    color="#B7473C"
)


ax.set_xlabel(
    "Maximum Feret diameter (mm)",
    fontsize=9
)

ax.set_ylabel(
    "AAA volume (mL)",
    fontsize=9
)


if p < 0.001:

    p_text = "P < 0.001"

else:

    p_text = (
        f"P = {p:.3f}"
    )


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


style_axis(
    ax
)

panel_label(
    ax,
    "(E)"
)


# ============================================================
# F — Correlation heatmap
# ============================================================

ax = axes[5]


im = ax.imshow(
    corr_matrix,
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    aspect="equal"
)


ax.set_xticks(
    np.arange(
        len(corr_labels)
    )
)

ax.set_yticks(
    np.arange(
        len(corr_labels)
    )
)


ax.set_xticklabels(
    corr_labels,
    rotation=40,
    ha="right",
    fontsize=7.5
)

ax.set_yticklabels(
    corr_labels,
    fontsize=7.5
)


for i in range(
    len(corr_labels)
):

    for j in range(
        len(corr_labels)
    ):

        rho_ij = corr_matrix[
            i,
            j
        ]

        p_ij = p_matrix[
            i,
            j
        ]


        if i == j:

            text = "1.00"

        else:

            if p_ij < 0.001:

                star = "***"

            elif p_ij < 0.01:

                star = "**"

            elif p_ij < 0.05:

                star = "*"

            else:

                star = ""

            text = (
                f"{rho_ij:.2f}"
                f"{star}"
            )


        text_color = (
            "white"
            if abs(rho_ij) > 0.55
            else
            "black"
        )


        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=7,
            color=text_color
        )


ax.set_title(
    "Spearman correlation",
    fontsize=9,
    pad=6
)


panel_label(
    ax,
    "(F)"
)


cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)


cbar.set_label(
    "Spearman ρ",
    fontsize=8
)

cbar.ax.tick_params(
    labelsize=7
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.075,
    right=0.975,
    top=0.965,
    bottom=0.105,
    wspace=0.31,
    hspace=0.34
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Figure5_automated_morphological_assessment"
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
print("Figure 5 completed")
print("============================================")

print(PNG)
print(SVG)
print(PDF)

print("\nCorrelation matrix:")
print(
    corr_df.round(3)
)

print("\nSaved correlation table:")
print(
    CORR_OUT
)


plt.close(
    fig
)

