from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

CSV = (
    ROOT /
    "9_results/surface_metrics/strict_surface_case_metrics.csv"
)

OUT_DIR = ROOT / "9_results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration
# ============================================================

METHODS = [
    "3D U-Net",
    "Attention U-Net",
    "MSAU w/o Multiscale",
    "MSAU w/o MSA",
    "MSAU-Net",
]

METRICS = [
    ("Dice",      "Dice"),
    ("IoU",       "IoU"),
    ("Precision", "Precision"),
    ("Recall",    "Recall"),
    ("HD95_mm",   "HD95 (mm)"),
    ("ASSD_mm",   "ASSD (mm)"),
]

# Only compare each model with full MSAU-Net
REFERENCE = "MSAU-Net"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV)

print("Input:")
print(CSV)

print("\nColumns:")
print(df.columns.tolist())

print("\nCases per method:")
print(df.groupby("Method")["case_id"].nunique())


# ============================================================
# Check data
# ============================================================

for method in METHODS:

    n = df.loc[
        df["Method"] == method,
        "case_id"
    ].nunique()

    if n != 20:
        print(
            f"[WARNING] {method}: "
            f"{n} cases instead of 20"
        )


# ============================================================
# Paired Wilcoxon test
# ============================================================

def paired_test(data, metric, method1, method2):

    a = data[
        data["Method"] == method1
    ][["case_id", metric]].copy()

    b = data[
        data["Method"] == method2
    ][["case_id", metric]].copy()

    merged = pd.merge(
        a,
        b,
        on="case_id",
        suffixes=("_1", "_2")
    ).dropna()

    if len(merged) == 0:
        return np.nan

    x = merged[f"{metric}_1"].values
    y = merged[f"{metric}_2"].values

    diff = x - y

    # all identical
    if np.allclose(diff, 0):
        return 1.0

    try:

        _, p = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox"
        )

    except ValueError:

        return np.nan

    return p


def p_to_star(p):

    if np.isnan(p):
        return "n.s."

    if p < 0.001:
        return "***"

    elif p < 0.01:
        return "**"

    elif p < 0.05:
        return "*"

    else:
        return "n.s."


# ============================================================
# Statistical table
# ============================================================

stats_rows = []

for metric, label in METRICS:

    for method in METHODS[:-1]:

        p = paired_test(
            df,
            metric,
            method,
            REFERENCE
        )

        stats_rows.append({
            "Metric": label,
            "Comparison":
                f"{method} vs {REFERENCE}",
            "P_value": p,
            "Significance": p_to_star(p)
        })


stats_df = pd.DataFrame(stats_rows)

stats_file = (
    ROOT /
    "9_results/surface_metrics/"
    "figure3_wilcoxon_statistics.csv"
)

stats_df.to_csv(
    stats_file,
    index=False
)

print("\nWilcoxon statistics:")
print(stats_df.to_string(index=False))


# ============================================================
# Helper: significance bracket
# ============================================================

def add_significance(
        ax,
        x1,
        x2,
        y,
        h,
        text
):

    ax.plot(
        [x1, x1, x2, x2],
        [y, y+h, y+h, y],
        linewidth=0.8,
        color="black",
        clip_on=False
    )

    ax.text(
        (x1+x2)/2,
        y+h,
        text,
        ha="center",
        va="bottom",
        fontsize=8
    )


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12.0, 7.4),
    dpi=300
)

axes = axes.flatten()

rng = np.random.default_rng(2026)

panel_letters = [
    "(A)",
    "(B)",
    "(C)",
    "(D)",
    "(E)",
    "(F)",
]


for panel_idx, ((metric, ylabel), ax) in enumerate(
        zip(METRICS, axes)
):

    values = []

    for method in METHODS:

        x = (
            df.loc[
                df["Method"] == method,
                metric
            ]
            .dropna()
            .values
        )

        values.append(x)


    # --------------------------------------------------------
    # Boxplots
    # --------------------------------------------------------

    bp = ax.boxplot(
        values,
        positions=np.arange(1, len(METHODS)+1),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color="black",
            linewidth=1.2
        ),
        whiskerprops=dict(
            linewidth=0.9
        ),
        capprops=dict(
            linewidth=0.9
        ),
        boxprops=dict(
            linewidth=0.9
        )
    )

    # grayscale-compatible fills
    fills = [
        "white",
        "0.90",
        "0.80",
        "0.70",
        "0.55"
    ]

    for patch, fc in zip(
        bp["boxes"],
        fills
    ):
        patch.set_facecolor(fc)


    # --------------------------------------------------------
    # Individual case points
    # --------------------------------------------------------

    for i, x in enumerate(values, start=1):

        jitter = rng.normal(
            0,
            0.045,
            size=len(x)
        )

        ax.scatter(
            np.full(len(x), i) + jitter,
            x,
            s=12,
            alpha=0.55,
            edgecolors="none",
            zorder=3
        )


    # --------------------------------------------------------
    # Mean marker
    # --------------------------------------------------------

    for i, x in enumerate(values, start=1):

        if len(x) == 0:
            continue

        ax.scatter(
            i,
            np.mean(x),
            marker="D",
            s=24,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=5
        )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_ylabel(
        ylabel,
        fontsize=10
    )

    ax.set_xticks(
        np.arange(
            1,
            len(METHODS)+1
        )
    )

    ax.set_xticklabels(
        [
            "3D\nU-Net",
            "Attention\nU-Net",
            "w/o\nMultiscale",
            "w/o\nMSA",
            "MSAU-Net"
        ],
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.45,
        alpha=0.35
    )


    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.14,
        1.05,
        panel_letters[panel_idx],
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )


    # --------------------------------------------------------
    # Significance vs MSAU-Net
    # --------------------------------------------------------

    all_values = np.concatenate(
        [x for x in values if len(x) > 0]
    )

    data_min = np.min(all_values)
    data_max = np.max(all_values)

    data_range = data_max - data_min

    if data_range == 0:
        data_range = abs(data_max) * 0.1

    if data_range == 0:
        data_range = 1.0


    base_y = (
        data_max +
        0.07 * data_range
    )

    step = (
        0.10 * data_range
    )

    bracket_h = (
        0.025 * data_range
    )


    # Four comparisons with full model
    for j, method in enumerate(
        METHODS[:-1]
    ):

        p = paired_test(
            df,
            metric,
            method,
            REFERENCE
        )

        star = p_to_star(p)

        y = base_y + j * step

        add_significance(
            ax,
            j + 1,
            5,
            y,
            bracket_h,
            star
        )


    upper = (
        base_y +
        4.2 * step
    )

    lower = (
        data_min -
        0.08 * data_range
    )

    ax.set_ylim(
        lower,
        upper
    )


# ============================================================
# Legend
# ============================================================

fig.text(
    0.5,
    0.015,
    "Boxes represent the interquartile range; "
    "horizontal lines indicate medians; "
    "diamonds indicate means; "
    "points represent individual CTA cases. "
    "* P < 0.05, ** P < 0.01, *** P < 0.001; "
    "n.s., not significant. "
    "Paired two-sided Wilcoxon signed-rank test.",
    ha="center",
    va="bottom",
    fontsize=8
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.965,
    bottom=0.15,
    wspace=0.30,
    hspace=0.38
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Figure3_segmentation_metrics"
)

PNG = BASE.with_suffix(".png")
SVG = BASE.with_suffix(".svg")
PDF = BASE.with_suffix(".pdf")


fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    SVG,
    format="svg",
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    PDF,
    format="pdf",
    bbox_inches="tight",
    facecolor="white"
)


print("\n======================================")
print("Figure 3 saved")
print("======================================")
print(PNG)
print(SVG)
print(PDF)
print("\nStatistics:")
print(stats_file)

plt.close(fig)

