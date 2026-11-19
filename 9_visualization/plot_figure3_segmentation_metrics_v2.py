from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

CSV = (
    ROOT /
    "9_results/surface_metrics/strict_surface_case_metrics.csv"
)

OUT_DIR = ROOT / "9_results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATS_OUT = (
    ROOT /
    "9_results/surface_metrics/"
    "Figure3_V2_wilcoxon_statistics.csv"
)


# ============================================================
# Methods and metrics
# ============================================================

METHODS = [
    "3D U-Net",
    "Attention U-Net",
    "MSAU w/o Multiscale",
    "MSAU w/o MSA",
    "MSAU-Net",
]

SHORT_NAMES = [
    "3D\nU-Net",
    "Attention\nU-Net",
    "w/o\nMultiscale",
    "w/o\nMSA",
    "MSAU-Net",
]

REFERENCE = "MSAU-Net"

METRICS = [
    ("Dice", "Dice"),
    ("IoU", "IoU"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("HD95_mm", "HD95 (mm)"),
    ("ASSD_mm", "ASSD (mm)"),
]

PANEL_LABELS = [
    "(A)",
    "(B)",
    "(C)",
    "(D)",
    "(E)",
    "(F)",
]


# ============================================================
# Publication style
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9

plt.rcParams["axes.linewidth"] = 0.75

plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7

plt.rcParams["svg.fonttype"] = "none"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# Load
# ============================================================

df = pd.read_csv(CSV)

print("Input:")
print(CSV)

print("\nCase counts:")

print(
    df.groupby("Method")["case_id"]
    .nunique()
)


# ============================================================
# Verify complete paired data
# ============================================================

for method in METHODS:

    n = df.loc[
        df["Method"] == method,
        "case_id"
    ].nunique()

    if n != 20:

        print(
            f"[WARNING] {method}: "
            f"{n} cases"
        )


# ============================================================
# Statistics
# ============================================================

def paired_wilcoxon(
        dataframe,
        metric,
        method_a,
        method_b
):

    a = dataframe[
        dataframe["Method"] == method_a
    ][
        ["case_id", metric]
    ].copy()

    b = dataframe[
        dataframe["Method"] == method_b
    ][
        ["case_id", metric]
    ].copy()

    pair = pd.merge(
        a,
        b,
        on="case_id",
        suffixes=("_a", "_b")
    )

    pair = pair.dropna()

    if len(pair) == 0:
        return np.nan

    x = pair[
        f"{metric}_a"
    ].to_numpy()

    y = pair[
        f"{metric}_b"
    ].to_numpy()

    diff = x - y

    if np.allclose(
        diff,
        0
    ):
        return 1.0

    try:

        _, p = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox"
        )

    except ValueError:

        p = np.nan

    return float(p)


def p_label(p):

    if np.isnan(p):
        return "n.s."

    if p < 0.001:
        return "***"

    if p < 0.01:
        return "**"

    if p < 0.05:
        return "*"

    return "n.s."


# ============================================================
# Save statistics table
# ============================================================

stat_rows = []

for metric, metric_name in METRICS:

    for method in METHODS[:-1]:

        p = paired_wilcoxon(
            df,
            metric,
            method,
            REFERENCE
        )

        stat_rows.append(
            {
                "Metric": metric_name,
                "Comparison":
                    f"{method} vs {REFERENCE}",
                "P_value": p,
                "Significance": p_label(p)
            }
        )


stats_df = pd.DataFrame(
    stat_rows
)

stats_df.to_csv(
    STATS_OUT,
    index=False
)


print("\nStatistics:")
print(
    stats_df.to_string(
        index=False
    )
)


# ============================================================
# Plot functions
# ============================================================

def get_values(
        dataframe,
        metric
):

    values = []

    for method in METHODS:

        x = (
            dataframe.loc[
                dataframe["Method"] == method,
                metric
            ]
            .dropna()
            .to_numpy()
        )

        values.append(
            x
        )

    return values


def draw_boxes(
        ax,
        values,
        show_points=True,
        point_size=11,
        seed=2026
):

    positions = np.arange(
        1,
        len(METHODS) + 1
    )


    bp = ax.boxplot(

        values,

        positions=positions,

        widths=0.52,

        patch_artist=True,

        showfliers=False,

        medianprops=dict(
            color="black",
            linewidth=1.05
        ),

        whiskerprops=dict(
            color="0.30",
            linewidth=0.80
        ),

        capprops=dict(
            color="0.30",
            linewidth=0.80
        ),

        boxprops=dict(
            color="0.30",
            linewidth=0.80
        )
    )


    box_colors = [
        "#B9BEC4",
        "#92D5EF",
        "#C7C9C5",
        "#E99A97",
        "#A9D49E",
    ]


    for patch, color in zip(
        bp["boxes"],
        box_colors
    ):

        patch.set_facecolor(
            color
        )

        patch.set_alpha(
            0.70
        )


    point_colors = [
        "#7A7F84",
        "#54B9E8",
        "#5FA45C",
        "#E36C65",
        "#69AE61",
    ]


    if show_points:

        rng = np.random.default_rng(
            seed
        )

        for idx, x in enumerate(
            values,
            start=1
        ):

            jitter = rng.normal(
                0,
                0.040,
                size=len(x)
            )

            ax.scatter(

                np.full(
                    len(x),
                    idx
                ) + jitter,

                x,

                s=point_size,

                alpha=0.62,

                color=point_colors[
                    idx - 1
                ],

                edgecolors="none",

                zorder=3
            )


    # Mean diamonds

    for idx, x in enumerate(
        values,
        start=1
    ):

        if len(x) == 0:
            continue

        ax.scatter(

            idx,

            np.mean(x),

            marker="D",

            s=27,

            facecolor="white",

            edgecolor="0.25",

            linewidth=0.70,

            zorder=5
        )


def add_bracket(
        ax,
        x1,
        x2,
        y,
        height,
        text
):

    ax.plot(

        [x1, x1, x2, x2],

        [
            y,
            y + height,
            y + height,
            y
        ],

        color="0.25",

        linewidth=0.70,

        clip_on=False
    )


    ax.text(

        (x1 + x2) / 2,

        y + height,

        text,

        ha="center",

        va="bottom",

        fontsize=8
    )


def format_axis(
        ax,
        ylabel
):

    ax.set_ylabel(
        ylabel,
        fontsize=10
    )

    ax.set_xticks(
        np.arange(
            1,
            len(METHODS) + 1
        )
    )

    ax.set_xticklabels(
        SHORT_NAMES,
        fontsize=8
    )

    ax.tick_params(
        axis="y",
        labelsize=8
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

    ax.grid(

        axis="y",

        linestyle="--",

        linewidth=0.40,

        alpha=0.25
    )


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(
        11.5,
        7.1
    ),

    dpi=300
)


axes = axes.flatten()


# ============================================================
# Main six panels
# ============================================================

for panel_index, (
        metric,
        ylabel
) in enumerate(
        METRICS
):

    ax = axes[
        panel_index
    ]

    values = get_values(
        df,
        metric
    )


    draw_boxes(
        ax,
        values,
        show_points=True,
        point_size=10,
        seed=2026 + panel_index
    )


    format_axis(
        ax,
        ylabel
    )


    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(

        -0.14,

        1.06,

        PANEL_LABELS[
            panel_index
        ],

        transform=ax.transAxes,

        fontsize=11,

        fontweight="bold",

        va="top"
    )


    # --------------------------------------------------------
    # Axis range
    # --------------------------------------------------------

    all_values = np.concatenate(
        values
    )

    data_min = np.min(
        all_values
    )

    data_max = np.max(
        all_values
    )

    data_range = (
        data_max -
        data_min
    )

    if data_range <= 0:
        data_range = 1.0


    # --------------------------------------------------------
    # Significance brackets:
    # each variant vs MSAU-Net
    # --------------------------------------------------------

    base_y = (
        data_max
        +
        0.055 * data_range
    )

    step = (
        0.085 * data_range
    )

    bracket_height = (
        0.018 * data_range
    )


    for comparison_index, method in enumerate(
        METHODS[:-1]
    ):

        p = paired_wilcoxon(
            df,
            metric,
            method,
            REFERENCE
        )

        text = p_label(
            p
        )

        y = (
            base_y
            +
            comparison_index * step
        )


        add_bracket(

            ax,

            comparison_index + 1,

            5,

            y,

            bracket_height,

            text
        )


    upper = (
        base_y
        +
        4.2 * step
    )


    lower = (
        data_min
        -
        0.06 * data_range
    )


    # --------------------------------------------------------
    # HD95 main panel:
    # preserve full range / outlier
    # --------------------------------------------------------

    if metric == "HD95_mm":

        lower = 0

        upper = max(
            upper,
            data_max * 1.22
        )


    ax.set_ylim(
        lower,
        upper
    )


# ============================================================
# HD95 inset: 0–5 mm
# ============================================================

hd_ax = axes[4]


inset = inset_axes(

    hd_ax,

    width="49%",

    height="45%",

    loc="upper right",

    borderpad=1.05
)


hd_values = get_values(
    df,
    "HD95_mm"
)


draw_boxes(

    inset,

    hd_values,

    show_points=True,

    point_size=6,

    seed=3030
)


inset.set_ylim(
    0,
    5
)


inset.set_xlim(
    0.55,
    5.45
)


inset.set_xticks(
    np.arange(
        1,
        6
    )
)


inset.set_xticklabels(
    [
        "U",
        "A",
        "M-",
        "A-",
        "M"
    ],
    fontsize=6
)


inset.tick_params(
    axis="y",
    labelsize=6,
    length=2
)

inset.tick_params(
    axis="x",
    length=2
)


inset.grid(

    axis="y",

    linestyle="--",

    linewidth=0.30,

    alpha=0.25
)


inset.set_title(

    "Zoom: 0–5 mm",

    fontsize=7,

    pad=2
)


inset.spines[
    "top"
].set_visible(
    True
)

inset.spines[
    "right"
].set_visible(
    True
)


# ============================================================
# Small note in HD95 panel
# ============================================================

hd_ax.text(

    0.03,

    0.94,

    "Inset shows the 0–5 mm range",

    transform=hd_ax.transAxes,

    fontsize=7,

    va="top"
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(

    left=0.070,

    right=0.985,

    top=0.965,

    bottom=0.115,

    wspace=0.30,

    hspace=0.36
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Figure3_segmentation_metrics_V2"
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
print("Figure 3 V2 completed")
print("============================================")

print(PNG)
print(SVG)
print(PDF)

print("\nStatistics:")
print(STATS_OUT)


plt.close(
    fig
)

