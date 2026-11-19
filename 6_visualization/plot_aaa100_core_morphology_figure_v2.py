#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-quality AAA-100 morphological phenotype figure.

A: Representative 3D AAA geometries
B: Aneurysm volume by diameter group
C: Sphericity by diameter group
D: Diameter-volume association
E: Diameter-tortuosity association
F: Spearman correlation heatmap

Outputs:
    PNG 600 dpi
    PDF vector
    SVG vector
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import ticker
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

DATA_FILE = (
    ROOT /
    "9_results/morphology_analysis_v3/"
    "AAA100_with_diameter_groups.csv"
)

POSTHOC_FILE = (
    ROOT /
    "9_results/morphology_analysis_v3/"
    "AAA100_diameter_groups_posthoc.csv"
)

CORR_FILE = (
    ROOT /
    "9_results/morphology_analysis_v3/"
    "AAA100_morphology_spearman_correlations.csv"
)

OUTDIR = (
    ROOT /
    "9_results/morphology_analysis_v3/"
    "figures"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUTBASE = (
    OUTDIR /
    "Figure_AAA100_core_morphological_phenotype_v2"
)


# ============================================================
# 2. STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif"
    ],

    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,

    "axes.linewidth": 0.9,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
posthoc = pd.read_csv(POSTHOC_FILE)
corr_results = pd.read_csv(CORR_FILE)

GROUP_ORDER = [
    "<50 mm",
    "50–54.9 mm",
    "≥55 mm"
]

df["diameter_group"] = pd.Categorical(
    df["diameter_group"],
    categories=GROUP_ORDER,
    ordered=True
)


# ============================================================
# 4. REPRESENTATIVE CASES
# ============================================================

def select_median_case(data, group_name):

    g = data[
        data["diameter_group"] == group_name
    ].copy()

    median_d = g[
        "max_feret_diameter_mm"
    ].median()

    idx = (
        g["max_feret_diameter_mm"] -
        median_d
    ).abs().idxmin()

    return g.loc[idx]


representatives = {
    g: select_median_case(df, g)
    for g in GROUP_ORDER
}


print("=" * 90)
print("REPRESENTATIVE CASES")
print("=" * 90)

for g in GROUP_ORDER:

    r = representatives[g]

    print(
        f"{g:12s} "
        f"{r['case_id']}  "
        f"D={r['max_feret_diameter_mm']:.2f} mm  "
        f"Volume={r['mesh_volume_mm3']/1000:.2f} cm³"
    )


# ============================================================
# 5. STL LOADER
# ============================================================

def load_stl_mesh(path):

    try:

        import trimesh

        mesh = trimesh.load(
            path,
            force="mesh",
            process=False
        )

        return (
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces)
        )

    except Exception as e:

        raise RuntimeError(
            "\nUnable to load STL.\n"
            "Install trimesh if necessary:\n"
            "pip install trimesh\n"
            f"\nOriginal error: {e}"
        )


# ============================================================
# 6. 3D REPRESENTATION
# ============================================================

def set_equal_3d(ax, vertices):

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    center = (
        mins + maxs
    ) / 2

    spans = (
        maxs - mins
    )

    # Slightly tighter framing than the old version
    radius = spans.max() * 0.43

    ax.set_xlim(
        center[0] - radius,
        center[0] + radius
    )

    ax.set_ylim(
        center[1] - radius,
        center[1] + radius
    )

    ax.set_zlim(
        center[2] - radius,
        center[2] + radius
    )

    try:
        ax.set_box_aspect(
            (
                spans[0],
                spans[1],
                spans[2]
            )
        )
    except Exception:
        pass


def draw_mesh(ax, stl_path):

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, faces = load_stl_mesh(
        stl_path
    )

    # --------------------------------------------------------
    # Downsample only for plotting
    # --------------------------------------------------------

    max_faces = 70000

    if len(faces) > max_faces:

        step = int(
            np.ceil(
                len(faces) / max_faces
            )
        )

        faces_plot = faces[::step]

    else:

        faces_plot = faces

    triangles = vertices[
        faces_plot
    ]

    poly = Poly3DCollection(
        triangles,
        linewidths=0.06,
        edgecolors="0.38",
        alpha=1.0
    )

    poly.set_facecolor(
        "0.58"
    )

    ax.add_collection3d(
        poly
    )

    set_equal_3d(
        ax,
        vertices
    )

    # Better vascular view
    ax.view_init(
        elev=10,
        azim=-72
    )

    ax.set_axis_off()

    # Zoom in
    try:
        ax.dist = 6.7
    except Exception:
        pass


# ============================================================
# 7. PANEL LABEL
# ============================================================

def panel_label(ax, label):

    ax.text(
        -0.105,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top"
    )


# ============================================================
# 8. CLEAN AXES
# ============================================================

def clean_axes(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(
        direction="out",
        width=0.8,
        length=3.5
    )


# ============================================================
# 9. BOX + POINT PLOT
# ============================================================

def boxplot_with_points(
    ax,
    variable,
    ylabel
):

    arrays = [
        df.loc[
            df["diameter_group"] == g,
            variable
        ].dropna().values
        for g in GROUP_ORDER
    ]

    bp = ax.boxplot(
        arrays,
        positions=[1, 2, 3],
        widths=0.55,
        patch_artist=True,
        showfliers=False,

        medianprops={
            "linewidth": 1.5,
            "color": "black"
        },

        boxprops={
            "linewidth": 0.9,
            "edgecolor": "0.15"
        },

        whiskerprops={
            "linewidth": 0.9,
            "color": "0.2"
        },

        capprops={
            "linewidth": 0.9,
            "color": "0.2"
        }
    )

    shades = [
        "0.88",
        "0.72",
        "0.54"
    ]

    for patch, shade in zip(
        bp["boxes"],
        shades
    ):
        patch.set_facecolor(shade)

    # --------------------------------------------------------
    # Raw observations
    # --------------------------------------------------------

    rng = np.random.default_rng(
        20260901
    )

    for i, values in enumerate(
        arrays,
        start=1
    ):

        jitter = rng.normal(
            loc=i,
            scale=0.045,
            size=len(values)
        )

        ax.scatter(
            jitter,
            values,
            s=11,
            facecolors="white",
            edgecolors="0.42",
            linewidths=0.45,
            alpha=0.72,
            zorder=4
        )

    ax.set_xticks(
        [1, 2, 3]
    )

    ax.set_xticklabels(
        GROUP_ORDER
    )

    ax.set_xlabel(
        "Maximum Feret diameter group",
        labelpad=4
    )

    ax.set_ylabel(
        ylabel,
        labelpad=5
    )

    clean_axes(ax)


# ============================================================
# 10. POST-HOC P VALUES
# ============================================================

def get_posthoc_p(
    metric,
    g1,
    g2
):

    mask = (
        (posthoc["metric"] == metric)
        &
        (
            (
                (posthoc["group1"] == g1)
                &
                (posthoc["group2"] == g2)
            )
            |
            (
                (posthoc["group1"] == g2)
                &
                (posthoc["group2"] == g1)
            )
        )
    )

    r = posthoc[mask]

    if len(r) == 0:
        return np.nan

    return float(
        r.iloc[0]["p_holm"]
    )


def format_p(p):

    if p < 0.001:
        return r"$P_{\mathrm{adj}}<0.001$"

    return (
        rf"$P_{{\mathrm{{adj}}}}={p:.3f}$"
    )


def significance_bar(
    ax,
    x1,
    x2,
    y,
    text,
    h
):

    ax.plot(
        [x1, x1, x2, x2],
        [y, y+h, y+h, y],
        color="0.18",
        linewidth=0.75,
        clip_on=False
    )

    ax.text(
        (x1+x2)/2,
        y+h*1.30,
        text,
        ha="center",
        va="bottom",
        fontsize=7.8
    )


# ============================================================
# 11. BOOTSTRAP LINEAR VISUAL CI
# ============================================================

def bootstrap_linear_ci(
    x,
    y,
    xgrid,
    n_boot=2000,
    seed=20260901
):

    rng = np.random.default_rng(
        seed
    )

    n = len(x)

    predictions = np.zeros(
        (
            n_boot,
            len(xgrid)
        )
    )

    for b in range(n_boot):

        idx = rng.integers(
            0,
            n,
            n
        )

        xb = x[idx]
        yb = y[idx]

        # Rare degenerate sample safeguard
        if np.std(xb) == 0:
            continue

        coef = np.polyfit(
            xb,
            yb,
            1
        )

        predictions[b] = (
            coef[0] * xgrid +
            coef[1]
        )

    lower = np.percentile(
        predictions,
        2.5,
        axis=0
    )

    upper = np.percentile(
        predictions,
        97.5,
        axis=0
    )

    return lower, upper


# ============================================================
# 12. CORRELATION LOOKUP
# ============================================================

def get_corr_result(
    xvar,
    yvar
):

    mask = (
        (
            (corr_results["variable_1"] == xvar)
            &
            (corr_results["variable_2"] == yvar)
        )
        |
        (
            (corr_results["variable_1"] == yvar)
            &
            (corr_results["variable_2"] == xvar)
        )
    )

    r = corr_results[mask]

    if len(r) == 1:

        rho = float(
            r.iloc[0]["spearman_rho"]
        )

        p = float(
            r.iloc[0]["p_holm"]
        )

        return rho, p

    rho, p = spearmanr(
        df[xvar],
        df[yvar]
    )

    return rho, p


# ============================================================
# 13. SCATTER + TREND
# ============================================================

def scatter_panel(
    ax,
    xvar,
    yvar,
    xlabel,
    ylabel
):

    x = df[xvar].to_numpy(
        dtype=float
    )

    y = df[yvar].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        s=18,
        facecolors="0.82",
        edgecolors="0.25",
        linewidths=0.45,
        alpha=0.82,
        zorder=3
    )

    # --------------------------------------------------------
    # Linear visual trend
    # --------------------------------------------------------

    coef = np.polyfit(
        x,
        y,
        1
    )

    xgrid = np.linspace(
        x.min(),
        x.max(),
        250
    )

    yfit = (
        coef[0] * xgrid +
        coef[1]
    )

    lower, upper = bootstrap_linear_ci(
        x,
        y,
        xgrid
    )

    ax.fill_between(
        xgrid,
        lower,
        upper,
        color="0.80",
        alpha=0.45,
        linewidth=0,
        zorder=1
    )

    ax.plot(
        xgrid,
        yfit,
        color="black",
        linewidth=1.35,
        zorder=4
    )

    # --------------------------------------------------------
    # Spearman statistics
    # --------------------------------------------------------

    rho, p = get_corr_result(
        xvar,
        yvar
    )

    if p < 0.001:

        p_line = (
            r"Holm-adjusted $P<0.001$"
        )

    else:

        p_line = (
            rf"Holm-adjusted $P={p:.3f}$"
        )

    text = (
        rf"Spearman $\rho={rho:.3f}$"
        "\n"
        + p_line
    )

    ax.text(
        0.035,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "linewidth": 0.6,
            "alpha": 0.94
        }
    )

    ax.set_xlabel(
        xlabel,
        labelpad=4
    )

    ax.set_ylabel(
        ylabel,
        labelpad=5
    )

    clean_axes(ax)


# ============================================================
# 14. HEATMAP DATA
# ============================================================

HEATMAP_METRICS = [
    "max_feret_diameter_mm",
    "max_section_area_mm2",
    "mesh_volume_mm3",
    "centerline_length_mm",
    "tortuosity",
    "sphericity",
]

HEATMAP_LABELS = [
    "Max diameter",
    "Max area",
    "Volume",
    "Centerline length",
    "Tortuosity",
    "Sphericity",
]

n_heat = len(
    HEATMAP_METRICS
)

rho_matrix = np.eye(
    n_heat
)

p_matrix = np.zeros(
    (
        n_heat,
        n_heat
    )
)


for i in range(n_heat):

    for j in range(
        i+1,
        n_heat
    ):

        a = HEATMAP_METRICS[i]
        b = HEATMAP_METRICS[j]

        rho, p = get_corr_result(
            a,
            b
        )

        rho_matrix[i, j] = rho
        rho_matrix[j, i] = rho

        p_matrix[i, j] = p
        p_matrix[j, i] = p


# ============================================================
# 15. MAIN FIGURE
# ============================================================

fig = plt.figure(
    figsize=(14.4, 8.9),
    facecolor="white"
)

gs = GridSpec(
    2,
    3,
    figure=fig,

    width_ratios=[
        1.10,
        1.00,
        1.05
    ],

    height_ratios=[
        1.0,
        1.0
    ],

    left=0.055,
    right=0.975,
    bottom=0.095,
    top=0.965,

    wspace=0.30,
    hspace=0.34
)


# ============================================================
# A — 3D REPRESENTATIVE CASES
# ============================================================

gsA = GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gs[0, 0],
    wspace=-0.03
)

for i, group in enumerate(
    GROUP_ORDER
):

    ax = fig.add_subplot(
        gsA[0, i],
        projection="3d"
    )

    r = representatives[group]

    draw_mesh(
        ax,
        r["stl_file"]
    )

    # --------------------------------------------------------
    # Group label ABOVE mesh
    # --------------------------------------------------------

    ax.text2D(
        0.50,
        0.985,
        group,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Case + diameter BELOW group
    # --------------------------------------------------------

    ax.text2D(
        0.50,
        0.895,
        (
            f"{r['case_id']}\n"
            f"D={r['max_feret_diameter_mm']:.1f} mm"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        linespacing=1.10
    )


axA = fig.add_subplot(
    gs[0, 0]
)

axA.patch.set_alpha(0)

for spine in axA.spines.values():
    spine.set_visible(False)

axA.set_xticks([])
axA.set_yticks([])

panel_label(
    axA,
    "A"
)

axA.text(
    0.50,
    -0.005,
    "Representative 3D aneurysm geometries",
    transform=axA.transAxes,
    ha="center",
    va="top",
    fontsize=10.5
)


# ============================================================
# B — VOLUME
# ============================================================

axB = fig.add_subplot(
    gs[0, 1]
)

boxplot_with_points(
    axB,
    "mesh_volume_mm3",
    "Aneurysm volume (mm³)"
)

panel_label(
    axB,
    "B"
)

axB.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda x, pos:
        f"{x/1000:.0f}k"
    )
)

p13 = get_posthoc_p(
    "mesh_volume_mm3",
    "<50 mm",
    "≥55 mm"
)

p23 = get_posthoc_p(
    "mesh_volume_mm3",
    "50–54.9 mm",
    "≥55 mm"
)

data_max = df[
    "mesh_volume_mm3"
].max()

data_min = df[
    "mesh_volume_mm3"
].min()

data_range = (
    data_max -
    data_min
)

axB.set_ylim(
    data_min - 0.08 * data_range,
    data_max + 0.29 * data_range
)

significance_bar(
    axB,
    1,
    3,
    data_max + 0.09 * data_range,
    format_p(p13),
    0.025 * data_range
)

significance_bar(
    axB,
    2,
    3,
    data_max + 0.20 * data_range,
    format_p(p23),
    0.025 * data_range
)


# ============================================================
# C — SPHERICITY
# ============================================================

axC = fig.add_subplot(
    gs[0, 2]
)

boxplot_with_points(
    axC,
    "sphericity",
    "Sphericity"
)

panel_label(
    axC,
    "C"
)

p12 = get_posthoc_p(
    "sphericity",
    "<50 mm",
    "50–54.9 mm"
)

p13 = get_posthoc_p(
    "sphericity",
    "<50 mm",
    "≥55 mm"
)

data_max = df[
    "sphericity"
].max()

data_min = df[
    "sphericity"
].min()

data_range = (
    data_max -
    data_min
)

axC.set_ylim(
    data_min - 0.08 * data_range,
    data_max + 0.29 * data_range
)

significance_bar(
    axC,
    1,
    2,
    data_max + 0.08 * data_range,
    format_p(p12),
    0.018 * data_range
)

significance_bar(
    axC,
    1,
    3,
    data_max + 0.19 * data_range,
    format_p(p13),
    0.018 * data_range
)


# ============================================================
# D — DIAMETER vs VOLUME
# ============================================================

axD = fig.add_subplot(
    gs[1, 0]
)

scatter_panel(
    axD,
    "max_feret_diameter_mm",
    "mesh_volume_mm3",
    "Maximum Feret diameter (mm)",
    "Aneurysm volume (mm³)"
)

panel_label(
    axD,
    "D"
)

axD.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda x, pos:
        f"{x/1000:.0f}k"
    )
)


# ============================================================
# E — DIAMETER vs TORTUOSITY
# ============================================================

axE = fig.add_subplot(
    gs[1, 1]
)

scatter_panel(
    axE,
    "max_feret_diameter_mm",
    "tortuosity",
    "Maximum Feret diameter (mm)",
    "Tortuosity"
)

panel_label(
    axE,
    "E"
)


# ============================================================
# F — SPEARMAN HEATMAP
# ============================================================

axF = fig.add_subplot(
    gs[1, 2]
)

im = axF.imshow(
    rho_matrix,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    interpolation="nearest",
    aspect="equal"
)

axF.set_xticks(
    np.arange(n_heat)
)

axF.set_yticks(
    np.arange(n_heat)
)

axF.set_xticklabels(
    HEATMAP_LABELS,
    rotation=48,
    ha="right",
    rotation_mode="anchor",
    fontsize=8.5
)

axF.set_yticklabels(
    HEATMAP_LABELS,
    fontsize=8.5
)

axF.tick_params(
    length=0
)


# ------------------------------------------------------------
# Thin white grid
# ------------------------------------------------------------

axF.set_xticks(
    np.arange(-0.5, n_heat, 1),
    minor=True
)

axF.set_yticks(
    np.arange(-0.5, n_heat, 1),
    minor=True
)

axF.grid(
    which="minor",
    color="white",
    linewidth=1.0
)

axF.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ------------------------------------------------------------
# Cell text
# ------------------------------------------------------------

for i in range(n_heat):

    for j in range(n_heat):

        rho = rho_matrix[
            i,
            j
        ]

        p = p_matrix[
            i,
            j
        ]

        if i == j:

            text = "1.00"

        else:

            if p < 0.001:
                stars = "***"

            elif p < 0.01:
                stars = "**"

            elif p < 0.05:
                stars = "*"

            else:
                stars = ""

            text = (
                f"{rho:.2f}"
                f"{stars}"
            )

        text_color = (
            "white"
            if abs(rho) >= 0.55
            else "black"
        )

        axF.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=7.6,
            color=text_color
        )


panel_label(
    axF,
    "F"
)


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(
    im,
    ax=axF,
    fraction=0.043,
    pad=0.035
)

cbar.set_label(
    r"Spearman $\rho$",
    fontsize=9.5
)

cbar.ax.tick_params(
    labelsize=8,
    width=0.7,
    length=3
)

cbar.outline.set_linewidth(
    0.7
)


# ------------------------------------------------------------
# Heatmap significance note
# ------------------------------------------------------------

axF.text(
    0.50,
    -0.33,
    (
        "* Holm-adjusted P<0.05; "
        "** P<0.01; "
        "*** P<0.001"
    ),
    transform=axF.transAxes,
    ha="center",
    va="top",
    fontsize=7.5
)


# ============================================================
# 16. EXPORT
# ============================================================

png_file = Path(
    str(OUTBASE) + ".png"
)

pdf_file = Path(
    str(OUTBASE) + ".pdf"
)

svg_file = Path(
    str(OUTBASE) + ".svg"
)


fig.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.05
)

fig.savefig(
    pdf_file,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.05
)

fig.savefig(
    svg_file,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.05
)

plt.close(fig)


# ============================================================
# 17. FINISHED
# ============================================================

print("\n" + "=" * 90)
print("PUBLICATION FIGURE V2 FINISHED")
print("=" * 90)

print("PNG :", png_file)
print("PDF :", pdf_file)
print("SVG :", svg_file)

print("\nPanels:")
print("A : Representative 3D AAA geometries")
print("B : Aneurysm volume by diameter group")
print("C : Sphericity by diameter group")
print("D : Diameter-volume association")
print("E : Diameter-tortuosity association")
print("F : Spearman morphology correlation heatmap")

print("=" * 90)
