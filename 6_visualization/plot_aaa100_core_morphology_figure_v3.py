#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# PATHS
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

DATA_FILE = ROOT / "9_results/morphology_analysis_v3/AAA100_with_diameter_groups.csv"
POSTHOC_FILE = ROOT / "9_results/morphology_analysis_v3/AAA100_diameter_groups_posthoc.csv"
CORR_FILE = ROOT / "9_results/morphology_analysis_v3/AAA100_morphology_spearman_correlations.csv"

OUTDIR = ROOT / "9_results/morphology_analysis_v3/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTBASE = OUTDIR / "Figure_AAA100_core_morphological_phenotype_v3_color"


# ============================================================
# COLORS
# ============================================================

GROUP_COLORS = {
    "<50 mm": "#4C78A8",
    "50–54.9 mm": "#F2A541",
    "≥55 mm": "#D95F5F",
}

BLUE = "#2878B5"
ORANGE = "#F39C34"
RED = "#C44E52"
DARK = "#303030"
LIGHT_BLUE = "#BFD7EA"
LIGHT_ORANGE = "#F8D7A0"


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
posthoc = pd.read_csv(POSTHOC_FILE)
corr_results = pd.read_csv(CORR_FILE)

GROUP_ORDER = ["<50 mm", "50–54.9 mm", "≥55 mm"]

df["diameter_group"] = pd.Categorical(
    df["diameter_group"],
    categories=GROUP_ORDER,
    ordered=True
)


# ============================================================
# REPRESENTATIVE CASES
# ============================================================

def select_median_case(data, group):
    g = data[data["diameter_group"] == group].copy()
    med = g["max_feret_diameter_mm"].median()
    idx = (g["max_feret_diameter_mm"] - med).abs().idxmin()
    return g.loc[idx]


representatives = {
    g: select_median_case(df, g)
    for g in GROUP_ORDER
}


# ============================================================
# STL
# ============================================================

def load_stl_mesh(path):
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


def set_equal_3d(ax, vertices):

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    center = (mins + maxs) / 2
    spans = maxs - mins

    radius = spans.max() * 0.40

    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

    try:
        ax.set_box_aspect((spans[0], spans[1], spans[2]))
    except Exception:
        pass


def draw_mesh(ax, stl_path, color):

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, faces = load_stl_mesh(stl_path)

    if len(faces) > 70000:
        step = int(np.ceil(len(faces) / 70000))
        faces = faces[::step]

    triangles = vertices[faces]

    poly = Poly3DCollection(
        triangles,
        linewidths=0.04,
        edgecolors=color,
        alpha=0.97
    )

    poly.set_facecolor(color)

    ax.add_collection3d(poly)

    set_equal_3d(ax, vertices)

    ax.view_init(
        elev=8,
        azim=-72
    )

    ax.set_axis_off()


# ============================================================
# COMMON AXES
# ============================================================

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.8
    )


def panel_label(ax, txt):
    ax.text(
        -0.10,
        1.055,
        txt,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top"
    )


# ============================================================
# BOXPLOTS
# ============================================================

def color_boxplot(ax, variable, ylabel):

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
            "color": "black",
            "linewidth": 1.6
        },
        whiskerprops={
            "color": "0.25",
            "linewidth": 0.9
        },
        capprops={
            "color": "0.25",
            "linewidth": 0.9
        },
        boxprops={
            "linewidth": 0.9,
            "edgecolor": "0.20"
        }
    )

    for patch, group in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(0.72)

    rng = np.random.default_rng(20260901)

    for i, (group, values) in enumerate(
        zip(GROUP_ORDER, arrays),
        start=1
    ):

        x = rng.normal(
            i,
            0.045,
            len(values)
        )

        ax.scatter(
            x,
            values,
            s=13,
            facecolor="white",
            edgecolor=GROUP_COLORS[group],
            linewidth=0.65,
            alpha=0.85,
            zorder=5
        )

    ax.set_xticks([1,2,3])
    ax.set_xticklabels(GROUP_ORDER)

    ax.set_xlabel(
        "Maximum Feret diameter group",
        labelpad=5
    )

    ax.set_ylabel(
        ylabel,
        labelpad=6
    )

    clean_axes(ax)


# ============================================================
# STATISTICS
# ============================================================

def get_posthoc_p(metric, g1, g2):

    r = posthoc[
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
    ]

    return float(r.iloc[0]["p_holm"])


def format_p(p):

    if p < 0.001:
        return r"$P_{\mathrm{adj}}<0.001$"

    return rf"$P_{{\mathrm{{adj}}}}={p:.3f}$"


def significance_bar(ax, x1, x2, y, text, h):

    ax.plot(
        [x1, x1, x2, x2],
        [y, y+h, y+h, y],
        color=DARK,
        linewidth=0.8,
        clip_on=False
    )

    ax.text(
        (x1+x2)/2,
        y+h*1.35,
        text,
        ha="center",
        va="bottom",
        fontsize=7.7,
        clip_on=False
    )


# ============================================================
# CORRELATION
# ============================================================

def get_corr_result(xvar, yvar):

    r = corr_results[
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
    ]

    if len(r) == 1:
        return (
            float(r.iloc[0]["spearman_rho"]),
            float(r.iloc[0]["p_holm"])
        )

    return spearmanr(df[xvar], df[yvar])


def bootstrap_linear_ci(
    x,
    y,
    xgrid,
    n_boot=1500,
    seed=20260901
):

    rng = np.random.default_rng(seed)

    preds = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            len(x),
            len(x)
        )

        xb = x[idx]
        yb = y[idx]

        if np.std(xb) == 0:
            continue

        coef = np.polyfit(
            xb,
            yb,
            1
        )

        preds.append(
            coef[0] * xgrid +
            coef[1]
        )

    preds = np.asarray(preds)

    return (
        np.percentile(preds, 2.5, axis=0),
        np.percentile(preds, 97.5, axis=0)
    )


def scatter_panel(
    ax,
    xvar,
    yvar,
    xlabel,
    ylabel,
    point_color,
    band_color
):

    x = df[xvar].to_numpy(float)
    y = df[yvar].to_numpy(float)

    ax.scatter(
        x,
        y,
        s=22,
        facecolors="white",
        edgecolors=point_color,
        linewidths=0.75,
        alpha=0.85,
        zorder=3
    )

    coef = np.polyfit(x, y, 1)

    xx = np.linspace(
        x.min(),
        x.max(),
        250
    )

    yy = coef[0] * xx + coef[1]

    low, high = bootstrap_linear_ci(
        x,
        y,
        xx
    )

    ax.fill_between(
        xx,
        low,
        high,
        color=band_color,
        alpha=0.38,
        linewidth=0,
        zorder=1
    )

    ax.plot(
        xx,
        yy,
        color=point_color,
        linewidth=1.8,
        zorder=4
    )

    rho, p = get_corr_result(
        xvar,
        yvar
    )

    # Statistics are placed ABOVE the plot, not on the data.
    if p < 0.001:
        ptxt = r"Holm-adjusted $P<0.001$"
    else:
        ptxt = rf"Holm-adjusted $P={p:.3f}$"

    ax.set_title(
        rf"Spearman $\rho={rho:.3f}$; {ptxt}",
        fontsize=9.2,
        pad=9,
        color=DARK
    )

    ax.set_xlabel(
        xlabel,
        labelpad=5
    )

    ax.set_ylabel(
        ylabel,
        labelpad=6
    )

    clean_axes(ax)


# ============================================================
# HEATMAP
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

n = len(HEATMAP_METRICS)

rho_matrix = np.eye(n)
p_matrix = np.zeros((n,n))

for i in range(n):
    for j in range(i+1, n):

        rho, p = get_corr_result(
            HEATMAP_METRICS[i],
            HEATMAP_METRICS[j]
        )

        rho_matrix[i,j] = rho
        rho_matrix[j,i] = rho

        p_matrix[i,j] = p
        p_matrix[j,i] = p


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(14.6, 9.4),
    facecolor="white"
)

gs = GridSpec(
    2,
    3,
    figure=fig,

    width_ratios=[
        1.10,
        1.02,
        1.08
    ],

    height_ratios=[
        1,
        1
    ],

    left=0.055,
    right=0.965,
    bottom=0.11,
    top=0.94,

    wspace=0.34,
    hspace=0.48
)


# ============================================================
# A
# ============================================================

gsA = GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gs[0,0],
    wspace=0.08
)

for i, group in enumerate(GROUP_ORDER):

    ax = fig.add_subplot(
        gsA[0,i],
        projection="3d"
    )

    r = representatives[group]

    draw_mesh(
        ax,
        r["stl_file"],
        GROUP_COLORS[group]
    )

    # All labels above mesh
    ax.text2D(
        0.50,
        1.01,
        group,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=GROUP_COLORS[group]
    )

    ax.text2D(
        0.50,
        0.92,
        f"{r['case_id']}\nD={r['max_feret_diameter_mm']:.1f} mm",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.1,
        color=DARK
    )


axA = fig.add_subplot(gs[0,0])

axA.patch.set_alpha(0)

for s in axA.spines.values():
    s.set_visible(False)

axA.set_xticks([])
axA.set_yticks([])

panel_label(axA, "A")

axA.text(
    0.5,
    -0.02,
    "Representative 3D aneurysm geometries",
    transform=axA.transAxes,
    ha="center",
    va="top",
    fontsize=10.5
)


# ============================================================
# B
# ============================================================

axB = fig.add_subplot(gs[0,1])

color_boxplot(
    axB,
    "mesh_volume_mm3",
    "Aneurysm volume (mm³)"
)

panel_label(axB, "B")

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

vmin = df["mesh_volume_mm3"].min()
vmax = df["mesh_volume_mm3"].max()
vr = vmax-vmin

axB.set_ylim(
    vmin-0.08*vr,
    vmax+0.34*vr
)

significance_bar(
    axB,
    1,3,
    vmax+0.10*vr,
    format_p(p13),
    0.022*vr
)

significance_bar(
    axB,
    2,3,
    vmax+0.23*vr,
    format_p(p23),
    0.022*vr
)


# ============================================================
# C
# ============================================================

axC = fig.add_subplot(gs[0,2])

color_boxplot(
    axC,
    "sphericity",
    "Sphericity"
)

panel_label(axC, "C")

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

smin = df["sphericity"].min()
smax = df["sphericity"].max()
sr = smax-smin

axC.set_ylim(
    smin-0.08*sr,
    smax+0.34*sr
)

significance_bar(
    axC,
    1,2,
    smax+0.09*sr,
    format_p(p12),
    0.018*sr
)

significance_bar(
    axC,
    1,3,
    smax+0.22*sr,
    format_p(p13),
    0.018*sr
)


# ============================================================
# D
# ============================================================

axD = fig.add_subplot(gs[1,0])

scatter_panel(
    axD,
    "max_feret_diameter_mm",
    "mesh_volume_mm3",
    "Maximum Feret diameter (mm)",
    "Aneurysm volume (mm³)",
    BLUE,
    LIGHT_BLUE
)

panel_label(axD, "D")

axD.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda x, pos:
        f"{x/1000:.0f}k"
    )
)


# ============================================================
# E
# ============================================================

axE = fig.add_subplot(gs[1,1])

scatter_panel(
    axE,
    "max_feret_diameter_mm",
    "tortuosity",
    "Maximum Feret diameter (mm)",
    "Tortuosity",
    ORANGE,
    LIGHT_ORANGE
)

panel_label(axE, "E")


# ============================================================
# F
# ============================================================

axF = fig.add_subplot(gs[1,2])

im = axF.imshow(
    rho_matrix,
    cmap="Spectral_r",
    vmin=-1,
    vmax=1,
    aspect="equal"
)

axF.set_xticks(np.arange(n))
axF.set_yticks(np.arange(n))

axF.set_xticklabels(
    HEATMAP_LABELS,
    rotation=42,
    ha="right",
    fontsize=8.2
)

axF.set_yticklabels(
    HEATMAP_LABELS,
    fontsize=8.4
)

axF.tick_params(length=0)

axF.set_xticks(
    np.arange(-0.5,n,1),
    minor=True
)

axF.set_yticks(
    np.arange(-0.5,n,1),
    minor=True
)

axF.grid(
    which="minor",
    color="white",
    linewidth=1.15
)

axF.tick_params(
    which="minor",
    bottom=False,
    left=False
)

for i in range(n):
    for j in range(n):

        rho = rho_matrix[i,j]

        if i == j:
            txt = "1.00"

        else:
            p = p_matrix[i,j]

            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = ""

            txt = f"{rho:.2f}{stars}"

        color = (
            "white"
            if abs(rho) >= 0.60
            else "black"
        )

        axF.text(
            j,
            i,
            txt,
            ha="center",
            va="center",
            fontsize=7.4,
            color=color
        )

panel_label(axF, "F")

cbar = fig.colorbar(
    im,
    ax=axF,
    fraction=0.043,
    pad=0.045
)

cbar.set_label(
    r"Spearman $\rho$",
    fontsize=9.4
)

cbar.ax.tick_params(
    labelsize=8
)

# Put significance note well below heatmap
axF.text(
    0.5,
    -0.40,
    (
        "* Holm-adjusted P<0.05; "
        "** P<0.01; "
        "*** P<0.001"
    ),
    transform=axF.transAxes,
    ha="center",
    va="top",
    fontsize=7.3
)


# ============================================================
# SAVE
# ============================================================

png = Path(str(OUTBASE) + ".png")
pdf = Path(str(OUTBASE) + ".pdf")
svg = Path(str(OUTBASE) + ".svg")

fig.savefig(
    png,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white"
)

fig.savefig(
    pdf,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white"
)

fig.savefig(
    svg,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white"
)

plt.close(fig)

print("="*90)
print("COLOR FIGURE V3 FINISHED")
print("="*90)
print("PNG:", png)
print("PDF:", pdf)
print("SVG:", svg)
