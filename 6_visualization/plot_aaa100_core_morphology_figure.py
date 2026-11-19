#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AAA-100 Core Morphological Phenotype Figure

Panel A: Representative 3D aneurysm geometries from three diameter groups
Panel B: Aneurysm volume across diameter groups
Panel C: Sphericity across diameter groups
Panel D: Maximum Feret diameter vs aneurysm volume
Panel E: Maximum Feret diameter vs tortuosity
Panel F: Spearman correlation heatmap

Outputs:
    PNG 600 dpi
    PDF vector
    SVG vector
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import PolyCollection
from matplotlib import ticker

from scipy.stats import spearmanr


# =====================================================================
# 1. PATHS
# =====================================================================

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

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTBASE = OUTDIR / "Figure_AAA100_core_morphological_phenotype"


# =====================================================================
# 2. GLOBAL STYLE
# =====================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif"
    ],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =====================================================================
# 3. LOAD DATA
# =====================================================================

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


# =====================================================================
# 4. REPRESENTATIVE CASE SELECTION
# =====================================================================

def select_median_case(data, group_name):
    """
    Select the case whose maximum Feret diameter is closest
    to the median value of its diameter group.
    """

    g = data[
        data["diameter_group"] == group_name
    ].copy()

    median_value = g[
        "max_feret_diameter_mm"
    ].median()

    idx = (
        g["max_feret_diameter_mm"] -
        median_value
    ).abs().idxmin()

    return g.loc[idx]


representatives = {
    group: select_median_case(
        df,
        group
    )
    for group in GROUP_ORDER
}


print("=" * 90)
print("REPRESENTATIVE CASES")
print("=" * 90)

for group in GROUP_ORDER:

    r = representatives[group]

    print(
        f"{group:12s}: "
        f"{r['case_id']} | "
        f"Feret={r['max_feret_diameter_mm']:.2f} mm | "
        f"Volume={r['mesh_volume_mm3'] / 1000:.2f} cm³"
    )


# =====================================================================
# 5. SIMPLE ASCII STL READER
#    Falls back to trimesh if available.
# =====================================================================

def load_stl_mesh(stl_path):

    stl_path = Path(stl_path)

    try:
        import trimesh

        mesh = trimesh.load(
            stl_path,
            force="mesh",
            process=False
        )

        vertices = np.asarray(
            mesh.vertices
        )

        faces = np.asarray(
            mesh.faces
        )

        return vertices, faces

    except Exception:

        # -------------------------------------------------------------
        # Minimal ASCII STL fallback
        # -------------------------------------------------------------

        vertices = []
        faces = []
        vertex_map = {}

        current_face = []

        with open(
            stl_path,
            "r",
            errors="ignore"
        ) as f:

            for line in f:

                line = line.strip()

                if line.startswith("vertex"):

                    parts = line.split()

                    xyz = tuple(
                        float(x)
                        for x in parts[1:4]
                    )

                    if xyz not in vertex_map:

                        vertex_map[xyz] = len(vertices)
                        vertices.append(xyz)

                    current_face.append(
                        vertex_map[xyz]
                    )

                    if len(current_face) == 3:

                        faces.append(
                            current_face
                        )

                        current_face = []

        if len(vertices) == 0:

            raise RuntimeError(
                f"Could not load STL: {stl_path}\n"
                "Install trimesh with:\n"
                "pip install trimesh"
            )

        return (
            np.asarray(vertices),
            np.asarray(faces)
        )


# =====================================================================
# 6. 3D MESH DRAWING
# =====================================================================

def set_axes_equal_3d(ax, vertices):

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    center = (
        mins + maxs
    ) / 2.0

    radius = (
        maxs - mins
    ).max() / 2.0

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


def draw_mesh(ax, stl_path, title):

    vertices, faces = load_stl_mesh(
        stl_path
    )

    # Limit extremely dense mesh rendering
    max_faces = 50000

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

    poly = PolyCollection(
        triangles[:, :, :2],
        edgecolors="none"
    )

    # PolyCollection cannot directly represent true 3D polygons.
    # Replace with Poly3DCollection.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    poly3d = Poly3DCollection(
        triangles,
        linewidths=0.05,
        edgecolors="0.30",
        alpha=0.92
    )

    poly3d.set_facecolor(
        "0.72"
    )

    ax.add_collection3d(
        poly3d
    )

    set_axes_equal_3d(
        ax,
        vertices
    )

    ax.view_init(
        elev=15,
        azim=-65
    )

    ax.set_title(
        title,
        fontsize=9,
        pad=2
    )

    ax.set_axis_off()


# =====================================================================
# 7. BOXPLOT FUNCTION
# =====================================================================

def make_boxplot(
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
        labels=GROUP_ORDER,
        widths=0.60,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "linewidth": 1.8,
            "color": "black"
        },
        boxprops={
            "linewidth": 1.0
        },
        whiskerprops={
            "linewidth": 1.0
        },
        capprops={
            "linewidth": 1.0
        }
    )

    # grayscale-friendly fills
    shades = [
        "0.90",
        "0.72",
        "0.52"
    ]

    for patch, shade in zip(
        bp["boxes"],
        shades
    ):

        patch.set_facecolor(
            shade
        )

    # Individual observations
    rng = np.random.default_rng(
        20260901
    )

    for i, values in enumerate(
        arrays,
        start=1
    ):

        jitter = rng.normal(
            i,
            0.045,
            len(values)
        )

        ax.scatter(
            jitter,
            values,
            s=10,
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.5,
            alpha=0.65,
            zorder=3
        )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlabel(
        "Maximum Feret diameter group"
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    ax.tick_params(
        axis="x",
        rotation=0
    )


# =====================================================================
# 8. P-VALUE ANNOTATION
# =====================================================================

def p_text(p):

    if p < 0.001:
        return "P$_{adj}$ < 0.001"

    return (
        f"P$_{{adj}}$ = {p:.3f}"
    )


def add_significance_bar(
    ax,
    x1,
    x2,
    y,
    text,
    height_fraction=0.025
):

    ymin, ymax = ax.get_ylim()

    h = (
        ymax - ymin
    ) * height_fraction

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        linewidth=0.8,
        color="black",
        clip_on=False
    )

    ax.text(
        (x1 + x2) / 2,
        y + h * 1.15,
        text,
        ha="center",
        va="bottom",
        fontsize=8
    )


# =====================================================================
# 9. FIND POST-HOC P VALUE
# =====================================================================

def get_posthoc_p(
    metric,
    g1,
    g2
):

    r = posthoc[
        (posthoc["metric"] == metric) &
        (
            (
                (posthoc["group1"] == g1) &
                (posthoc["group2"] == g2)
            )
            |
            (
                (posthoc["group1"] == g2) &
                (posthoc["group2"] == g1)
            )
        )
    ]

    if len(r) == 0:
        return np.nan

    return float(
        r.iloc[0]["p_holm"]
    )


# =====================================================================
# 10. SCATTER FUNCTION
# =====================================================================

def scatter_with_trend(
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

    ax.scatter(
        x,
        y,
        s=23,
        facecolors="0.80",
        edgecolors="black",
        linewidths=0.45,
        alpha=0.85
    )

    # Simple visual trend line.
    # Statistical inference is still based on Spearman correlation.
    coef = np.polyfit(
        x,
        y,
        1
    )

    xx = np.linspace(
        x.min(),
        x.max(),
        200
    )

    yy = (
        coef[0] * xx +
        coef[1]
    )

    ax.plot(
        xx,
        yy,
        linewidth=1.3,
        color="black"
    )

    # Use previously Holm-corrected statistics
    match = corr_results[
        (
            (
                corr_results["variable_1"] == xvar
            )
            &
            (
                corr_results["variable_2"] == yvar
            )
        )
        |
        (
            (
                corr_results["variable_1"] == yvar
            )
            &
            (
                corr_results["variable_2"] == xvar
            )
        )
    ]

    if len(match) == 1:

        rho = float(
            match.iloc[0][
                "spearman_rho"
            ]
        )

        p_adj = float(
            match.iloc[0][
                "p_holm"
            ]
        )

    else:

        rho, p_adj = spearmanr(
            x,
            y
        )

    if p_adj < 0.001:

        p_string = (
            "Holm-adjusted P < 0.001"
        )

    else:

        p_string = (
            f"Holm-adjusted P = {p_adj:.3f}"
        )

    ax.text(
        0.04,
        0.96,
        (
            f"Spearman ρ = {rho:.3f}\n"
            f"{p_string}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.7",
            "linewidth": 0.7,
            "alpha": 0.90
        }
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)


# =====================================================================
# 11. CORRELATION MATRIX
# =====================================================================

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


rho_matrix = pd.DataFrame(
    np.eye(
        len(HEATMAP_METRICS)
    ),
    index=HEATMAP_METRICS,
    columns=HEATMAP_METRICS
)

p_matrix = pd.DataFrame(
    np.zeros(
        (
            len(HEATMAP_METRICS),
            len(HEATMAP_METRICS)
        )
    ),
    index=HEATMAP_METRICS,
    columns=HEATMAP_METRICS
)


for i in range(
    len(HEATMAP_METRICS)
):

    for j in range(
        i + 1,
        len(HEATMAP_METRICS)
    ):

        a = HEATMAP_METRICS[i]
        b = HEATMAP_METRICS[j]

        match = corr_results[
            (
                (
                    corr_results["variable_1"] == a
                )
                &
                (
                    corr_results["variable_2"] == b
                )
            )
            |
            (
                (
                    corr_results["variable_1"] == b
                )
                &
                (
                    corr_results["variable_2"] == a
                )
            )
        ]

        if len(match) == 1:

            rho = float(
                match.iloc[0][
                    "spearman_rho"
                ]
            )

            p_adj = float(
                match.iloc[0][
                    "p_holm"
                ]
            )

        else:

            rho, p_adj = spearmanr(
                df[a],
                df[b]
            )

        rho_matrix.loc[
            a,
            b
        ] = rho

        rho_matrix.loc[
            b,
            a
        ] = rho

        p_matrix.loc[
            a,
            b
        ] = p_adj

        p_matrix.loc[
            b,
            a
        ] = p_adj


# =====================================================================
# 12. FIGURE LAYOUT
# =====================================================================

fig = plt.figure(
    figsize=(15.2, 10.2)
)

gs = GridSpec(
    2,
    3,
    figure=fig,
    left=0.055,
    right=0.965,
    bottom=0.075,
    top=0.955,
    wspace=0.33,
    hspace=0.36
)


# =====================================================================
# PANEL A
# =====================================================================

gs_a = GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gs[0, 0],
    wspace=0.02
)

for idx, group in enumerate(
    GROUP_ORDER
):

    ax = fig.add_subplot(
        gs_a[0, idx],
        projection="3d"
    )

    r = representatives[group]

    title = (
        f"{group}\n"
        f"{r['case_id']}\n"
        f"D={r['max_feret_diameter_mm']:.1f} mm"
    )

    draw_mesh(
        ax,
        r["stl_file"],
        title
    )


# Invisible overlay axis for panel label
axA_label = fig.add_subplot(
    gs[0, 0]
)

axA_label.set_axis_off()

axA_label.text(
    -0.08,
    1.06,
    "A",
    transform=axA_label.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)

axA_label.text(
    0.50,
    -0.04,
    "Representative 3D aneurysm geometries",
    transform=axA_label.transAxes,
    fontsize=11,
    ha="center"
)


# =====================================================================
# PANEL B — VOLUME
# =====================================================================

axB = fig.add_subplot(
    gs[0, 1]
)

make_boxplot(
    axB,
    "mesh_volume_mm3",
    "Aneurysm volume (mm³)"
)

axB.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda x, pos:
        f"{x / 1000:.0f}k"
    )
)

axB.text(
    -0.15,
    1.06,
    "B",
    transform=axB.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)

# Significant post-hoc comparisons
p_b1 = get_posthoc_p(
    "mesh_volume_mm3",
    "<50 mm",
    "≥55 mm"
)

p_b2 = get_posthoc_p(
    "mesh_volume_mm3",
    "50–54.9 mm",
    "≥55 mm"
)

ymin, ymax = axB.get_ylim()
yrange = ymax - ymin

axB.set_ylim(
    ymin,
    ymax + 0.22 * yrange
)

add_significance_bar(
    axB,
    1,
    3,
    ymax + 0.05 * yrange,
    p_text(p_b1)
)

add_significance_bar(
    axB,
    2,
    3,
    ymax + 0.14 * yrange,
    p_text(p_b2)
)


# =====================================================================
# PANEL C — SPHERICITY
# =====================================================================

axC = fig.add_subplot(
    gs[0, 2]
)

make_boxplot(
    axC,
    "sphericity",
    "Sphericity"
)

axC.text(
    -0.15,
    1.06,
    "C",
    transform=axC.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)

p_c1 = get_posthoc_p(
    "sphericity",
    "<50 mm",
    "50–54.9 mm"
)

p_c2 = get_posthoc_p(
    "sphericity",
    "<50 mm",
    "≥55 mm"
)

ymin, ymax = axC.get_ylim()
yrange = ymax - ymin

axC.set_ylim(
    ymin,
    ymax + 0.22 * yrange
)

add_significance_bar(
    axC,
    1,
    2,
    ymax + 0.05 * yrange,
    p_text(p_c1)
)

add_significance_bar(
    axC,
    1,
    3,
    ymax + 0.14 * yrange,
    p_text(p_c2)
)


# =====================================================================
# PANEL D — DIAMETER vs VOLUME
# =====================================================================

axD = fig.add_subplot(
    gs[1, 0]
)

scatter_with_trend(
    axD,
    "max_feret_diameter_mm",
    "mesh_volume_mm3",
    "Maximum Feret diameter (mm)",
    "Aneurysm volume (mm³)"
)

axD.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda x, pos:
        f"{x / 1000:.0f}k"
    )
)

axD.text(
    -0.15,
    1.06,
    "D",
    transform=axD.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)


# =====================================================================
# PANEL E — DIAMETER vs TORTUOSITY
# =====================================================================

axE = fig.add_subplot(
    gs[1, 1]
)

scatter_with_trend(
    axE,
    "max_feret_diameter_mm",
    "tortuosity",
    "Maximum Feret diameter (mm)",
    "Tortuosity"
)

axE.text(
    -0.15,
    1.06,
    "E",
    transform=axE.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)


# =====================================================================
# PANEL F — CORRELATION HEATMAP
# =====================================================================

axF = fig.add_subplot(
    gs[1, 2]
)

im = axF.imshow(
    rho_matrix.values,
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    aspect="equal"
)

axF.set_xticks(
    np.arange(
        len(HEATMAP_LABELS)
    )
)

axF.set_yticks(
    np.arange(
        len(HEATMAP_LABELS)
    )
)

axF.set_xticklabels(
    HEATMAP_LABELS,
    rotation=45,
    ha="right"
)

axF.set_yticklabels(
    HEATMAP_LABELS
)

for i in range(
    len(HEATMAP_METRICS)
):

    for j in range(
        len(HEATMAP_METRICS)
    ):

        rho = rho_matrix.iloc[
            i,
            j
        ]

        p = p_matrix.iloc[
            i,
            j
        ]

        if i == j:

            text = "1.00"

        else:

            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""

            text = (
                f"{rho:.2f}"
                f"{star}"
            )

        text_color = (
            "white"
            if abs(rho) >= 0.60
            else "black"
        )

        axF.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color=text_color
        )

axF.text(
    -0.18,
    1.06,
    "F",
    transform=axF.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)

cbar = fig.colorbar(
    im,
    ax=axF,
    fraction=0.046,
    pad=0.04
)

cbar.set_label(
    "Spearman ρ",
    fontsize=10
)

axF.text(
    0.5,
    -0.37,
    (
        "* Holm-adjusted P<0.05; "
        "** P<0.01; *** P<0.001"
    ),
    transform=axF.transAxes,
    ha="center",
    va="top",
    fontsize=8
)


# =====================================================================
# 13. SAVE — THREE FORMATS
# =====================================================================

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
    bbox_inches="tight"
)

fig.savefig(
    pdf_file,
    bbox_inches="tight"
)

fig.savefig(
    svg_file,
    bbox_inches="tight"
)

plt.close(
    fig
)


# =====================================================================
# 14. OUTPUT
# =====================================================================

print("\n" + "=" * 90)
print("FIGURE EXPORT FINISHED")
print("=" * 90)

print("PNG :", png_file)
print("PDF :", pdf_file)
print("SVG :", svg_file)

print("\nPanel definitions:")
print("A  Representative 3D aneurysm geometries")
print("B  Aneurysm volume across diameter groups")
print("C  Sphericity across diameter groups")
print("D  Maximum diameter versus aneurysm volume")
print("E  Maximum diameter versus tortuosity")
print("F  Spearman morphology correlation matrix")

print("=" * 90)
