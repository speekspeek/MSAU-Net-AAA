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

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTBASE = (
    OUTDIR /
    "Figure_AAA100_core_morphological_phenotype_v4"
)


# ============================================================
# 2. COLORS
# ============================================================

GROUP_COLORS = {
    "<50 mm": "#4C78A8",
    "50–54.9 mm": "#F2A541",
    "≥55 mm": "#D95F5F",
}

BLUE = "#2878B5"
ORANGE = "#F39C34"

LIGHT_BLUE = "#BFD7EA"
LIGHT_ORANGE = "#F8D7A0"

DARK = "#303030"


# ============================================================
# 3. GLOBAL STYLE
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
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

posthoc = pd.read_csv(
    POSTHOC_FILE
)

corr_results = pd.read_csv(
    CORR_FILE
)

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
# 5. REPRESENTATIVE CASE SELECTION
# ============================================================

def select_median_case(
    data,
    group
):

    g = data[
        data["diameter_group"] == group
    ].copy()

    median_value = g[
        "max_feret_diameter_mm"
    ].median()

    idx = (
        g["max_feret_diameter_mm"]
        - median_value
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
        f"{group:12s} | "
        f"{r['case_id']} | "
        f"D={r['max_feret_diameter_mm']:.2f} mm | "
        f"Volume={r['mesh_volume_mm3']/1000:.2f} cm³"
    )


# ============================================================
# 6. STL LOADER
# ============================================================

def load_stl_mesh(
    path
):

    import trimesh

    mesh = trimesh.load(
        path,
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


# ============================================================
# 7. 3D VIEW
# ============================================================

def set_equal_3d(
    ax,
    vertices
):

    mins = vertices.min(
        axis=0
    )

    maxs = vertices.max(
        axis=0
    )

    center = (
        mins + maxs
    ) / 2

    spans = (
        maxs - mins
    )

    # Smaller radius = visually larger vessel
    radius = spans.max() * 0.34

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


def draw_mesh(
    ax,
    stl_path,
    color
):

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, faces = load_stl_mesh(
        stl_path
    )

    # Reduce rendering load only
    if len(faces) > 70000:

        step = int(
            np.ceil(
                len(faces) / 70000
            )
        )

        faces_plot = faces[
            ::step
        ]

    else:

        faces_plot = faces

    triangles = vertices[
        faces_plot
    ]

    poly = Poly3DCollection(
        triangles,
        linewidths=0.035,
        edgecolors=color,
        alpha=0.98
    )

    poly.set_facecolor(
        color
    )

    ax.add_collection3d(
        poly
    )

    set_equal_3d(
        ax,
        vertices
    )

    ax.view_init(
        elev=8,
        azim=-72
    )

    ax.set_axis_off()

    # Matplotlib 3D zoom
    try:
        ax.dist = 5.8
    except Exception:
        pass


# ============================================================
# 8. PANEL LABEL
# ============================================================

def panel_label(
    ax,
    text
):

    ax.text(
        -0.10,
        1.055,
        text,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top"
    )


# ============================================================
# 9. CLEAN AXES
# ============================================================

def clean_axes(
    ax
):

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.8
    )


# ============================================================
# 10. COLOR BOXPLOT
# ============================================================

def color_boxplot(
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
        positions=[
            1,
            2,
            3
        ],
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

    for patch, group in zip(
        bp["boxes"],
        GROUP_ORDER
    ):

        patch.set_facecolor(
            GROUP_COLORS[group]
        )

        patch.set_alpha(
            0.72
        )

    rng = np.random.default_rng(
        20260901
    )

    for i, (
        group,
        values
    ) in enumerate(
        zip(
            GROUP_ORDER,
            arrays
        ),
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

    ax.set_xticks(
        [
            1,
            2,
            3
        ]
    )

    ax.set_xticklabels(
        GROUP_ORDER
    )

    ax.set_xlabel(
        "Maximum Feret diameter group",
        labelpad=5
    )

    ax.set_ylabel(
        ylabel,
        labelpad=6
    )

    clean_axes(
        ax
    )


# ============================================================
# 11. POST-HOC
# ============================================================

def get_posthoc_p(
    metric,
    g1,
    g2
):

    r = posthoc[
        (
            posthoc["metric"] == metric
        )
        &
        (
            (
                (
                    posthoc["group1"] == g1
                )
                &
                (
                    posthoc["group2"] == g2
                )
            )
            |
            (
                (
                    posthoc["group1"] == g2
                )
                &
                (
                    posthoc["group2"] == g1
                )
            )
        )
    ]

    return float(
        r.iloc[0]["p_holm"]
    )


def format_p(
    p
):

    if p < 0.001:

        return (
            r"$P_{\mathrm{adj}}<0.001$"
        )

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
        [
            x1,
            x1,
            x2,
            x2
        ],
        [
            y,
            y+h,
            y+h,
            y
        ],
        color=DARK,
        linewidth=0.8,
        clip_on=False
    )

    ax.text(
        (
            x1+x2
        ) / 2,
        y+h*1.35,
        text,
        ha="center",
        va="bottom",
        fontsize=7.7,
        clip_on=False
    )


# ============================================================
# 12. CORRELATION LOOKUP
# ============================================================

def get_corr_result(
    xvar,
    yvar
):

    r = corr_results[
        (
            (
                corr_results[
                    "variable_1"
                ] == xvar
            )
            &
            (
                corr_results[
                    "variable_2"
                ] == yvar
            )
        )
        |
        (
            (
                corr_results[
                    "variable_1"
                ] == yvar
            )
            &
            (
                corr_results[
                    "variable_2"
                ] == xvar
            )
        )
    ]

    if len(r) == 1:

        return (
            float(
                r.iloc[0][
                    "spearman_rho"
                ]
            ),
            float(
                r.iloc[0][
                    "p_holm"
                ]
            )
        )

    return spearmanr(
        df[xvar],
        df[yvar]
    )


# ============================================================
# 13. BOOTSTRAP CI
# ============================================================

def bootstrap_linear_ci(
    x,
    y,
    xgrid,
    n_boot=1500,
    seed=20260901
):

    rng = np.random.default_rng(
        seed
    )

    predictions = []

    for _ in range(
        n_boot
    ):

        idx = rng.integers(
            0,
            len(x),
            len(x)
        )

        xb = x[idx]
        yb = y[idx]

        if np.std(
            xb
        ) == 0:

            continue

        coef = np.polyfit(
            xb,
            yb,
            1
        )

        predictions.append(
            coef[0] * xgrid
            +
            coef[1]
        )

    predictions = np.asarray(
        predictions
    )

    low = np.percentile(
        predictions,
        2.5,
        axis=0
    )

    high = np.percentile(
        predictions,
        97.5,
        axis=0
    )

    return low, high


# ============================================================
# 14. SCATTER PANEL
# ============================================================

def scatter_panel(
    ax,
    xvar,
    yvar,
    xlabel,
    ylabel,
    point_color,
    band_color
):

    x = df[
        xvar
    ].to_numpy(
        float
    )

    y = df[
        yvar
    ].to_numpy(
        float
    )

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

    coef = np.polyfit(
        x,
        y,
        1
    )

    xx = np.linspace(
        x.min(),
        x.max(),
        250
    )

    yy = (
        coef[0] * xx
        +
        coef[1]
    )

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

    if p < 0.001:

        ptxt = (
            r"Holm-adjusted $P<0.001$"
        )

    else:

        ptxt = (
            rf"Holm-adjusted $P={p:.3f}$"
        )

    # Put statistics ABOVE plotting area
    ax.set_title(
        (
            rf"Spearman $\rho={rho:.3f}$; "
            + ptxt
        ),
        fontsize=9.2,
        pad=10,
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

    clean_axes(
        ax
    )


# ============================================================
# 15. HEATMAP DATA
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

n = len(
    HEATMAP_METRICS
)

rho_matrix = np.eye(
    n
)

p_matrix = np.zeros(
    (
        n,
        n
    )
)

for i in range(
    n
):

    for j in range(
        i+1,
        n
    ):

        rho, p = get_corr_result(
            HEATMAP_METRICS[i],
            HEATMAP_METRICS[j]
        )

        rho_matrix[
            i,
            j
        ] = rho

        rho_matrix[
            j,
            i
        ] = rho

        p_matrix[
            i,
            j
        ] = p

        p_matrix[
            j,
            i
        ] = p


# ============================================================
# 16. FIGURE LAYOUT
# ============================================================

fig = plt.figure(
    figsize=(
        14.8,
        9.4
    ),
    facecolor="white"
)

gs = GridSpec(
    2,
    3,
    figure=fig,

    width_ratios=[
        1.16,
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
# 17. PANEL A
# ============================================================

# ------------------------------------------------------------
# A is deliberately split vertically:
#
# row 1 = group heading
# row 2 = 3D model
# row 3 = case / diameter text
#
# This completely avoids text overlapping with the mesh.
# ------------------------------------------------------------

gsA_outer = GridSpecFromSubplotSpec(
    3,
    1,
    subplot_spec=gs[
        0,
        0
    ],
    height_ratios=[
        0.13,
        0.72,
        0.15
    ],
    hspace=0.00
)


# -------------------------
# A1. Group headings
# -------------------------

gsA_top = GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gsA_outer[
        0,
        0
    ],
    wspace=0.04
)

for i, group in enumerate(
    GROUP_ORDER
):

    ax = fig.add_subplot(
        gsA_top[
            0,
            i
        ]
    )

    ax.set_axis_off()

    ax.text(
        0.5,
        0.35,
        group,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=GROUP_COLORS[group]
    )


# -------------------------
# A2. 3D models
# -------------------------

gsA_mesh = GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gsA_outer[
        1,
        0
    ],
    wspace=-0.04
)

for i, group in enumerate(
    GROUP_ORDER
):

    r = representatives[
        group
    ]

    ax = fig.add_subplot(
        gsA_mesh[
            0,
            i
        ],
        projection="3d"
    )

    draw_mesh(
        ax,
        r[
            "stl_file"
        ],
        GROUP_COLORS[
            group
        ]
    )


# -------------------------
# A3. Case labels BELOW mesh
# -------------------------

gsA_bottom = GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gsA_outer[
        2,
        0
    ],
    wspace=0.04
)

for i, group in enumerate(
    GROUP_ORDER
):

    r = representatives[
        group
    ]

    ax = fig.add_subplot(
        gsA_bottom[
            0,
            i
        ]
    )

    ax.set_axis_off()

    ax.text(
        0.5,
        0.65,
        r[
            "case_id"
        ],
        ha="center",
        va="center",
        fontsize=8.5,
        color=DARK
    )

    ax.text(
        0.5,
        0.18,
        (
            f"D = "
            f"{r['max_feret_diameter_mm']:.1f} mm"
        ),
        ha="center",
        va="center",
        fontsize=8.2,
        color=DARK
    )


# Transparent overlay axis for A label and bottom title
axA = fig.add_subplot(
    gs[
        0,
        0
    ]
)

axA.patch.set_alpha(
    0
)

for spine in axA.spines.values():
    spine.set_visible(
        False
    )

axA.set_xticks([])
axA.set_yticks([])

panel_label(
    axA,
    "A"
)

axA.text(
    0.50,
    -0.035,
    "Representative 3D aneurysm geometries",
    transform=axA.transAxes,
    ha="center",
    va="top",
    fontsize=10.5
)


# ============================================================
# 18. PANEL B
# ============================================================

axB = fig.add_subplot(
    gs[
        0,
        1
    ]
)

color_boxplot(
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

vmin = df[
    "mesh_volume_mm3"
].min()

vmax = df[
    "mesh_volume_mm3"
].max()

vrange = (
    vmax - vmin
)

axB.set_ylim(
    vmin - 0.08 * vrange,
    vmax + 0.34 * vrange
)

significance_bar(
    axB,
    1,
    3,
    vmax + 0.10 * vrange,
    format_p(
        p13
    ),
    0.022 * vrange
)

significance_bar(
    axB,
    2,
    3,
    vmax + 0.23 * vrange,
    format_p(
        p23
    ),
    0.022 * vrange
)


# ============================================================
# 19. PANEL C
# ============================================================

axC = fig.add_subplot(
    gs[
        0,
        2
    ]
)

color_boxplot(
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

smin = df[
    "sphericity"
].min()

smax = df[
    "sphericity"
].max()

srange = (
    smax - smin
)

axC.set_ylim(
    smin - 0.08 * srange,
    smax + 0.34 * srange
)

significance_bar(
    axC,
    1,
    2,
    smax + 0.09 * srange,
    format_p(
        p12
    ),
    0.018 * srange
)

significance_bar(
    axC,
    1,
    3,
    smax + 0.22 * srange,
    format_p(
        p13
    ),
    0.018 * srange
)


# ============================================================
# 20. PANEL D
# ============================================================

axD = fig.add_subplot(
    gs[
        1,
        0
    ]
)

scatter_panel(
    axD,

    "max_feret_diameter_mm",
    "mesh_volume_mm3",

    "Maximum Feret diameter (mm)",
    "Aneurysm volume (mm³)",

    BLUE,
    LIGHT_BLUE
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
# 21. PANEL E
# ============================================================

axE = fig.add_subplot(
    gs[
        1,
        1
    ]
)

scatter_panel(
    axE,

    "max_feret_diameter_mm",
    "tortuosity",

    "Maximum Feret diameter (mm)",
    "Tortuosity",

    ORANGE,
    LIGHT_ORANGE
)

panel_label(
    axE,
    "E"
)


# ============================================================
# 22. PANEL F
# ============================================================

axF = fig.add_subplot(
    gs[
        1,
        2
    ]
)

im = axF.imshow(
    rho_matrix,
    cmap="Spectral_r",
    vmin=-1,
    vmax=1,
    aspect="equal"
)

axF.set_xticks(
    np.arange(
        n
    )
)

axF.set_yticks(
    np.arange(
        n
    )
)

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

axF.tick_params(
    length=0
)

axF.set_xticks(
    np.arange(
        -0.5,
        n,
        1
    ),
    minor=True
)

axF.set_yticks(
    np.arange(
        -0.5,
        n,
        1
    ),
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


# ------------------------------------------------------------
# Heatmap numbers
# ------------------------------------------------------------

for i in range(
    n
):

    for j in range(
        n
    ):

        rho = rho_matrix[
            i,
            j
        ]

        if i == j:

            txt = (
                "1.00"
            )

        else:

            p = p_matrix[
                i,
                j
            ]

            if p < 0.001:
                stars = "***"

            elif p < 0.01:
                stars = "**"

            elif p < 0.05:
                stars = "*"

            else:
                stars = ""

            txt = (
                f"{rho:.2f}"
                f"{stars}"
            )

        txt_color = (
            "white"
            if abs(
                rho
            ) >= 0.60
            else "black"
        )

        axF.text(
            j,
            i,
            txt,
            ha="center",
            va="center",
            fontsize=7.4,
            color=txt_color
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
    pad=0.045
)

cbar.set_label(
    r"Spearman $\rho$",
    fontsize=9.4
)

cbar.ax.tick_params(
    labelsize=8
)


# ------------------------------------------------------------
# P-value note
# ------------------------------------------------------------

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
# 23. EXPORT
# ============================================================

png_file = Path(
    str(
        OUTBASE
    ) + ".png"
)

pdf_file = Path(
    str(
        OUTBASE
    ) + ".pdf"
)

svg_file = Path(
    str(
        OUTBASE
    ) + ".svg"
)

fig.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white"
)

fig.savefig(
    pdf_file,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white"
)

fig.savefig(
    svg_file,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white"
)

plt.close(
    fig
)


# ============================================================
# 24. FINISHED
# ============================================================

print("\n" + "=" * 90)
print("FIGURE V4 FINISHED")
print("=" * 90)

print(
    "PNG:",
    png_file
)

print(
    "PDF:",
    pdf_file
)

print(
    "SVG:",
    svg_file
)

print("\nA-panel layout:")
print("Top    : diameter group")
print("Middle : enlarged 3D geometry")
print("Bottom : case ID + diameter")
print("No text overlaps the 3D model.")

print("=" * 90)
