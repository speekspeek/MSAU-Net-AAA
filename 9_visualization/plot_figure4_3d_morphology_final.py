from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from skimage.measure import marching_cubes, mesh_surface_area

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

PRED_ROOT = (
    ROOT /
    "8_results/predictions/msau_unet3d_v2_fold"
)

IMAGE_ROOT = (
    ROOT /
    "3_preprocessed/Anev/images"
)

GT_ROOT = (
    ROOT /
    "3_preprocessed/Anev/labels"
)

OUT_DIR = (
    ROOT /
    "9_results/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MEASURE_OUT = (
    ROOT /
    "9_results/Figure4_morphology_measurements.csv"
)


# ============================================================
# Representative AAA cases
# ============================================================

CASES = [
    "Anev_05",
    "Anev_07",
]


# ============================================================
# Resampled voxel spacing
# ============================================================
# Current preprocessing is approximately 1-mm isotropic.
# Array order is Y, X, Z.

SPACING = np.array(
    [1.0, 1.0, 1.0],
    dtype=float
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
plt.rcParams["axes.linewidth"] = 0.7

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# Colors
# ============================================================

SEGMENT_COLOR = "#DF6F66"
GT_COLOR = "#2FA65A"

SURFACE_COLOR = "#D8897C"
SURFACE_EDGE = "#9A4F48"

DIAMETER_COLOR = "#F2B134"

TEXT_COLOR = "#222222"
MUTED_COLOR = "#666666"


# ============================================================
# File utilities
# ============================================================

def find_prediction(case_id):

    files = sorted(
        PRED_ROOT.rglob(
            f"{case_id}_pred.npy"
        )
    )

    if len(files) == 0:

        raise FileNotFoundError(
            f"Prediction not found for {case_id}"
        )

    return files[0]


def load_case(case_id):

    image_path = (
        IMAGE_ROOT /
        f"{case_id}_crop_img.npy"
    )

    gt_path = (
        GT_ROOT /
        f"{case_id}_crop_lab.npy"
    )

    pred_path = find_prediction(
        case_id
    )

    image = np.load(
        image_path
    ).astype(np.float32)

    gt = np.load(
        gt_path
    ) > 0

    pred = np.load(
        pred_path
    ) > 0

    if pred.shape != gt.shape:

        raise ValueError(
            f"{case_id}: "
            f"prediction shape {pred.shape} "
            f"!= GT shape {gt.shape}"
        )

    return image, gt, pred


# ============================================================
# CT normalization
# ============================================================

def normalize_ct(x):

    x = x.astype(
        np.float32
    )

    p1 = np.percentile(
        x,
        1
    )

    p99 = np.percentile(
        x,
        99
    )

    if p99 <= p1:
        return x

    x = np.clip(
        x,
        p1,
        p99
    )

    return (
        x - p1
    ) / (
        p99 - p1
    )


# ============================================================
# Representative axial slice
# ============================================================

def select_largest_slice(mask):

    area = mask.sum(
        axis=(0, 1)
    )

    if area.max() == 0:

        return (
            mask.shape[2] // 2
        )

    return int(
        np.argmax(area)
    )


# ============================================================
# Maximum Feret diameter in largest axial section
# ============================================================

def max_feret_diameter(mask):

    z = select_largest_slice(
        mask
    )

    section = mask[
        :,
        :,
        z
    ]

    if section.sum() == 0:

        return (
            np.nan,
            z,
            None,
            None
        )

    # surface pixels
    eroded = binary_erosion(
        section,
        structure=np.ones(
            (3, 3),
            dtype=bool
        )
    )

    boundary = (
        section &
        (~eroded)
    )

    points = np.column_stack(
        np.where(boundary)
    )

    if len(points) < 2:

        return (
            np.nan,
            z,
            None,
            None
        )

    # Convex hull reduces pairwise calculations
    if len(points) >= 3:

        try:

            hull = ConvexHull(
                points
            )

            hull_points = points[
                hull.vertices
            ]

        except Exception:

            hull_points = points

    else:

        hull_points = points


    # convert to millimetres
    points_mm = (
        hull_points.astype(float)
        *
        SPACING[:2]
    )

    dist_matrix = squareform(
        pdist(points_mm)
    )

    idx = np.unravel_index(
        np.argmax(dist_matrix),
        dist_matrix.shape
    )

    p1 = hull_points[
        idx[0]
    ]

    p2 = hull_points[
        idx[1]
    ]

    diameter = float(
        dist_matrix[idx]
    )

    return (
        diameter,
        z,
        p1,
        p2
    )


# ============================================================
# Morphological measurements
# ============================================================

def compute_morphology(mask):

    voxel_volume = float(
        np.prod(SPACING)
    )

    volume_mm3 = float(
        mask.sum() *
        voxel_volume
    )

    if mask.sum() == 0:

        return {
            "volume_mm3": 0.0,
            "surface_area_mm2": 0.0,
            "sphericity": np.nan,
            "max_feret_diameter_mm": np.nan
        }


    vertices, faces, _, _ = marching_cubes(

        mask.astype(
            np.float32
        ),

        level=0.5,

        spacing=tuple(
            SPACING
        )
    )


    area_mm2 = float(
        mesh_surface_area(
            vertices,
            faces
        )
    )


    if area_mm2 > 0:

        sphericity = float(
            (
                np.pi ** (1.0 / 3.0)
            )
            *
            (
                (6.0 * volume_mm3)
                ** (2.0 / 3.0)
            )
            /
            area_mm2
        )

    else:

        sphericity = np.nan


    diameter, _, _, _ = (
        max_feret_diameter(
            mask
        )
    )


    return {

        "volume_mm3":
            volume_mm3,

        "surface_area_mm2":
            area_mm2,

        "sphericity":
            sphericity,

        "max_feret_diameter_mm":
            diameter,
    }


# ============================================================
# 2D panel
# ============================================================

def draw_2d_panel(
        ax,
        image,
        gt,
        pred,
        z
):

    image_2d = normalize_ct(
        image[:, :, z]
    )

    gt_2d = gt[
        :,
        :,
        z
    ]

    pred_2d = pred[
        :,
        :,
        z
    ]


    ax.imshow(

        image_2d,

        cmap="gray",

        vmin=0,

        vmax=1
    )


    # predicted region
    rgba = np.zeros(
        pred_2d.shape + (4,),
        dtype=float
    )

    rgba[
        pred_2d
    ] = (
        0.87,
        0.43,
        0.40,
        0.36
    )


    ax.imshow(
        rgba,
        interpolation="nearest"
    )


    # GT contour
    if gt_2d.any():

        ax.contour(

            gt_2d.astype(float),

            levels=[0.5],

            colors=[
                GT_COLOR
            ],

            linewidths=1.25
        )


    # prediction contour
    if pred_2d.any():

        ax.contour(

            pred_2d.astype(float),

            levels=[0.5],

            colors=[
                SEGMENT_COLOR
            ],

            linewidths=0.90
        )


    ax.set_axis_off()


# ============================================================
# 3D panel
# ============================================================

def draw_3d_panel(
        ax,
        mask
):

    verts, faces, _, _ = marching_cubes(

        mask.astype(
            np.float32
        ),

        level=0.5,

        spacing=tuple(
            SPACING
        )
    )


    # Reduce polygon density for SVG/PDF size
    max_faces = 18000

    if len(faces) > max_faces:

        step = int(
            np.ceil(
                len(faces) /
                max_faces
            )
        )

        plot_faces = faces[
            ::step
        ]

    else:

        plot_faces = faces


    mesh = Poly3DCollection(

        verts[
            plot_faces
        ],

        alpha=0.82,

        linewidth=0.08
    )

    mesh.set_facecolor(
        SURFACE_COLOR
    )

    mesh.set_edgecolor(
        SURFACE_EDGE
    )

    ax.add_collection3d(
        mesh
    )


    # --------------------------------------------------------
    # Maximum axial Feret diameter
    # --------------------------------------------------------

    diameter, z, p1, p2 = (
        max_feret_diameter(
            mask
        )
    )


    if (
        p1 is not None
        and
        p2 is not None
    ):

        # p = [Y, X]
        point1 = np.array(
            [
                p1[0] * SPACING[0],
                p1[1] * SPACING[1],
                z * SPACING[2]
            ]
        )

        point2 = np.array(
            [
                p2[0] * SPACING[0],
                p2[1] * SPACING[1],
                z * SPACING[2]
            ]
        )


        ax.plot(

            [
                point1[0],
                point2[0]
            ],

            [
                point1[1],
                point2[1]
            ],

            [
                point1[2],
                point2[2]
            ],

            color=DIAMETER_COLOR,

            linewidth=3.2,

            zorder=20
        )


        ax.scatter(

            [
                point1[0],
                point2[0]
            ],

            [
                point1[1],
                point2[1]
            ],

            [
                point1[2],
                point2[2]
            ],

            color=DIAMETER_COLOR,

            s=18,

            depthshade=False
        )


        midpoint = (
            point1 + point2
        ) / 2.0


        ax.text(

            midpoint[0],

            midpoint[1],

            midpoint[2],

            f"  {diameter:.1f} mm",

            fontsize=8,

            color="#6A4700",

            fontweight="bold"
        )


    # --------------------------------------------------------
    # Equal-ish 3D limits
    # --------------------------------------------------------

    mins = verts.min(
        axis=0
    )

    maxs = verts.max(
        axis=0
    )

    centers = (
        mins + maxs
    ) / 2.0

    ranges = (
        maxs - mins
    )

    radius = (
        np.max(ranges) /
        2.0
    )


    ax.set_xlim(
        centers[0] - radius,
        centers[0] + radius
    )

    ax.set_ylim(
        centers[1] - radius,
        centers[1] + radius
    )

    ax.set_zlim(
        centers[2] - radius,
        centers[2] + radius
    )


    try:

        ax.set_box_aspect(
            (
                1,
                1,
                1.15
            )
        )

    except Exception:

        pass


    ax.view_init(

        elev=20,

        azim=-58
    )


    ax.set_axis_off()


# ============================================================
# Morphology measurement panel
# ============================================================

def draw_measurement_panel(
        ax,
        case_id,
        features
):

    ax.set_axis_off()


    ax.text(

        0.04,

        0.91,

        "Automated morphology",

        transform=ax.transAxes,

        fontsize=11,

        fontweight="bold",

        color=TEXT_COLOR
    )


    ax.text(

        0.04,

        0.82,

        case_id,

        transform=ax.transAxes,

        fontsize=9,

        color=MUTED_COLOR
    )


    metrics = [

        (
            "Maximum diameter",
            f"{features['max_feret_diameter_mm']:.1f} mm"
        ),

        (
            "Volume",
            f"{features['volume_mm3']/1000.0:.1f} mL"
        ),

        (
            "Surface area",
            f"{features['surface_area_mm2']:.0f} mm²"
        ),

        (
            "Sphericity",
            f"{features['sphericity']:.3f}"
        ),
    ]


    y_positions = [
        0.67,
        0.51,
        0.35,
        0.19
    ]


    for (
        label,
        value
    ), y in zip(
        metrics,
        y_positions
    ):

        ax.text(

            0.05,

            y,

            label,

            transform=ax.transAxes,

            fontsize=8.5,

            color=MUTED_COLOR,

            va="center"
        )


        ax.text(

            0.95,

            y,

            value,

            transform=ax.transAxes,

            fontsize=11,

            color=TEXT_COLOR,

            fontweight="bold",

            ha="right",

            va="center"
        )


        ax.plot(

            [
                0.05,
                0.95
            ],

            [
                y - 0.07,
                y - 0.07
            ],

            transform=ax.transAxes,

            linewidth=0.55,

            color="#D5D5D5"
        )


# ============================================================
# Load data and measurements
# ============================================================

DATA = {}

measurement_rows = []


for case_id in CASES:

    print(
        "Processing",
        case_id
    )


    image, gt, pred = (
        load_case(
            case_id
        )
    )


    features = compute_morphology(
        pred
    )


    z = select_largest_slice(
        pred
    )


    DATA[
        case_id
    ] = {

        "image":
            image,

        "gt":
            gt,

        "pred":
            pred,

        "features":
            features,

        "z":
            z
    }


    measurement_rows.append(
        {
            "case_id":
                case_id,

            "volume_mm3":
                features[
                    "volume_mm3"
                ],

            "volume_ml":
                features[
                    "volume_mm3"
                ] / 1000.0,

            "max_feret_diameter_mm":
                features[
                    "max_feret_diameter_mm"
                ],

            "surface_area_mm2":
                features[
                    "surface_area_mm2"
                ],

            "sphericity":
                features[
                    "sphericity"
                ],
        }
    )


measurement_df = pd.DataFrame(
    measurement_rows
)


measurement_df.to_csv(

    MEASURE_OUT,

    index=False
)


print("\nFigure 4 measurements:")

print(
    measurement_df.to_string(
        index=False
    )
)


# ============================================================
# Create Figure 4
# ============================================================

fig = plt.figure(

    figsize=(
        11.5,
        7.4
    ),

    dpi=300,

    facecolor="white"
)


gs = fig.add_gridspec(

    nrows=2,

    ncols=3,

    width_ratios=[
        1.05,
        1.20,
        0.90
    ],

    height_ratios=[
        1,
        1
    ],

    wspace=0.10,

    hspace=0.18
)


panel_labels = [

    "(A)",
    "(B)",
    "(C)",

    "(D)",
    "(E)",
    "(F)",
]


panel_counter = 0


for row_idx, case_id in enumerate(
        CASES
):

    item = DATA[
        case_id
    ]


    # --------------------------------------------------------
    # 2D CTA + segmentation
    # --------------------------------------------------------

    ax_2d = fig.add_subplot(
        gs[
            row_idx,
            0
        ]
    )


    draw_2d_panel(

        ax_2d,

        item[
            "image"
        ],

        item[
            "gt"
        ],

        item[
            "pred"
        ],

        item[
            "z"
        ]
    )


    ax_2d.text(

        -0.02,

        1.02,

        panel_labels[
            panel_counter
        ],

        transform=ax_2d.transAxes,

        fontsize=11,

        fontweight="bold",

        va="bottom"
    )


    ax_2d.set_title(

        f"{case_id}: CTA and segmentation",

        fontsize=10,

        pad=5
    )


    panel_counter += 1


    # --------------------------------------------------------
    # 3D reconstruction
    # --------------------------------------------------------

    ax_3d = fig.add_subplot(

        gs[
            row_idx,
            1
        ],

        projection="3d"
    )


    draw_3d_panel(

        ax_3d,

        item[
            "pred"
        ]
    )


    ax_3d.text2D(

        -0.02,

        1.02,

        panel_labels[
            panel_counter
        ],

        transform=ax_3d.transAxes,

        fontsize=11,

        fontweight="bold"
    )


    ax_3d.set_title(

        "3D reconstruction and diameter",

        fontsize=10,

        pad=4
    )


    panel_counter += 1


    # --------------------------------------------------------
    # Morphology measurement panel
    # --------------------------------------------------------

    ax_measure = fig.add_subplot(

        gs[
            row_idx,
            2
        ]
    )


    draw_measurement_panel(

        ax_measure,

        case_id,

        item[
            "features"
        ]
    )


    ax_measure.text(

        -0.02,

        1.02,

        panel_labels[
            panel_counter
        ],

        transform=ax_measure.transAxes,

        fontsize=11,

        fontweight="bold",

        va="bottom"
    )


    panel_counter += 1


# ============================================================
# Compact legend
# ============================================================

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


legend_handles = [

    Patch(

        facecolor=SEGMENT_COLOR,

        edgecolor="none",

        alpha=0.55,

        label="MSAU-Net segmentation"
    ),

    Line2D(

        [0],
        [0],

        color=GT_COLOR,

        linewidth=1.5,

        label="Ground-truth boundary"
    ),

    Line2D(

        [0],
        [0],

        color=DIAMETER_COLOR,

        linewidth=3.0,

        label="Maximum Feret diameter"
    ),
]


fig.legend(

    handles=legend_handles,

    loc="lower center",

    bbox_to_anchor=(
        0.5,
        0.015
    ),

    ncol=3,

    frameon=False,

    fontsize=8.5,

    handlelength=2.0,

    columnspacing=2.2
)


plt.subplots_adjust(

    left=0.045,

    right=0.985,

    top=0.955,

    bottom=0.095
)


# ============================================================
# Save PNG / SVG / PDF
# ============================================================

BASE = (
    OUT_DIR /
    "Figure4_3D_morphological_assessment"
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


print("\n==============================================")
print("Figure 4 completed")
print("==============================================")

print(PNG)
print(SVG)
print(PDF)

print("\nMeasurement table:")
print(MEASURE_OUT)


plt.close(
    fig
)

