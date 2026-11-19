from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = Path("/home/ubuntu/AAA_CTA_Project")

# ============================================================
# Representative AAA cases
# ============================================================

CASES = [
    "Anev_05",
    "Anev_07",
    "Anev_09",
]


# ============================================================
# Prediction directories
# ============================================================

PRED_DIRS = {

    "3D U-Net":
        ROOT / "8_results/predictions/unet3d_fold",

    "Attention U-Net":
        ROOT / "8_results/predictions/attention_unet3d_fold",

    "MSAU-Net":
        ROOT / "8_results/predictions/msau_unet3d_v2_fold",
}


OUT_DIR = ROOT / "9_results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Publication settings
# ============================================================

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10

# Keep text editable in SVG
plt.rcParams["svg.fonttype"] = "none"

# TrueType embedding in PDF
plt.rcParams["pdf.fonttype"] = 42

plt.rcParams["axes.linewidth"] = 0.5


# ============================================================
# Colors
# ============================================================

PRED_FILL = "#E64B35"
GT_FILL = "#E64B35"

GT_BOUNDARY = "#00A651"

# fill transparencies
GT_ALPHA = 0.48
PRED_ALPHA = 0.42


# ============================================================
# Helper: find prediction across fold directories
# ============================================================

def find_prediction(pred_root, case_id):

    files = sorted(
        pred_root.rglob(
            f"{case_id}_pred.npy"
        )
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"Prediction not found:\n"
            f"case={case_id}\n"
            f"directory={pred_root}"
        )

    if len(files) > 1:

        print(
            f"[WARNING] Multiple predictions found for {case_id}:"
        )

        for f in files:
            print("  ", f)

        print(
            "Using:",
            files[0]
        )

    return files[0]


# ============================================================
# Load case
# ============================================================

def load_case(case_id):

    image_path = (
        ROOT
        / "3_preprocessed/Anev/images"
        / f"{case_id}_crop_img.npy"
    )

    gt_path = (
        ROOT
        / "3_preprocessed/Anev/labels"
        / f"{case_id}_crop_lab.npy"
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    image = np.load(
        image_path
    ).astype(np.float32)

    gt = np.load(
        gt_path
    ) > 0

    predictions = {}

    for model_name, pred_root in PRED_DIRS.items():

        pred_path = find_prediction(
            pred_root,
            case_id
        )

        pred = np.load(
            pred_path
        ) > 0

        if pred.shape != gt.shape:

            raise ValueError(
                f"\nShape mismatch for {case_id} / {model_name}\n"
                f"Prediction: {pred.shape}\n"
                f"GT:         {gt.shape}"
            )

        predictions[
            model_name
        ] = pred

    return image, gt, predictions


# ============================================================
# Select representative axial slice
# ============================================================

def select_slice(gt):

    """
    Data ordering:
    Y, X, Z

    Select slice with maximum GT cross-sectional area.
    """

    areas = gt.sum(
        axis=(0, 1)
    )

    if areas.max() == 0:
        return gt.shape[2] // 2

    return int(
        np.argmax(areas)
    )


# ============================================================
# CTA display normalization
# ============================================================

def normalize_ct(x):

    x = x.astype(
        np.float32
    )

    # robust image display
    low = np.percentile(
        x,
        1.0
    )

    high = np.percentile(
        x,
        99.0
    )

    if high <= low:
        return x

    x = np.clip(
        x,
        low,
        high
    )

    x = (
        x - low
    ) / (
        high - low
    )

    return x


# ============================================================
# Ground-truth panel
# ============================================================

def plot_gt(
        ax,
        image,
        gt
):

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    if gt.any():

        masked = np.ma.masked_where(
            ~gt,
            gt
        )

        ax.imshow(
            masked,
            cmap=ListedColormap(
                [GT_FILL]
            ),
            alpha=GT_ALPHA,
            interpolation="nearest"
        )

        ax.contour(
            gt.astype(float),
            levels=[0.5],
            colors=[GT_BOUNDARY],
            linewidths=1.05
        )

    ax.set_axis_off()


# ============================================================
# Prediction panel
# ============================================================

def plot_prediction(
        ax,
        image,
        gt,
        pred
):

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    # predicted segmentation
    if pred.any():

        masked_pred = np.ma.masked_where(
            ~pred,
            pred
        )

        ax.imshow(
            masked_pred,
            cmap=ListedColormap(
                [PRED_FILL]
            ),
            alpha=PRED_ALPHA,
            interpolation="nearest"
        )

    # GT contour over prediction
    if gt.any():

        ax.contour(
            gt.astype(float),
            levels=[0.5],
            colors=[GT_BOUNDARY],
            linewidths=1.10
        )

    ax.set_axis_off()


# ============================================================
# Load all data first
# ============================================================

DATA = {}


for case_id in CASES:

    image, gt, predictions = load_case(
        case_id
    )

    z = select_slice(
        gt
    )

    image_slice = normalize_ct(
        image[:, :, z]
    )

    gt_slice = gt[
        :,
        :,
        z
    ]

    pred_slices = {

        name:
            pred[:, :, z]

        for name, pred in predictions.items()
    }

    DATA[case_id] = {

        "image":
            image_slice,

        "gt":
            gt_slice,

        "predictions":
            pred_slices,

        "slice":
            z,
    }

    print(
        f"{case_id}: "
        f"slice={z}, "
        f"GT area={int(gt_slice.sum())}"
    )


# ============================================================
# Figure layout
# ============================================================

COLUMN_NAMES = [

    "Image",

    "GT",

    "3D U-Net",

    "Attention U-Net",

    "MSAU-Net",
]


N_ROWS = len(
    CASES
)

N_COLS = len(
    COLUMN_NAMES
)


fig, axes = plt.subplots(

    N_ROWS,

    N_COLS,

    figsize=(10.4, 6.3),

    dpi=300
)


if N_ROWS == 1:

    axes = axes[
        np.newaxis,
        :
    ]


# ============================================================
# Draw figure
# ============================================================

for row_idx, case_id in enumerate(CASES):

    item = DATA[
        case_id
    ]

    image = item[
        "image"
    ]

    gt = item[
        "gt"
    ]

    predictions = item[
        "predictions"
    ]


    # --------------------------------------------------------
    # Column A: Original CTA
    # --------------------------------------------------------

    axes[
        row_idx,
        0
    ].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[
        row_idx,
        0
    ].set_axis_off()


    # --------------------------------------------------------
    # Column B: Ground truth
    # --------------------------------------------------------

    plot_gt(

        axes[
            row_idx,
            1
        ],

        image,

        gt
    )


    # --------------------------------------------------------
    # Column C: 3D U-Net
    # --------------------------------------------------------

    plot_prediction(

        axes[
            row_idx,
            2
        ],

        image,

        gt,

        predictions[
            "3D U-Net"
        ]
    )


    # --------------------------------------------------------
    # Column D: Attention U-Net
    # --------------------------------------------------------

    plot_prediction(

        axes[
            row_idx,
            3
        ],

        image,

        gt,

        predictions[
            "Attention U-Net"
        ]
    )


    # --------------------------------------------------------
    # Column E: MSAU-Net
    # --------------------------------------------------------

    plot_prediction(

        axes[
            row_idx,
            4
        ],

        image,

        gt,

        predictions[
            "MSAU-Net"
        ]
    )


# ============================================================
# Column headings
# ============================================================

for col_idx, title in enumerate(
        COLUMN_NAMES
):

    axes[
        0,
        col_idx
    ].set_title(

        title,

        fontsize=11,

        pad=5
    )


# ============================================================
# Optional case labels on far-left margin
# ============================================================

for row_idx, case_id in enumerate(
        CASES
):

    axes[
        row_idx,
        0
    ].text(

        -0.08,

        0.5,

        case_id,

        transform=axes[
            row_idx,
            0
        ].transAxes,

        rotation=90,

        ha="right",

        va="center",

        fontsize=8.5
    )


# ============================================================
# Compact legend
# ============================================================

legend_elements = [

    Patch(
        facecolor=PRED_FILL,
        edgecolor="none",
        alpha=0.65,
        label="Segmented AAA"
    ),

    Line2D(
        [0],
        [0],
        color=GT_BOUNDARY,
        linewidth=1.5,
        label="Ground-truth boundary"
    ),
]


fig.legend(

    handles=legend_elements,

    loc="lower center",

    bbox_to_anchor=(
        0.5,
        0.012
    ),

    ncol=2,

    frameon=False,

    fontsize=9,

    handlelength=1.5,

    handletextpad=0.5,

    columnspacing=2.0
)


# ============================================================
# Tight journal-like spacing
# ============================================================

plt.subplots_adjust(

    left=0.065,

    right=0.995,

    top=0.925,

    bottom=0.105,

    wspace=0.025,

    hspace=0.035
)


# ============================================================
# Save Figure 2
# ============================================================

BASE = (
    OUT_DIR
    / "Figure2_qualitative_segmentation_final"
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

    pad_inches=0.03
)


fig.savefig(

    SVG,

    format="svg",

    bbox_inches="tight",

    facecolor="white",

    pad_inches=0.03
)


fig.savefig(

    PDF,

    format="pdf",

    bbox_inches="tight",

    facecolor="white",

    pad_inches=0.03
)


print("\n==========================================")
print("Figure 2 completed")
print("==========================================")

print(PNG)
print(SVG)
print(PDF)


plt.close(fig)

