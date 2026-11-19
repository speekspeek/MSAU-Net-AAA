from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path("/home/ubuntu/AAA_CTA_Project")

# ============================================================
# Cases shown in the main manuscript figure
# ============================================================

CASES = [
    "Anev_05",
    "Anev_07",
    "Anev_09",
]


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
# Publication style
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


# ============================================================
# Utilities
# ============================================================

def find_prediction(pred_root, case_id):

    files = list(
        pred_root.rglob(
            f"{case_id}_pred.npy"
        )
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"Prediction not found: {case_id} in {pred_root}"
        )

    if len(files) > 1:
        print(
            f"[Warning] Multiple predictions found for {case_id}:"
        )

        for f in files:
            print("   ", f)

    return files[0]


def load_case(case_id):

    image_path = (
        ROOT /
        "3_preprocessed/Anev/images" /
        f"{case_id}_crop_img.npy"
    )

    gt_path = (
        ROOT /
        "3_preprocessed/Anev/labels" /
        f"{case_id}_crop_lab.npy"
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    image = np.load(image_path).astype(np.float32)

    gt = np.load(gt_path) > 0


    preds = {}

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
                f"{case_id} / {model_name}: "
                f"pred shape {pred.shape} != GT shape {gt.shape}"
            )


        preds[model_name] = pred


    return image, gt, preds


def select_representative_slice(gt):

    """
    Select the axial slice with the largest GT area.

    Data format used in this project:
    (Y, X, Z)
    """

    area_per_slice = gt.sum(
        axis=(0, 1)
    )

    if np.max(area_per_slice) == 0:

        return gt.shape[2] // 2


    return int(
        np.argmax(area_per_slice)
    )


def normalize_ct(image_2d):

    """
    Robust display normalization using 1st-99th percentiles.
    """

    x = image_2d.astype(np.float32)

    low = np.percentile(
        x,
        1
    )

    high = np.percentile(
        x,
        99
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
# Plotting functions
# ============================================================

def draw_ct(ax, img):

    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax.axis("off")


def draw_gt(ax, img, gt):

    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=1
    )


    if gt.any():

        overlay = np.ma.masked_where(
            ~gt,
            gt
        )

        ax.imshow(
            overlay,
            cmap=ListedColormap(
                ["#64D77A"]
            ),
            alpha=0.32,
            interpolation="nearest"
        )


        ax.contour(
            gt.astype(float),
            levels=[0.5],
            colors=["#00A94F"],
            linewidths=1.35
        )


    ax.axis("off")


def draw_prediction(ax, img, gt, pred):

    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=1
    )


    # Prediction region
    if pred.any():

        overlay = np.ma.masked_where(
            ~pred,
            pred
        )

        ax.imshow(
            overlay,
            cmap=ListedColormap(
                ["#F28B82"]
            ),
            alpha=0.30,
            interpolation="nearest"
        )


    # GT boundary
    if gt.any():

        ax.contour(
            gt.astype(float),
            levels=[0.5],
            colors=["#00B050"],
            linewidths=1.25
        )


    # Prediction boundary
    if pred.any():

        ax.contour(
            pred.astype(float),
            levels=[0.5],
            colors=["#D62728"],
            linewidths=1.05
        )


    ax.axis("off")


# ============================================================
# Load all representative cases
# ============================================================

loaded = {}


for case_id in CASES:

    image, gt, preds = load_case(
        case_id
    )


    z = select_representative_slice(
        gt
    )


    image_2d = normalize_ct(
        image[:, :, z]
    )


    loaded[case_id] = {

        "image": image_2d,

        "gt": gt[:, :, z],

        "preds": {
            method: pred[:, :, z]
            for method, pred in preds.items()
        },

        "slice": z,
    }


    print(
        f"{case_id}: "
        f"selected slice = {z}, "
        f"GT area = {gt[:, :, z].sum()}"
    )


# ============================================================
# Figure layout
# ============================================================

COLUMN_NAMES = [

    "(A) Image",

    "(B) Ground Truth",

    "(C) 3D U-Net",

    "(D) Attention U-Net",

    "(E) MSAU-Net",
]


n_rows = len(CASES)

n_cols = len(COLUMN_NAMES)


fig, axes = plt.subplots(

    n_rows,

    n_cols,

    figsize=(11.5, 6.8),

    dpi=300
)


if n_rows == 1:

    axes = axes[
        np.newaxis,
        :
    ]


# ============================================================
# Draw panels
# ============================================================

for row_idx, case_id in enumerate(CASES):

    item = loaded[
        case_id
    ]

    img = item[
        "image"
    ]

    gt = item[
        "gt"
    ]


    # A - CTA
    draw_ct(
        axes[row_idx, 0],
        img
    )


    # B - Ground truth
    draw_gt(
        axes[row_idx, 1],
        img,
        gt
    )


    # C - 3D U-Net
    draw_prediction(
        axes[row_idx, 2],
        img,
        gt,
        item["preds"]["3D U-Net"]
    )


    # D - Attention U-Net
    draw_prediction(
        axes[row_idx, 3],
        img,
        gt,
        item["preds"]["Attention U-Net"]
    )


    # E - MSAU-Net
    draw_prediction(
        axes[row_idx, 4],
        img,
        gt,
        item["preds"]["MSAU-Net"]
    )


    # case label
    axes[row_idx, 0].text(

        -0.065,

        0.50,

        case_id,

        transform=axes[row_idx, 0].transAxes,

        rotation=90,

        va="center",

        ha="right",

        fontsize=9,

        fontweight="bold"
    )


# ============================================================
# Column headers
# ============================================================

for col_idx, title in enumerate(COLUMN_NAMES):

    axes[0, col_idx].set_title(

        title,

        fontsize=10,

        pad=8
    )


# ============================================================
# Legend
# ============================================================

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


legend_elements = [

    Patch(
        facecolor="#64D77A",
        edgecolor="none",
        alpha=0.50,
        label="Ground-truth region"
    ),

    Patch(
        facecolor="#F28B82",
        edgecolor="none",
        alpha=0.50,
        label="Predicted region"
    ),

    Line2D(
        [0],
        [0],
        color="#00B050",
        lw=1.5,
        label="Ground-truth boundary"
    ),

    Line2D(
        [0],
        [0],
        color="#D62728",
        lw=1.3,
        label="Predicted boundary"
    ),
]


fig.legend(

    handles=legend_elements,

    loc="lower center",

    bbox_to_anchor=(
        0.5,
        0.005
    ),

    ncol=4,

    frameon=False,

    fontsize=8
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(

    left=0.065,

    right=0.995,

    top=0.93,

    bottom=0.105,

    wspace=0.025,

    hspace=0.045
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Figure2_qualitative_comparison"
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


print("\nFigure 2 saved:")

print(PNG)

print(SVG)

print(PDF)


plt.close(fig)

