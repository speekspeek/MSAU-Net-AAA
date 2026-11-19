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
# Ablation prediction directories
# ============================================================

PRED_DIRS = {

    "MSAU w/o Multiscale":
        ROOT / "8_results/predictions/msau_no_multiscale_fold",

    "MSAU w/o MSA":
        ROOT / "8_results/predictions/msau_msa_only_fold",

    "MSAU-Net":
        ROOT / "8_results/predictions/msau_unet3d_v2_fold",
}


OUT_DIR = ROOT / "9_results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
plt.rcParams["axes.linewidth"] = 0.6

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# Colors
# ============================================================

GT_FILL = "#E64B35"
PRED_FILL = "#E64B35"

GT_BOUNDARY = "#00A651"
PRED_BOUNDARY = "#C7382D"

GT_ALPHA = 0.45
PRED_ALPHA = 0.38


# ============================================================
# Utilities
# ============================================================

def find_prediction(pred_root, case_id):

    files = sorted(
        pred_root.rglob(
            f"{case_id}_pred.npy"
        )
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"Prediction not found for "
            f"{case_id} in {pred_root}"
        )

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

    image = np.load(
        image_path
    ).astype(np.float32)

    gt = np.load(
        gt_path
    ) > 0

    preds = {}

    for name, pred_root in PRED_DIRS.items():

        pred_path = find_prediction(
            pred_root,
            case_id
        )

        pred = np.load(
            pred_path
        ) > 0

        if pred.shape != gt.shape:
            raise ValueError(
                f"{case_id} / {name}: "
                f"{pred.shape} != {gt.shape}"
            )

        preds[name] = pred

    return image, gt, preds


def choose_slice(gt):

    area = gt.sum(
        axis=(0, 1)
    )

    if area.max() == 0:
        return gt.shape[2] // 2

    return int(
        np.argmax(area)
    )


def normalize_ct(x):

    x = x.astype(
        np.float32
    )

    lo = np.percentile(
        x,
        1
    )

    hi = np.percentile(
        x,
        99
    )

    if hi <= lo:
        return x

    x = np.clip(
        x,
        lo,
        hi
    )

    return (
        x - lo
    ) / (
        hi - lo
    )


# ============================================================
# Plot functions
# ============================================================

def draw_gt(ax, image, gt):

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
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
            linewidths=1.20
        )

    ax.axis("off")


def draw_prediction(
        ax,
        image,
        gt,
        pred
):

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    if pred.any():

        masked = np.ma.masked_where(
            ~pred,
            pred
        )

        ax.imshow(
            masked,
            cmap=ListedColormap(
                [PRED_FILL]
            ),
            alpha=PRED_ALPHA,
            interpolation="nearest"
        )

        ax.contour(
            pred.astype(float),
            levels=[0.5],
            colors=[PRED_BOUNDARY],
            linewidths=0.95
        )

    if gt.any():

        ax.contour(
            gt.astype(float),
            levels=[0.5],
            colors=[GT_BOUNDARY],
            linewidths=1.20
        )

    ax.axis("off")


# ============================================================
# Load representative slices
# ============================================================

DATA = {}

for case_id in CASES:

    image, gt, preds = load_case(
        case_id
    )

    z = choose_slice(
        gt
    )

    image_2d = normalize_ct(
        image[:, :, z]
    )

    DATA[case_id] = {

        "image":
            image_2d,

        "gt":
            gt[:, :, z],

        "preds": {
            name: pred[:, :, z]
            for name, pred in preds.items()
        },

        "slice":
            z
    }

    print(
        f"{case_id}: "
        f"slice={z}, "
        f"GT area={int(gt[:, :, z].sum())}"
    )


# ============================================================
# Layout
# ============================================================

COLUMNS = [
    "Ground Truth",
    "w/o Multiscale",
    "w/o MSA",
    "MSAU-Net",
]


fig, axes = plt.subplots(
    len(CASES),
    len(COLUMNS),
    figsize=(8.8, 6.3),
    dpi=300
)


# ============================================================
# Draw panels
# ============================================================

for r, case_id in enumerate(CASES):

    item = DATA[
        case_id
    ]

    image = item[
        "image"
    ]

    gt = item[
        "gt"
    ]


    # Ground Truth
    draw_gt(
        axes[r, 0],
        image,
        gt
    )


    # w/o Multiscale
    draw_prediction(
        axes[r, 1],
        image,
        gt,
        item["preds"][
            "MSAU w/o Multiscale"
        ]
    )


    # w/o MSA
    draw_prediction(
        axes[r, 2],
        image,
        gt,
        item["preds"][
            "MSAU w/o MSA"
        ]
    )


    # Full MSAU-Net
    draw_prediction(
        axes[r, 3],
        image,
        gt,
        item["preds"][
            "MSAU-Net"
        ]
    )


    # Case label
    axes[r, 0].text(
        -0.075,
        0.50,
        case_id,
        transform=axes[r, 0].transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=8.5
    )


# ============================================================
# Column headings
# ============================================================

for c, title in enumerate(COLUMNS):

    axes[0, c].set_title(
        title,
        fontsize=10.5,
        pad=6
    )


# ============================================================
# Legend
# ============================================================

legend_handles = [

    Patch(
        facecolor=PRED_FILL,
        edgecolor="none",
        alpha=0.55,
        label="Segmented AAA"
    ),

    Line2D(
        [0],
        [0],
        color=GT_BOUNDARY,
        linewidth=1.5,
        label="Ground-truth boundary"
    ),

    Line2D(
        [0],
        [0],
        color=PRED_BOUNDARY,
        linewidth=1.3,
        label="Predicted boundary"
    ),
]


fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.012),
    ncol=3,
    frameon=False,
    fontsize=8.5,
    handlelength=1.8,
    columnspacing=1.8
)


# ============================================================
# Tight layout
# ============================================================

plt.subplots_adjust(
    left=0.075,
    right=0.995,
    top=0.925,
    bottom=0.11,
    wspace=0.025,
    hspace=0.04
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Supplementary_Figure_S2_ablation_qualitative"
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


print("\n============================================")
print("Supplementary Figure S2 completed")
print("============================================")

print(PNG)
print(SVG)
print(PDF)


plt.close(
    fig
)

