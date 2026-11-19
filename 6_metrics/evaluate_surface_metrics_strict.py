from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt
)


ROOT = Path("/home/ubuntu/AAA_CTA_Project")


MODELS = {

    "3D U-Net":
        ROOT / "8_results/predictions/unet3d_fold",

    "Attention U-Net":
        ROOT / "8_results/predictions/attention_unet3d_fold",

    "MSAU w/o Multiscale":
        ROOT / "8_results/predictions/msau_no_multiscale_fold",

    "MSAU w/o MSA":
        ROOT / "8_results/predictions/msau_msa_only_fold",

    "MSAU-Net":
        ROOT / "8_results/predictions/msau_unet3d_v2_fold",
}


GT_ROOT = ROOT / "3_preprocessed"

OUT_DIR = ROOT / "9_results/surface_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# overlap metrics
# ==========================================================

def dice_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()

    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0

    return 2.0 * inter / denom


def iou_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return inter / union


def precision_score_binary(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()

    if tp + fp == 0:
        return 1.0 if gt.sum() == 0 else 0.0

    return tp / (tp + fp)


def recall_score_binary(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    if tp + fn == 0:
        return 1.0

    return tp / (tp + fn)


# ==========================================================
# strict surface extraction
# ==========================================================

def extract_surface(mask):

    mask = mask.astype(bool)

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0
    )

    surface = np.logical_and(
        mask,
        ~eroded
    )

    return surface


# ==========================================================
# strict symmetric surface distances
# ==========================================================

def surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_surface = extract_surface(pred)
    gt_surface = extract_surface(gt)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return None, None

    # distance to nearest GT surface voxel
    dt_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )

    # distance to nearest prediction surface voxel
    dt_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    pred_to_gt = dt_gt[
        pred_surface
    ]

    gt_to_pred = dt_pred[
        gt_surface
    ]

    return pred_to_gt, gt_to_pred


def hd95_surface(pred, gt, spacing=(1.0, 1.0, 1.0)):

    d1, d2 = surface_distances(
        pred,
        gt,
        spacing
    )

    if d1 is None:
        return np.nan

    all_distances = np.concatenate(
        [d1, d2]
    )

    return float(
        np.percentile(
            all_distances,
            95
        )
    )


def assd_surface(pred, gt, spacing=(1.0, 1.0, 1.0)):

    d1, d2 = surface_distances(
        pred,
        gt,
        spacing
    )

    if d1 is None:
        return np.nan

    return float(
        (
            d1.mean()
            +
            d2.mean()
        ) / 2.0
    )


# ==========================================================
# find GT
# ==========================================================

def find_gt(case_id):

    matches = list(
        GT_ROOT.rglob(
            f"{case_id}_crop_lab.npy"
        )
    )

    if len(matches) == 0:

        raise FileNotFoundError(
            f"GT not found: {case_id}"
        )

    return matches[0]


# ==========================================================
# main evaluation
# ==========================================================

all_case_results = []


for model_name, pred_root in MODELS.items():

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    pred_files = sorted(
        pred_root.rglob(
            "*_pred.npy"
        )
    )

    print(
        "Prediction count:",
        len(pred_files)
    )

    for pred_file in pred_files:

        case_id = pred_file.stem.replace(
            "_pred",
            ""
        )

        gt_file = find_gt(
            case_id
        )

        pred = np.load(
            pred_file
        ) > 0

        gt = np.load(
            gt_file
        ) > 0


        if pred.shape != gt.shape:

            print(
                "[SKIP SHAPE]",
                case_id,
                pred.shape,
                gt.shape
            )

            continue


        # --------------------------------------------------
        # IMPORTANT:
        # preprocessing was resampled to approximately
        # isotropic 1 mm space, therefore use 1 mm spacing.
        #
        # If you later confirm another resampled spacing,
        # replace this tuple.
        # --------------------------------------------------

        spacing = (
            1.0,
            1.0,
            1.0
        )


        dice = dice_score(
            pred,
            gt
        )

        iou = iou_score(
            pred,
            gt
        )

        precision = precision_score_binary(
            pred,
            gt
        )

        recall = recall_score_binary(
            pred,
            gt
        )

        hd95 = hd95_surface(
            pred,
            gt,
            spacing
        )

        assd = assd_surface(
            pred,
            gt,
            spacing
        )


        print(
            f"{case_id:10s} "
            f"Dice={dice:.4f} "
            f"IoU={iou:.4f} "
            f"HD95={hd95:.4f} "
            f"ASSD={assd:.4f}"
        )


        all_case_results.append(
            {
                "Method": model_name,
                "case_id": case_id,
                "Dice": dice,
                "IoU": iou,
                "Precision": precision,
                "Recall": recall,
                "HD95_mm": hd95,
                "ASSD_mm": assd,
            }
        )


# ==========================================================
# case-level CSV
# ==========================================================

case_df = pd.DataFrame(
    all_case_results
)


case_csv = (
    OUT_DIR /
    "strict_surface_case_metrics.csv"
)


case_df.to_csv(
    case_csv,
    index=False
)


# ==========================================================
# model-level summary
# ==========================================================

summary_rows = []


metrics = [
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "HD95_mm",
    "ASSD_mm",
]


for model_name in MODELS.keys():

    x = case_df[
        case_df["Method"] == model_name
    ]


    row = {
        "Method": model_name,
        "N": len(x)
    }


    for metric in metrics:

        row[f"{metric}_mean"] = (
            x[metric].mean()
        )

        row[f"{metric}_std"] = (
            x[metric].std(ddof=1)
        )


    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_csv = (
    OUT_DIR /
    "strict_surface_model_summary.csv"
)


summary_df.to_csv(
    summary_csv,
    index=False
)


# ==========================================================
# print
# ==========================================================

print("\n")
print("=" * 100)
print("STRICT SURFACE-BASED RESULTS")
print("=" * 100)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

print(
    summary_df.to_string(
        index=False
    )
)


print("\nSaved:")
print(case_csv)
print(summary_csv)

