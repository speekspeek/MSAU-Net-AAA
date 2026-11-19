import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "8_results/tables"
OUT.mkdir(parents=True, exist_ok=True)

def dice(pred, gt):
    pred, gt = pred > 0, gt > 0
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2 * inter / denom

def iou(pred, gt):
    pred, gt = pred > 0, gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union

def precision(pred, gt):
    pred, gt = pred > 0, gt > 0
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return 1.0 if tp + fp == 0 else tp / (tp + fp)

def recall(pred, gt):
    pred, gt = pred > 0, gt > 0
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return 1.0 if tp + fn == 0 else tp / (tp + fn)

def surface(mask):
    mask = mask > 0
    if mask.sum() == 0:
        return mask
    return mask ^ binary_erosion(mask)

def hd95_assd(pred, gt):
    pred, gt = pred > 0, gt > 0
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan

    ps, gs = surface(pred), surface(gt)
    dt_gt = distance_transform_edt(~gs)
    dt_pr = distance_transform_edt(~ps)
    d = np.concatenate([dt_gt[ps], dt_pr[gs]])

    return np.percentile(d, 95), d.mean()

rows = []

for fold in range(1, 6):
    csv_path = ROOT / "2_dataset/folds" / f"fold{fold}_test.csv"
    pred_dir = ROOT / "8_results/predictions" / f"fold{fold}_attention_unet3d_fullmodel"

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        case_id = row["case_id"]
        gt = np.load(ROOT / row["label_npy"])
        pred = np.load(pred_dir / f"{case_id}_pred.npy")

        h, a = hd95_assd(pred, gt)

        rows.append({
            "fold": fold,
            "case_id": case_id,
            "class": row["class"],
            "Dice": dice(pred, gt),
            "IoU": iou(pred, gt),
            "Precision": precision(pred, gt),
            "Recall": recall(pred, gt),
            "HD95_voxel": h,
            "ASSD_voxel": a,
            "pred_voxels": int((pred > 0).sum()),
            "gt_voxels": int((gt > 0).sum())
        })

case_df = pd.DataFrame(rows)
case_df.to_csv(OUT / "attention_unet3d_5fold_fullmodel_case_metrics.csv", index=False)

fold_df = case_df.groupby("fold")[[
    "Dice", "IoU", "Precision", "Recall", "HD95_voxel", "ASSD_voxel"
]].mean().reset_index()
fold_df.to_csv(OUT / "attention_unet3d_5fold_fullmodel_fold_metrics.csv", index=False)

summary = fold_df[[
    "Dice", "IoU", "Precision", "Recall", "HD95_voxel", "ASSD_voxel"
]].agg(["mean", "std"])
summary.to_csv(OUT / "attention_unet3d_5fold_fullmodel_mean_std.csv")

print("Case metrics:")
print(case_df)
print("\nFold metrics:")
print(fold_df)
print("\nMean ± SD:")
print(summary)
