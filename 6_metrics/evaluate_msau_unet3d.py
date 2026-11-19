import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "2_dataset/finetune_test.csv"
PRED_DIR = ROOT / "8_results/predictions/msau_unet3d"
OUT_DIR = ROOT / "8_results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def dice_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2 * inter / denom

def iou_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union

def precision_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return 1.0 if tp + fp == 0 else tp / (tp + fp)

def recall_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return 1.0 if tp + fn == 0 else tp / (tp + fn)

rows = []

for _, row in df.iterrows():
    case_id = row["case_id"]
    gt = np.load(ROOT / row["label_npy"])
    pred = np.load(PRED_DIR / f"{case_id}_pred.npy")

    rows.append({
        "case_id": case_id,
        "Dice": dice_score(pred, gt),
        "IoU": iou_score(pred, gt),
        "Precision": precision_score(pred, gt),
        "Recall": recall_score(pred, gt),
        "pred_voxels": int((pred > 0).sum()),
        "gt_voxels": int((gt > 0).sum())
    })

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "msau_unet3d_test_metrics.csv", index=False)

summary = res[["Dice", "IoU", "Precision", "Recall"]].mean().to_frame("Mean").T
summary.to_csv(OUT_DIR / "msau_unet3d_test_metrics_summary.csv", index=False)

print(res)
print("\nMean metrics:")
print(summary)
print("\nSaved:", OUT_DIR)
