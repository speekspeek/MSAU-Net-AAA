import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "2_dataset/finetune_test.csv"
PRED_DIR = ROOT / "8_results/predictions/msau_unet3d"
OUT_DIR = ROOT / "8_results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def surface(mask):
    mask = mask > 0
    if mask.sum() == 0:
        return mask
    return mask ^ binary_erosion(mask)

def hd95_assd(pred, gt):
    pred = pred > 0
    gt = gt > 0
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan

    ps = surface(pred)
    gs = surface(gt)

    dt_gt = distance_transform_edt(~gs)
    dt_pr = distance_transform_edt(~ps)

    d1 = dt_gt[ps]
    d2 = dt_pr[gs]
    d = np.concatenate([d1, d2])

    return np.percentile(d, 95), d.mean()

rows = []
for _, row in df.iterrows():
    case_id = row["case_id"]
    gt = np.load(ROOT / row["label_npy"])
    pred = np.load(PRED_DIR / f"{case_id}_pred.npy")

    hd95, assd = hd95_assd(pred, gt)
    rows.append({
        "case_id": case_id,
        "HD95_voxel": hd95,
        "ASSD_voxel": assd
    })

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "msau_unet3d_hd95_assd.csv", index=False)

summary = res[["HD95_voxel", "ASSD_voxel"]].mean().to_frame("Mean").T
summary.to_csv(OUT_DIR / "msau_unet3d_hd95_assd_summary.csv", index=False)

print(res)
print("\nMean:")
print(summary)
