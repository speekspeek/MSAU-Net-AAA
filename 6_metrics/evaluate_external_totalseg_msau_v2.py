import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from scipy.ndimage import zoom, binary_erosion, distance_transform_edt

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
sys.path.append(str(ROOT))

model_mod = SourceFileLoader(
    "msau_unet3d_v2",
    str(ROOT / "3_models/msau_unet3d_v2.py")
).load_module()

CSV_PATH = ROOT / "2_dataset/TotalSeg_val.csv"
CKPT_PATH = ROOT / "5_checkpoints/finetune/msau_unet3d_v2_best.pth"

OUT_DIR = ROOT / "8_results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CASES = 10
PATCH = (64, 96, 96)
STRIDE = (32, 48, 48)

def normalize_ct(x):
    x = np.clip(x, -200, 800)
    x = (x + 200) / 1000.0
    return x.astype(np.float32)

def resize_to_limit(img, lab, max_shape=(160, 192, 192)):
    scale = min([m / s for m, s in zip(max_shape, img.shape)] + [1.0])
    if scale < 1.0:
        img = zoom(img, scale, order=1)
        lab = zoom(lab, scale, order=0)
    return img, lab

def pad_to_patch(x, patch):
    pads = []
    for s, p in zip(x.shape, patch):
        pads.append((0, max(0, p - s)))
    return np.pad(x, pads, mode="constant"), pads

def sliding_predict(model, img, device):
    img, pads = pad_to_patch(img, PATCH)
    D, H, W = img.shape
    pd, ph, pw = PATCH
    sd, sh, sw = STRIDE

    prob_sum = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    zs = list(range(0, max(D - pd, 0) + 1, sd))
    ys = list(range(0, max(H - ph, 0) + 1, sh))
    xs = list(range(0, max(W - pw, 0) + 1, sw))

    if zs[-1] != D - pd: zs.append(D - pd)
    if ys[-1] != H - ph: ys.append(H - ph)
    if xs[-1] != W - pw: xs.append(W - pw)

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    patch = img[z:z+pd, y:y+ph, x:x+pw]
                    t = torch.from_numpy(patch).float()[None, None].to(device)
                    logits = model(t)
                    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                    prob_sum[z:z+pd, y:y+ph, x:x+pw] += prob
                    count[z:z+pd, y:y+ph, x:x+pw] += 1

    prob = prob_sum / np.maximum(count, 1)
    pred = (prob > 0.5).astype(np.uint8)

    od = pred.shape[0] - pads[0][1]
    oh = pred.shape[1] - pads[1][1]
    ow = pred.shape[2] - pads[2][1]
    return pred[:od, :oh, :ow]

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

def surface(mask):
    mask = mask > 0
    return mask ^ binary_erosion(mask) if mask.sum() > 0 else mask

def hd95_assd(pred, gt):
    pred, gt = pred > 0, gt > 0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan
    ps, gs = surface(pred), surface(gt)
    dt_gt = distance_transform_edt(~gs)
    dt_pr = distance_transform_edt(~ps)
    d = np.concatenate([dt_gt[ps], dt_pr[gs]])
    return float(np.percentile(d, 95)), float(d.mean())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_mod.MSAUNet3DV2(in_channels=1, num_classes=2, base_ch=16).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    df = pd.read_csv(CSV_PATH).head(MAX_CASES)

    rows = []
    for i, r in df.iterrows():
        case_id = Path(r["image_path"]).parent.name
        print("Case:", case_id)

        img = nib.load(str(ROOT / r["image_path"])).get_fdata().astype(np.float32)
        lab = nib.load(str(ROOT / r["label_path"])).get_fdata()
        lab = (lab > 0).astype(np.uint8)

        img = normalize_ct(img)
        img, lab = resize_to_limit(img, lab)

        pred = sliding_predict(model, img, device)

        d = dice(pred, lab)
        j = iou(pred, lab)
        h, a = hd95_assd(pred, lab)

        rows.append({
            "case_id": case_id,
            "Dice": d,
            "IoU": j,
            "HD95_voxel": h,
            "ASSD_voxel": a,
            "pred_voxels": int(pred.sum()),
            "gt_voxels": int(lab.sum())
        })

        print(case_id, "Dice", d, "IoU", j, "HD95", h, "ASSD", a)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "external_totalseg_msau_v2_case_metrics.csv", index=False)

    summary = res[["Dice", "IoU", "HD95_voxel", "ASSD_voxel"]].agg(["mean", "std"])
    summary.to_csv(OUT_DIR / "external_totalseg_msau_v2_summary.csv")

    print("\nSummary:")
    print(summary)
    print("Saved:", OUT_DIR)

if __name__ == "__main__":
    main()
