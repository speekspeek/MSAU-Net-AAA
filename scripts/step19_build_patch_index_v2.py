import ast
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
df = pd.read_csv(ROOT / "2_dataset" / "finetune_preprocessed_manifest.csv")

PATCH_SIZE = (96, 96, 96)
STRIDE = (48, 48, 48)

rows = []

def starts(length, patch, stride):
    if length <= patch:
        return [0]
    s = list(range(0, length - patch + 1, stride))
    last = length - patch
    if s[-1] != last:
        s.append(last)
    return sorted(set(s))

for _, r in df.iterrows():
    lab = np.load(ROOT / r["label_npy"])
    z, y, x = lab.shape

    for z0 in starts(z, PATCH_SIZE[0], STRIDE[0]):
        for y0 in starts(y, PATCH_SIZE[1], STRIDE[1]):
            for x0 in starts(x, PATCH_SIZE[2], STRIDE[2]):
                patch_lab = lab[z0:z0+96, y0:y0+96, x0:x0+96]
                fg_voxels = int((patch_lab > 0).sum())

                rows.append({
                    "case_id": r["case_id"],
                    "dataset": r["dataset"],
                    "class": r["class"],
                    "image_npy": r["image_npy"],
                    "label_npy": r["label_npy"],
                    "z0": z0,
                    "y0": y0,
                    "x0": x0,
                    "dz": 96,
                    "dy": 96,
                    "dx": 96,
                    "fg_voxels": fg_voxels,
                    "has_fg": int(fg_voxels > 0)
                })

out_csv = ROOT / "2_dataset" / "finetune_patch_index_v2.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)

res = pd.DataFrame(rows)

print("Patch size:", PATCH_SIZE)
print("Stride:", STRIDE)
print("Total patches:", len(res))
print("Foreground patches:", int(res["has_fg"].sum()))
print("Background patches:", int((res["has_fg"] == 0).sum()))
print("Saved:", out_csv)
print(res.groupby(["class","has_fg"]).size())
