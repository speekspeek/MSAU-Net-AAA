import ast
import pandas as pd
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
df = pd.read_csv(ROOT / "2_dataset" / "finetune_preprocessed_manifest.csv")

PATCH_SIZE = (96, 96, 96)
STRIDE = (64, 64, 64)

rows = []

for _, r in df.iterrows():
    shape = ast.literal_eval(r["crop_shape"])
    z, y, x = shape

    for z0 in range(0, max(z - PATCH_SIZE[0] + 1, 1), STRIDE[0]):
        for y0 in range(0, max(y - PATCH_SIZE[1] + 1, 1), STRIDE[1]):
            for x0 in range(0, max(x - PATCH_SIZE[2] + 1, 1), STRIDE[2]):
                rows.append({
                    "case_id": r["case_id"],
                    "dataset": r["dataset"],
                    "class": r["class"],
                    "image_npy": r["image_npy"],
                    "label_npy": r["label_npy"],
                    "z0": z0,
                    "y0": y0,
                    "x0": x0,
                    "dz": PATCH_SIZE[0],
                    "dy": PATCH_SIZE[1],
                    "dx": PATCH_SIZE[2],
                })

out_csv = ROOT / "2_dataset" / "finetune_patch_index.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Patch size:", PATCH_SIZE)
print("Stride:", STRIDE)
print("Total patches:", len(rows))
print("Saved:", out_csv)
print(pd.DataFrame(rows).head())
