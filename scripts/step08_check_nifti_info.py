import pandas as pd
from pathlib import Path
import nibabel as nib

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
OUT = ROOT / "4_results" / "pretrain"
OUT.mkdir(parents=True, exist_ok=True)

csv_path = ROOT / "2_dataset" / "TotalSeg_train.csv"
df = pd.read_csv(csv_path).head(20)

rows = []

for _, r in df.iterrows():
    img_path = ROOT / r["image_path"]
    lab_path = ROOT / r["label_path"]

    img = nib.load(str(img_path))
    lab = nib.load(str(lab_path))

    rows.append({
        "case_id": r["case_id"],
        "image_shape": str(img.shape),
        "label_shape": str(lab.shape),
        "image_spacing": str(img.header.get_zooms()[:3]),
        "label_spacing": str(lab.header.get_zooms()[:3]),
    })

out_csv = OUT / "step08_nifti_info_sample.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Checked cases:", len(rows))
print("Saved:", out_csv)
print(pd.DataFrame(rows).head())
