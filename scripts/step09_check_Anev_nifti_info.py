import pandas as pd
from pathlib import Path
import nibabel as nib

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
OUT = ROOT / "4_results" / "finetune"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "2_dataset" / "Anev_case_manifest.csv")

rows = []

for _, r in df.iterrows():
    img = nib.load(str(ROOT / r["image_path"]))
    lab = nib.load(str(ROOT / r["label_path"]))

    rows.append({
        "case_id": r["case_id"],
        "image_shape": str(img.shape),
        "label_shape": str(lab.shape),
        "image_spacing": str(img.header.get_zooms()[:3]),
        "label_spacing": str(lab.header.get_zooms()[:3]),
    })

out_csv = OUT / "step09_Anev_nifti_info.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Checked Anev cases:", len(rows))
print("Saved:", out_csv)
print(pd.DataFrame(rows))
