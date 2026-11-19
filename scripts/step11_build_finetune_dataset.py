import pandas as pd
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
OUT = ROOT / "2_dataset"

anev = pd.read_csv(OUT / "Anev_case_manifest.csv")

anev = anev.rename(columns={
    "image_path": "image",
    "label_path": "label"
})

anev["class"] = "AAA"

nat_rows = []

nat_root = ROOT / "0_raw_data" / "Nat"

for case_dir in sorted(nat_root.glob("Nat_*")):

    img = list(case_dir.glob("nifti/*_image.nii.gz"))
    seg = list(case_dir.glob("segmentation/*_segmentation.nii.gz"))

    if len(img) == 1 and len(seg) == 1:
        nat_rows.append({
            "case_id": case_dir.name,
            "dataset": "Nat",
            "image": str(img[0].relative_to(ROOT)),
            "label": str(seg[0].relative_to(ROOT)),
            "class": "Normal"
        })

nat = pd.DataFrame(nat_rows)

final_df = pd.concat([
    anev[["case_id","dataset","image","label","class"]],
    nat[["case_id","dataset","image","label","class"]]
], ignore_index=True)

out_csv = OUT / "AAA_finetune_dataset.csv"

final_df.to_csv(out_csv, index=False)

print("AAA cases:", (final_df["class"]=="AAA").sum())
print("Normal cases:", (final_df["class"]=="Normal").sum())
print("Total:", len(final_df))
print("Saved:", out_csv)

print(final_df.head(20))
