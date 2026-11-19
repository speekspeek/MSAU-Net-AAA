import csv
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
ANEV = ROOT / "0_raw_data" / "Anev"
OUT = ROOT / "2_dataset"
LOG = ROOT / "logs"

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

rows = []

for case_dir in sorted(ANEV.glob("Anev_*")):
    if not case_dir.is_dir():
        continue

    case_id = case_dir.name
    imgs = list(case_dir.glob("nifti/*_image.nii.gz"))
    labs = list(case_dir.glob("segmentation/*_segmentation.nii.gz"))

    if len(imgs) == 1 and len(labs) == 1:
        rows.append({
            "case_id": case_id,
            "dataset": "Anev",
            "image_path": str(imgs[0].relative_to(ROOT)),
            "label_path": str(labs[0].relative_to(ROOT))
        })

out_csv = OUT / "Anev_case_manifest.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case_id", "dataset", "image_path", "label_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Valid Anev paired cases:", len(rows))
print("Saved:", out_csv)
for r in rows:
    print(r["case_id"], r["image_path"], r["label_path"])
