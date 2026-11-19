import csv
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
TS = ROOT / "0_raw_data" / "TotalSegmentator" / "Totalsegmentator_dataset"
OUT = ROOT / "2_dataset"
LOG = ROOT / "logs"

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

rows = []

for case_dir in sorted(TS.glob("s*")):
    if not case_dir.is_dir():
        continue

    ct = case_dir / "ct.nii.gz"
    aorta = case_dir / "segmentations" / "aorta.nii.gz"

    if ct.exists() and aorta.exists():
        rows.append({
            "case_id": case_dir.name,
            "dataset": "TotalSegmentator",
            "image_path": str(ct.relative_to(ROOT)),
            "label_path": str(aorta.relative_to(ROOT))
        })

out_csv = OUT / "TotalSeg_aorta_manifest.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case_id", "dataset", "image_path", "label_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Valid TotalSegmentator aorta cases:", len(rows))
print("Saved:", out_csv)
for r in rows[:20]:
    print(r["case_id"], r["image_path"], r["label_path"])
