import csv
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
NAT = ROOT / "0_raw_data" / "Nat"
OUT = ROOT / "2_dataset"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for p in sorted(NAT.rglob("*.nii.gz")):
    if "__MACOSX" in str(p) or p.name.startswith("._"):
        continue

    case_id = ""
    for part in p.parts:
        if part.startswith("Nat_"):
            case_id = part
            break

    rows.append({
        "case_id": case_id if case_id else p.parent.name,
        "dataset": "Nat",
        "file_name": p.name,
        "path": str(p.relative_to(ROOT))
    })

out_csv = OUT / "Nat_manifest.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case_id", "dataset", "file_name", "path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Nat nifti files:", len(rows))
print("Saved:", out_csv)
for r in rows[:50]:
    print(r["case_id"], r["file_name"], r["path"])
