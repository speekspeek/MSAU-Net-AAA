import os
import csv
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
RAW = ROOT / "0_raw_data"
OUT_DIR = ROOT / "2_dataset"
LOG_DIR = ROOT / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

exts = [".nii", ".nii.gz", ".dcm", ".mha", ".nrrd", ".stl", ".vtp"]

def get_ext(p: Path):
    name = p.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return p.suffix.lower()

rows = []
for p in RAW.rglob("*"):
    if p.is_file():
        ext = get_ext(p)
        if ext in exts:
            rel = p.relative_to(ROOT)
            dataset = "unknown"
            parts = p.parts

            if "Anev" in parts:
                dataset = "Anev"
            elif "Nat" in parts:
                dataset = "Nat"
            elif "TotalSegmentator" in parts or "Totalsegmentator_dataset" in parts:
                dataset = "TotalSegmentator"
            elif "centerlines" in parts:
                dataset = "centerlines"
            elif "meshes" in parts:
                dataset = "meshes"

            case_id = ""
            for part in parts:
                if part.startswith("AAA") or part.startswith("Nat_") or part.startswith("s"):
                    case_id = part
                    break

            rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "file_type": ext,
                "file_name": p.name,
                "relative_path": str(rel)
            })

out_csv = OUT_DIR / "dataset_file_inventory.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["dataset", "case_id", "file_type", "file_name", "relative_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for r in rows:
    key = (r["dataset"], r["file_type"])
    summary[key] = summary.get(key, 0) + 1

summary_csv = OUT_DIR / "dataset_file_summary.csv"
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "file_type", "count"])
    for (dataset, file_type), count in sorted(summary.items()):
        writer.writerow([dataset, file_type, count])

print("Saved:", out_csv)
print("Saved:", summary_csv)
print("\n===== Summary =====")
for (dataset, file_type), count in sorted(summary.items()):
    print(f"{dataset:20s} {file_type:8s} {count}")
