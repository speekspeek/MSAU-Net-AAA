import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTER_DIR = ROOT / "0_raw_data/centerlines/centerlines"
OUT_DIR = ROOT / "8_results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCHES = [
    "abdominal_aorta",
    "renal_left",
    "renal_right",
    "iliac_left",
    "iliac_right"
]

def read_vtp_points(path):
    tree = ET.parse(path)
    root = tree.getroot()

    candidates = []

    for da in root.iter():
        if not da.tag.endswith("DataArray"):
            continue

        text = da.text
        if text is None:
            continue

        vals = text.strip().split()
        if len(vals) < 6:
            continue

        try:
            nums = np.array([float(x) for x in vals], dtype=float)
        except Exception:
            continue

        if nums.size % 3 == 0:
            pts = nums.reshape(-1, 3)
            if pts.shape[0] >= 2:
                candidates.append(pts)

    if not candidates:
        return None

    return max(candidates, key=lambda x: x.shape[0])

def length_and_tortuosity(pts):
    diffs = np.diff(pts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)

    length = float(seg.sum())
    straight = float(np.linalg.norm(pts[-1] - pts[0]))
    tort = float(length / straight) if straight > 0 else np.nan

    return length, straight, tort

def min_distance_between_points(a, b):
    if a is None or b is None:
        return np.nan
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min())

def branch_angle(a_pts, b_pts):
    if a_pts is None or b_pts is None:
        return np.nan

    if len(a_pts) < 2 or len(b_pts) < 2:
        return np.nan

    va = a_pts[-1] - a_pts[0]
    vb = b_pts[-1] - b_pts[0]

    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)

    if na == 0 or nb == 0:
        return np.nan

    cosang = np.dot(va, vb) / (na * nb)
    cosang = np.clip(cosang, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))

rows = []

for case_dir in sorted(CENTER_DIR.glob("AAA*")):
    if not case_dir.is_dir():
        continue

    case_id = case_dir.name
    row = {"case_id": case_id}
    pts_dict = {}

    for branch in BRANCHES:
        vtp = case_dir / f"{branch}.vtp"

        if not vtp.exists():
            row[f"{branch}_exists"] = 0
            continue

        pts = read_vtp_points(vtp)
        pts_dict[branch] = pts
        row[f"{branch}_exists"] = 1

        if pts is None:
            row[f"{branch}_points"] = 0
            continue

        length, straight, tort = length_and_tortuosity(pts)

        row[f"{branch}_points"] = int(len(pts))
        row[f"{branch}_length"] = length
        row[f"{branch}_straight_length"] = straight
        row[f"{branch}_tortuosity"] = tort

        bbox = pts.max(axis=0) - pts.min(axis=0)
        row[f"{branch}_bbox_x"] = float(bbox[0])
        row[f"{branch}_bbox_y"] = float(bbox[1])
        row[f"{branch}_bbox_z"] = float(bbox[2])

    # Renal branch distance to abdominal aorta
    row["renal_left_to_aorta_min_dist"] = min_distance_between_points(
        pts_dict.get("renal_left"), pts_dict.get("abdominal_aorta")
    )
    row["renal_right_to_aorta_min_dist"] = min_distance_between_points(
        pts_dict.get("renal_right"), pts_dict.get("abdominal_aorta")
    )

    # Iliac branch distance to abdominal aorta
    row["iliac_left_to_aorta_min_dist"] = min_distance_between_points(
        pts_dict.get("iliac_left"), pts_dict.get("abdominal_aorta")
    )
    row["iliac_right_to_aorta_min_dist"] = min_distance_between_points(
        pts_dict.get("iliac_right"), pts_dict.get("abdominal_aorta")
    )

    # Branch angles
    row["renal_bifurcation_angle"] = branch_angle(
        pts_dict.get("renal_left"), pts_dict.get("renal_right")
    )
    row["iliac_bifurcation_angle"] = branch_angle(
        pts_dict.get("iliac_left"), pts_dict.get("iliac_right")
    )

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "morphology_centerline_features.csv", index=False)

numeric_cols = df.select_dtypes(include=[np.number]).columns
summary = df[numeric_cols].describe().T
summary.to_csv(OUT_DIR / "morphology_centerline_summary.csv")

print("Cases:", len(df))
print("Columns:", len(df.columns))
print(df.head())
print()
print("Saved:")
print(OUT_DIR / "morphology_centerline_features.csv")
print(OUT_DIR / "morphology_centerline_summary.csv")
