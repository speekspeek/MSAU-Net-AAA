import numpy as np
import pandas as pd
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
CENTER_DIR = ROOT / "0_raw_data/centerlines/centerlines"
OUT_DIR = ROOT / "8_results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCHES = ["abdominal_aorta", "renal_left", "renal_right", "iliac_left", "iliac_right"]

def read_vtp_points(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    if pts is None:
        return None
    arr = vtk_to_numpy(pts.GetData())
    return arr.astype(float)

def length_tortuosity(pts):
    if pts is None or len(pts) < 2:
        return np.nan, np.nan, np.nan
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg.sum())
    straight = float(np.linalg.norm(pts[-1] - pts[0]))
    tort = float(length / straight) if straight > 0 else np.nan
    return length, straight, tort

def min_dist(a, b):
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.nan
    dmin = np.inf
    for i in range(0, len(a), max(1, len(a)//200)):
        d = np.linalg.norm(b - a[i], axis=1).min()
        dmin = min(dmin, d)
    return float(dmin)

def angle_between(a, b):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return np.nan
    va = a[-1] - a[0]
    vb = b[-1] - b[0]
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return np.nan
    cosang = np.clip(np.dot(va, vb) / (na * nb), -1, 1)
    return float(np.degrees(np.arccos(cosang)))

rows = []

for case_dir in sorted(CENTER_DIR.glob("AAA*")):
    row = {"case_id": case_dir.name}
    pts_dict = {}

    for branch in BRANCHES:
        f = case_dir / f"{branch}.vtp"
        row[f"{branch}_exists"] = int(f.exists())

        if not f.exists():
            continue

        pts = read_vtp_points(f)
        pts_dict[branch] = pts

        if pts is None:
            row[f"{branch}_points"] = 0
            continue

        length, straight, tort = length_tortuosity(pts)
        bbox = pts.max(axis=0) - pts.min(axis=0)

        row[f"{branch}_points"] = int(len(pts))
        row[f"{branch}_length"] = length
        row[f"{branch}_straight_length"] = straight
        row[f"{branch}_tortuosity"] = tort
        row[f"{branch}_bbox_x"] = float(bbox[0])
        row[f"{branch}_bbox_y"] = float(bbox[1])
        row[f"{branch}_bbox_z"] = float(bbox[2])

    row["renal_left_to_aorta_min_dist"] = min_dist(pts_dict.get("renal_left"), pts_dict.get("abdominal_aorta"))
    row["renal_right_to_aorta_min_dist"] = min_dist(pts_dict.get("renal_right"), pts_dict.get("abdominal_aorta"))
    row["iliac_left_to_aorta_min_dist"] = min_dist(pts_dict.get("iliac_left"), pts_dict.get("abdominal_aorta"))
    row["iliac_right_to_aorta_min_dist"] = min_dist(pts_dict.get("iliac_right"), pts_dict.get("abdominal_aorta"))
    row["renal_bifurcation_angle"] = angle_between(pts_dict.get("renal_left"), pts_dict.get("renal_right"))
    row["iliac_bifurcation_angle"] = angle_between(pts_dict.get("iliac_left"), pts_dict.get("iliac_right"))

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "morphology_centerline_features.csv", index=False)
df.describe().T.to_csv(OUT_DIR / "morphology_centerline_summary.csv")

print("Cases:", len(df))
print("Columns:", len(df.columns))
print(df.head())
print("Saved:", OUT_DIR / "morphology_centerline_features.csv")
