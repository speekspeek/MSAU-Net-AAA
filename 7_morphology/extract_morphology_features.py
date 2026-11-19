import os
import re
import struct
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CENTER_DIR = ROOT / "0_raw_data/centerlines/centerlines"
MESH_DIR = ROOT / "0_raw_data/meshes/meshes"

OUT_DIR = ROOT / "8_results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_vtp_points(path):
    tree = ET.parse(path)
    root = tree.getroot()

    points = []
    for da in root.iter():
        if da.tag.endswith("DataArray"):
            text = da.text
            if text is None:
                continue

            vals = text.strip().split()
            if len(vals) >= 3:
                try:
                    nums = np.array([float(x) for x in vals], dtype=float)
                    if nums.size % 3 == 0:
                        pts = nums.reshape(-1, 3)
                        if pts.shape[0] > 1:
                            points.append(pts)
                except:
                    pass

    if len(points) == 0:
        return None

    return max(points, key=lambda x: x.shape[0])

def centerline_features(vtp_path):
    pts = read_vtp_points(vtp_path)
    if pts is None or len(pts) < 2:
        return None

    diffs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    length = float(seg_len.sum())

    straight = float(np.linalg.norm(pts[-1] - pts[0]))
    tortuosity = float(length / straight) if straight > 0 else np.nan

    bbox = pts.max(axis=0) - pts.min(axis=0)

    return {
        "centerline_points": len(pts),
        "centerline_length": length,
        "straight_length": straight,
        "tortuosity": tortuosity,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
    }

def read_stl_triangles(path):
    data = Path(path).read_bytes()

    # binary STL
    if len(data) > 84:
        n_tri = struct.unpack("<I", data[80:84])[0]
        expected = 84 + n_tri * 50
        if expected == len(data):
            tris = np.zeros((n_tri, 3, 3), dtype=np.float64)
            offset = 84
            for i in range(n_tri):
                offset += 12
                vals = struct.unpack("<9f", data[offset:offset+36])
                tris[i] = np.array(vals).reshape(3, 3)
                offset += 38
            return tris

    # ascii STL fallback
    text = data.decode(errors="ignore")
    verts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            if len(parts) == 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.array(verts, dtype=np.float64)
    if len(verts) % 3 != 0 or len(verts) == 0:
        return None
    return verts.reshape(-1, 3, 3)

def mesh_features(stl_path):
    tris = read_stl_triangles(stl_path)
    if tris is None:
        return None

    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]

    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1).sum()

    volume = np.abs(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)

    pts = tris.reshape(-1, 3)
    bbox = pts.max(axis=0) - pts.min(axis=0)

    return {
        "mesh_triangles": int(tris.shape[0]),
        "surface_area": float(area),
        "volume": float(volume),
        "mesh_bbox_x": float(bbox[0]),
        "mesh_bbox_y": float(bbox[1]),
        "mesh_bbox_z": float(bbox[2]),
        "max_bbox_diameter": float(max(bbox)),
    }

rows = []

for stl in sorted(MESH_DIR.glob("*.stl")):
    case_id = stl.stem

    row = {
        "case_id": case_id,
        "mesh_path": str(stl),
    }

    mf = mesh_features(stl)
    if mf:
        row.update(mf)

    center_case_dir = CENTER_DIR / case_id
    if center_case_dir.exists():
        for branch in ["abdominal_aorta", "renal_left", "renal_right", "iliac_left", "iliac_right"]:
            vtp = center_case_dir / f"{branch}.vtp"
            if vtp.exists():
                cf = centerline_features(vtp)
                if cf:
                    for k, v in cf.items():
                        row[f"{branch}_{k}"] = v

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "morphology_features_all.csv", index=False)

summary = df.describe(include="all")
summary.to_csv(OUT_DIR / "morphology_features_summary.csv")

print("Cases:", len(df))
print("Columns:", len(df.columns))
print(df.head())
print("\nSaved:")
print(OUT_DIR / "morphology_features_all.csv")
print(OUT_DIR / "morphology_features_summary.csv")
