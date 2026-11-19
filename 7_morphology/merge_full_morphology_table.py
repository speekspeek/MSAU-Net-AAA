import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "8_results/tables"

mesh = pd.read_csv(T / "morphology_risk_table.csv")
center = pd.read_csv(T / "morphology_centerline_features.csv")

df = mesh.merge(center, on="case_id", how="left")

df.to_csv(T / "full_morphology_risk_features.csv", index=False)

summary = df.describe().T
summary.to_csv(T / "full_morphology_risk_features_summary.csv")

print("Full table:", df.shape)
print(df[[
    "case_id",
    "max_bbox_diameter",
    "volume",
    "surface_area",
    "abdominal_aorta_length",
    "abdominal_aorta_tortuosity",
    "renal_bifurcation_angle",
    "iliac_bifurcation_angle",
    "risk_level",
    "morphology_risk_score"
]].head())

print("Saved:")
print(T / "full_morphology_risk_features.csv")
print(T / "full_morphology_risk_features_summary.csv")
