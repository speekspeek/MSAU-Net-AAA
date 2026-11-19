import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "8_results/tables/morphology_features_all.csv"
OUT = ROOT / "8_results/tables"

df = pd.read_csv(IN)

def risk_level(d):
    if d < 220:
        return "Low"
    elif d < 240:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df["max_bbox_diameter"].apply(risk_level)

# 归一化风险评分
for col in ["max_bbox_diameter", "volume", "surface_area"]:
    df[col + "_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df["morphology_risk_score"] = (
    0.4 * df["max_bbox_diameter_norm"] +
    0.4 * df["volume_norm"] +
    0.2 * df["surface_area_norm"]
)

df.to_csv(OUT / "morphology_risk_table.csv", index=False)

summary = df.groupby("risk_level")[[
    "max_bbox_diameter", "volume", "surface_area", "morphology_risk_score"
]].agg(["mean", "std", "count"])

summary.to_csv(OUT / "morphology_risk_summary.csv")

print(df[["case_id","max_bbox_diameter","volume","surface_area","risk_level","morphology_risk_score"]].head(20))
print()
print(summary)
print()
print("Saved:")
print(OUT / "morphology_risk_table.csv")
print(OUT / "morphology_risk_summary.csv")
