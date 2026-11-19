import pandas as pd
from pathlib import Path
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "8_results/tables"

df = pd.read_csv(T / "full_morphology_risk_features.csv")

features = [
    "max_bbox_diameter",
    "volume",
    "surface_area",
    "abdominal_aorta_length",
    "abdominal_aorta_tortuosity",
    "renal_bifurcation_angle",
    "iliac_bifurcation_angle",
    "morphology_risk_score"
]

rows = []
for f in features:
    groups = [g[f].dropna().values for _, g in df.groupby("risk_level")]
    stat, p = kruskal(*groups)
    rows.append({
        "feature": f,
        "kruskal_stat": stat,
        "p_value": p
    })

res = pd.DataFrame(rows)
res.to_csv(T / "risk_group_statistical_tests.csv", index=False)

print(res)
print("Saved:", T / "risk_group_statistical_tests.csv")
