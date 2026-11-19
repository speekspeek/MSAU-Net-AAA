import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "8_results/tables"

rows = []

for model in ["unet3d", "attention_unet3d"]:
    m = pd.read_csv(OUT / f"{model}_test_metrics_summary.csv").iloc[0]
    h = pd.read_csv(OUT / f"{model}_hd95_assd_summary.csv").iloc[0]
    rows.append({
        "Model": model,
        "Dice": m["Dice"],
        "IoU": m["IoU"],
        "Precision": m["Precision"],
        "Recall": m["Recall"],
        "HD95_voxel": h["HD95_voxel"],
        "ASSD_voxel": h["ASSD_voxel"],
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "model_comparison_summary.csv", index=False)
print(df)
print("Saved:", OUT / "model_comparison_summary.csv")
