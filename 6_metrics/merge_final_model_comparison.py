import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "8_results/tables"

models = [
    ("U-Net", "unet3d"),
    ("Attention U-Net", "attention_unet3d"),
    ("MSAU-Net V1", "msau_unet3d"),
    ("MSAU-Net V2 (Ours)", "msau_unet3d_v2"),
]

rows = []
for name, key in models:
    m = pd.read_csv(T / f"{key}_test_metrics_summary.csv").iloc[0]
    h = pd.read_csv(T / f"{key}_hd95_assd_summary.csv").iloc[0]
    rows.append({
        "Model": name,
        "Dice": m["Dice"],
        "IoU": m["IoU"],
        "Precision": m["Precision"],
        "Recall": m["Recall"],
        "HD95_voxel": h["HD95_voxel"],
        "ASSD_voxel": h["ASSD_voxel"],
    })

df = pd.DataFrame(rows)
df.to_csv(T / "final_model_comparison_summary.csv", index=False)

print(df)
print("Saved:", T / "final_model_comparison_summary.csv")
