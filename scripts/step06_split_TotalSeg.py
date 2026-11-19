import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

csv_path = ROOT / "2_dataset" / "TotalSeg_aorta_manifest.csv"

df = pd.read_csv(csv_path)

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

out_dir = ROOT / "2_dataset"

train_csv = out_dir / "TotalSeg_train.csv"
val_csv = out_dir / "TotalSeg_val.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print("Total:", len(df))
print("Train:", len(train_df))
print("Val:", len(val_df))

print("\nSaved:")
print(train_csv)
print(val_csv)
