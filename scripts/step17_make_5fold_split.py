import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
df = pd.read_csv(ROOT / "2_dataset" / "finetune_preprocessed_manifest.csv")

out_dir = ROOT / "2_dataset" / "folds"
out_dir.mkdir(parents=True, exist_ok=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["class"]), start=1):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df.to_csv(out_dir / f"fold{fold}_train.csv", index=False)
    test_df.to_csv(out_dir / f"fold{fold}_test.csv", index=False)

    print(f"Fold {fold}")
    print(" Train:", len(train_df), train_df["class"].value_counts().to_dict())
    print(" Test :", len(test_df), test_df["class"].value_counts().to_dict())

print("Saved 5-fold split files:", out_dir)
