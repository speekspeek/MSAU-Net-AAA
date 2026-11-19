import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
df = pd.read_csv(ROOT / "2_dataset" / "finetune_preprocessed_manifest.csv")

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["class"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["class"]
)

out = ROOT / "2_dataset"
train_df.to_csv(out / "finetune_train.csv", index=False)
val_df.to_csv(out / "finetune_val.csv", index=False)
test_df.to_csv(out / "finetune_test.csv", index=False)

print("Train:", len(train_df), train_df["class"].value_counts().to_dict())
print("Val:", len(val_df), val_df["class"].value_counts().to_dict())
print("Test:", len(test_df), test_df["class"].value_counts().to_dict())
print("Saved split csv files.")
