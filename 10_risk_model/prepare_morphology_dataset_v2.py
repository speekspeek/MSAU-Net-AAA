import pandas as pd


FEATURE="9_results/AAA_morphological_features_20cases.csv"

LABEL="9_results/AAA_risk_stratification_20cases.csv"


OUT="9_results/AAA_morphology_dataset_v2.csv"



feat=pd.read_csv(FEATURE)

label=pd.read_csv(LABEL)



label=label[
[
"case_id",
"class"
]
]



df=feat.merge(
    label,
    on="case_id",
    how="left"
)



df.to_csv(
    OUT,
    index=False
)



print("Saved:")
print(OUT)


print(df)


print(df.groupby("class").size())

