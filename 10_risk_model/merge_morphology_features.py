import pandas as pd


A = "9_results/AAA_risk_stratification_20cases.csv"

B = "9_results/AAA_morphological_features.csv"


OUT = "9_results/AAA_morphology_features_20cases.csv"



df1 = pd.read_csv(A)

df2 = pd.read_csv(B)



# rename

df2 = df2.rename(
    columns={
        "AAA_volume_mm3":"volume_mm3",
        "max_diameter_mm":"diameter_mm"
    }
)



# 保留完整形态学指标

df2 = df2[
[
"case_id",
"volume_mm3",
"diameter_mm",
"surface_area_mm2",
"sphericity",
"volume_diameter_ratio"
]
]



# 合并

df = df1.drop(
    columns=[
        "volume_mm3",
        "diameter_mm"
    ]
).merge(
    df2,
    on="case_id",
    how="left"
)



df.to_csv(
    OUT,
    index=False
)



print("Saved:")
print(OUT)


print(df.head())

print(df.columns)

