import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(
"/home/ubuntu/AAA_CTA_Project"
)


PRED_DIR=ROOT / \
"8_results/predictions/msau_unet3d_v2"


MANIFEST=ROOT / \
"2_dataset/finetune_preprocessed_manifest.csv"


OUT=ROOT / \
"9_results/AAA_risk_stratification_v2.csv"



df_manifest=pd.read_csv(
    MANIFEST
)



def get_spacing(case):

    row=df_manifest[
        df_manifest.case_id==case
    ].iloc[0]


    # resampled后默认spacing=1mm
    return np.array(
        [1.0,1.0,1.0]
    )



def volume(mask,spacing):

    return (
        mask.sum()
        *
        np.prod(spacing)
    )



def diameter(mask,spacing):

    pts=np.argwhere(mask)


    if len(pts)<2:
        return 0


    # numpy轴:
    # z,y,x
    # spacing也对应 z,y,x

    pts=pts*spacing


    maxd=0


    for i in range(len(pts)):

        d=np.sqrt(
            ((pts[i]-pts)**2)
            .sum(axis=1)
        )

        maxd=max(
            maxd,
            d.max()
        )


    return maxd



def risk(d):

    if d < 40:
        return "Low"

    elif d <55:
        return "Moderate"

    else:
        return "High"



results=[]


for f in sorted(
    PRED_DIR.glob("*_pred.npy")
):

    case=f.stem.replace(
        "_pred",
        ""
    )


    print(
        "Processing",
        case
    )


    mask=np.load(f)>0


    spacing=get_spacing(case)


    dia=diameter(
        mask,
        spacing
    )


    vol=volume(
        mask,
        spacing
    )


    results.append(
        {
        "case_id":case,
        "volume_mm3":vol,
        "diameter_mm":dia,
        "risk_level":risk(dia)
        }
    )



df=pd.DataFrame(results)


df.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)

print(df)

print(
df.risk_level.value_counts()
)

