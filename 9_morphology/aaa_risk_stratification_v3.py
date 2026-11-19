from pathlib import Path
import numpy as np
import pandas as pd


ROOT=Path(
"/home/ubuntu/AAA_CTA_Project"
)


PRED_DIR=ROOT / \
"8_results/predictions/msau_unet3d_v2"


OUT=ROOT / \
"9_results/AAA_risk_stratification_v3.csv"



def compute_volume(mask):

    # resampled spacing = 1mm
    return float(mask.sum())



def compute_max_diameter(mask):

    """
    Axial maximum diameter
    based on maximum cross-sectional area
    """

    max_d=0


    # assume mask shape:
    # Z,Y,X

    for z in range(mask.shape[0]):

        slice_mask=mask[z]


        area=slice_mask.sum()


        if area==0:
            continue


        diameter=2*np.sqrt(
            area/np.pi
        )


        max_d=max(
            max_d,
            diameter
        )


    return max_d



def risk(dia):

    if dia < 40:
        return "Low"

    elif dia < 55:
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


    vol=compute_volume(
        mask
    )


    dia=compute_max_diameter(
        mask
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

