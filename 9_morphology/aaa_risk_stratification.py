import os
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project"
)


# MSAU最终预测
PRED_DIR = (
    ROOT /
    "8_results" /
    "predictions" /
    "msau_unet3d_v2"
)


OUT_DIR = (
    ROOT /
    "9_results"
)

OUT_DIR.mkdir(
    exist_ok=True,
    parents=True
)



# =========================
# spacing
# =========================

# 这里先读取manifest
# 后续替换成真实spacing

SPACING = np.array(
    [
        1.0,
        1.0,
        1.0
    ]
)



# =========================
# morphology
# =========================

def volume(mask):

    voxel_volume = np.prod(SPACING)

    return (
        mask.sum()
        *
        voxel_volume
    )



def diameter(mask):

    pts=np.argwhere(mask)


    if len(pts)<2:
        return 0


    pts=pts*SPACING


    max_d=0


    for i in range(len(pts)):

        d=np.sqrt(
            ((pts[i]-pts)**2)
            .sum(axis=1)
        )

        max_d=max(
            max_d,
            d.max()
        )


    return max_d



def risk_level(d):

    """
    AAA diameter risk stratification

    <40 mm:
        low

    40-55 mm:
        moderate

    >55 mm:
        high
    """

    if d < 40:

        return "Low"

    elif d < 55:

        return "Moderate"

    else:

        return "High"





# =========================
# main
# =========================


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


    mask=np.load(f)

    mask=mask>0



    vol=volume(mask)


    dia=diameter(mask)



    risk=risk_level(
        dia
    )


    results.append(

        {

        "case_id":case,

        "volume_mm3":
            vol,

        "diameter_mm":
            dia,

        "risk_level":
            risk

        }

    )



df=pd.DataFrame(results)



save=(
    OUT_DIR /
    "AAA_risk_stratification.csv"
)



df.to_csv(
    save,
    index=False
)



print("\nSaved:")
print(save)


print("\nRisk distribution:")

print(
    df["risk_level"]
    .value_counts()
)


print("\nSummary:")

print(
    df.groupby(
        "risk_level"
    )
    [
        [
        "volume_mm3",
        "diameter_mm"
        ]
    ]
    .mean()
)

