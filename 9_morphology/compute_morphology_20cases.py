from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import marching_cubes


ROOT=Path(
"/home/ubuntu/AAA_CTA_Project"
)


PRED_DIR=ROOT / \
"8_results/predictions/msau_no_multiscale_fold"


OUT=ROOT / \
"9_results/AAA_morphological_features_20cases.csv"



def surface_area(mask):

    if mask.sum()==0:
        return 0


    verts, faces, _, _ = marching_cubes(
        mask.astype(np.float32),
        level=0.5
    )


    area=0


    for tri in faces:

        a=verts[tri[0]]
        b=verts[tri[1]]
        c=verts[tri[2]]


        area += (
            np.linalg.norm(
                np.cross(
                    b-a,
                    c-a
                )
            )/2
        )


    return area



def max_diameter(mask):

    max_d=0


    for z in range(mask.shape[0]):

        area=mask[z].sum()


        if area==0:
            continue


        d=2*np.sqrt(
            area/np.pi
        )


        max_d=max(
            max_d,
            d
        )


    return max_d



def sphericity(volume, area):

    if area==0:
        return 0


    return (
        (np.pi**(1/3))
        *
        (6*volume)**(2/3)
        /
        area
    )



results=[]



for f in sorted(
    PRED_DIR.rglob("*_pred.npy")
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


    vol=float(
        mask.sum()
    )


    dia=max_diameter(
        mask
    )


    sa=surface_area(
        mask
    )


    sph=sphericity(
        vol,
        sa
    )


    ratio=vol/(dia+1e-6)


    results.append(
        {
        "case_id":case,
        "volume_mm3":vol,
        "diameter_mm":dia,
        "surface_area_mm2":sa,
        "sphericity":sph,
        "volume_diameter_ratio":ratio
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

