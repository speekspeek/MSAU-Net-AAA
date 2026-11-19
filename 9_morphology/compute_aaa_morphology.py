import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import ndimage
from skimage.measure import marching_cubes


ROOT = Path("/home/ubuntu/AAA_CTA_Project")


# 使用最终MSAU预测
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
    parents=True,
    exist_ok=True
)



# -------------------------
# voxel spacing
# -------------------------

# 如果没有保存spacing，默认
# 后续可替换为真实CTA spacing

SPACING = (
    1.0,
    1.0,
    1.0
)



# -------------------------
# volume
# -------------------------

def compute_volume(mask, spacing):

    voxel_volume = np.prod(spacing)

    return (
        mask.sum()
        *
        voxel_volume
    )



# -------------------------
# surface area
# -------------------------

def compute_surface_area(mask, spacing):

    if mask.sum()==0:
        return 0


    verts, faces, _, _ = marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        spacing=spacing
    )


    area=0


    for tri in faces:

        p1=verts[tri[0]]
        p2=verts[tri[1]]
        p3=verts[tri[2]]

        area += (
            np.linalg.norm(
                np.cross(
                    p2-p1,
                    p3-p1
                )
            )
            /
            2
        )


    return area



# -------------------------
# maximum diameter
# -------------------------

def compute_max_diameter(mask, spacing):

    points=np.argwhere(mask)


    if len(points)<2:
        return 0


    points = (
        points
        *
        np.array(spacing)
    )


    # 分块计算避免内存爆炸

    max_dist=0


    for i in range(len(points)):

        d=np.sqrt(
            (
                (
                    points[i]
                    -
                    points
                )
                **2
            )
            .sum(axis=1)
        )


        max_dist=max(
            max_dist,
            d.max()
        )


    return max_dist



# -------------------------
# sphericity
# -------------------------

def compute_sphericity(volume, area):

    if area==0:
        return 0


    return (
        np.pi**(1/3)
        *
        (6*volume)**(2/3)
        /
        area
    )



# -------------------------
# main
# -------------------------


results=[]


for file in sorted(
    PRED_DIR.glob("*_pred.npy")
):


    case_id=file.stem.replace(
        "_pred",
        ""
    )


    print(
        "Processing",
        case_id
    )


    mask=np.load(
        file
    )


    mask=mask>0



    volume=compute_volume(
        mask,
        SPACING
    )


    diameter=compute_max_diameter(
        mask,
        SPACING
    )


    area=compute_surface_area(
        mask,
        SPACING
    )


    sph=compute_sphericity(
        volume,
        area
    )


    vd_ratio=(
        volume /
        (diameter+1e-8)
    )



    results.append(

        {

        "case_id":case_id,

        "AAA_volume_mm3":
            volume,

        "max_diameter_mm":
            diameter,

        "surface_area_mm2":
            area,

        "sphericity":
            sph,

        "volume_diameter_ratio":
            vd_ratio

        }

    )



df=pd.DataFrame(results)


save_path=(
    OUT_DIR /
    "AAA_morphological_features.csv"
)


df.to_csv(
    save_path,
    index=False
)



print("\nSaved:")
print(save_path)

print(df.describe())

