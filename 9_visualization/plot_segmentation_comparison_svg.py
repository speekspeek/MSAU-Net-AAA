from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT=Path(
"/home/ubuntu/AAA_CTA_Project"
)

CASE="Anev_07"


IMAGE=(
ROOT/
"3_preprocessed/Anev/images"/
f"{CASE}_crop_img.npy"
)


GT=(
ROOT/
"3_preprocessed/Anev/labels"/
f"{CASE}_crop_lab.npy"
)



PRED={

"3D U-Net":
ROOT/"8_results/predictions/unet3d_fold/fold5"/f"{CASE}_pred.npy",

"Attention U-Net":
ROOT/"8_results/predictions/attention_unet3d_fold/fold5"/f"{CASE}_pred.npy",

"MSAU w/o Multiscale":
ROOT/"8_results/predictions/msau_no_multiscale_fold/fold5"/f"{CASE}_pred.npy",

"MSAU w/o MSA":
ROOT/"8_results/predictions/msau_msa_only_fold/fold5"/f"{CASE}_pred.npy",

"MSAU-Net":
ROOT/"8_results/predictions/msau_unet3d_v2_fold/fold5"/f"{CASE}_pred.npy"

}



OUT=ROOT/"9_results/figures"

OUT.mkdir(
    parents=True,
    exist_ok=True
)



# =====================
# load data
# =====================

img=np.load(IMAGE)

gt=np.load(GT)


preds={}

for k,v in PRED.items():

    preds[k]=np.load(v)



# =====================
# choose maximum lesion slice
# =====================

z=np.argmax(
    gt.sum(axis=(0,1))
)


print("Selected slice:",z)



img2d=img[:,:,z]
gt2d=gt[:,:,z]



# =====================
# figure style
# =====================

plt.rcParams["font.family"]="Arial"

plt.rcParams["svg.fonttype"]="none"


fig,axs=plt.subplots(
    2,
    4,
    figsize=(12,6),
    dpi=300
)


axs=axs.flatten()



# CTA

axs[0].imshow(
    img2d,
    cmap="gray"
)

axs[0].set_title(
    "(a) CTA"
)



# GT

axs[1].imshow(
    gt2d,
    cmap="gray"
)

axs[1].set_title(
    "(b) Ground Truth"
)



# predictions

for idx,(name,pred) in enumerate(preds.items()):


    ax=axs[idx+2]


    ax.imshow(
        img2d,
        cmap="gray"
    )


    ax.imshow(
        pred[:,:,z],
        cmap="Reds",
        alpha=0.45
    )


    labels=[
        "(c) 3D U-Net",
        "(d) Attention U-Net",
        "(e) MSAU w/o Multiscale",
        "(f) MSAU w/o MSA",
        "(g) MSAU-Net"
    ]


    ax.set_title(
        labels[idx]
    )



for ax in axs:

    ax.axis("off")



plt.tight_layout()



# =====================
# save
# =====================


png=OUT/"Fig_segmentation_comparison_Anev07.png"

svg=OUT/"Fig_segmentation_comparison_Anev07.svg"

pdf=OUT/"Fig_segmentation_comparison_Anev07.pdf"



plt.savefig(
    png,
    dpi=600,
    bbox_inches="tight"
)


plt.savefig(
    svg,
    format="svg",
    bbox_inches="tight"
)


plt.savefig(
    pdf,
    format="pdf",
    bbox_inches="tight"
)



print("Saved:")
print(png)
print(svg)
print(pdf)

