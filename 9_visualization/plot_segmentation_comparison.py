from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT=Path(
"/home/ubuntu/AAA_CTA_Project"
)


CASE="Anev_07"



# =========================
# paths
# =========================


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
ROOT/
"8_results/predictions/unet3d_fold/fold5"/
f"{CASE}_pred.npy",


"Attention U-Net":
ROOT/
"8_results/predictions/attention_unet3d_fold/fold5"/
f"{CASE}_pred.npy",


"MSAU w/o Multiscale":
ROOT/
"8_results/predictions/msau_no_multiscale_fold/fold5"/
f"{CASE}_pred.npy",


"MSAU w/o MSA":
ROOT/
"8_results/predictions/msau_msa_only_fold/fold5"/
f"{CASE}_pred.npy",


"MSAU-Net":
ROOT/
"8_results/predictions/msau_unet3d_v2_fold/fold5"/
f"{CASE}_pred.npy"

}



OUT=ROOT/"9_results/figures"

OUT.mkdir(
parents=True,
exist_ok=True
)



# =========================
# load
# =========================


img=np.load(
    IMAGE
)


gt=np.load(
    GT
)



preds={}

for k,v in PRED.items():

    preds[k]=np.load(v)



# =========================
# choose slice
# =========================


area=gt.sum(axis=(0,1))

z=np.argmax(area)


print(
"Best slice:",
z
)



img2d=img[:,:,z]
gt2d=gt[:,:,z]


# =========================
# plot
# =========================


fig,axs=plt.subplots(
    2,
    4,
    figsize=(12,6)
)


axs=axs.flatten()



axs[0].imshow(
    img2d,
    cmap="gray"
)

axs[0].set_title(
    "CTA"
)



axs[1].imshow(
    gt2d,
    cmap="gray"
)

axs[1].set_title(
    "Ground Truth"
)



titles=list(preds.keys())


for i,(k,p) in enumerate(preds.items()):

    axs[i+2].imshow(
        img2d,
        cmap="gray"
    )


    mask=p[:,:,z]


    axs[i+2].imshow(
        mask,
        alpha=0.45,
        cmap="Reds"
    )


    axs[i+2].set_title(
        k
    )



for ax in axs:

    ax.axis("off")



plt.tight_layout()


save=(
OUT/
"Fig_segmentation_comparison_Anev07.png"
)


plt.savefig(
    save,
    dpi=300,
    bbox_inches="tight"
)


print(
"Saved:",
save
)


