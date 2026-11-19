import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
OUT = ROOT / "4_results/raw_label_alignment_visual"
OUT.mkdir(parents=True, exist_ok=True)

CASES = {
    "Anev_01": (
        ROOT / "0_raw_data/Anev/Anev_01/nifti/Anev_01_image.nii.gz",
        ROOT / "0_raw_data/Anev/Anev_01/segmentation/Anev_01_segmentation.nii.gz",
    ),
    "Nat_05": (
        ROOT / "0_raw_data/Nat/Nat_05/nifti/Nat_05_image.nii.gz",
        ROOT / "0_raw_data/Nat/Nat_05/segmentation/Nat_05_segmentation.nii.gz",
    ),
}

for case_id, (img_path, lab_path) in CASES.items():

    img = nib.load(str(img_path)).get_fdata().astype(np.float32)
    lab = nib.load(str(lab_path)).get_fdata()

    lab = lab > 0
    lab_flip = np.flip(lab, axis=2)

    # Find representative slices from each interpretation
    direct_counts = lab.sum(axis=(0, 1))
    flip_counts = lab_flip.sum(axis=(0, 1))

    z_direct = int(np.argmax(direct_counts))
    z_flip = int(np.argmax(flip_counts))

    print("\n" + "=" * 70)
    print(case_id)
    print("Image shape :", img.shape)
    print("Direct max-label slice :", z_direct)
    print("Flipped max-label slice:", z_flip)

    # ---------------------------------------------------------
    # DIRECT interpretation
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        img[:, :, z_direct].T,
        cmap="gray",
        origin="lower",
        vmin=-200,
        vmax=400
    )

    ax.contour(
        lab[:, :, z_direct].T.astype(float),
        levels=[0.5],
        linewidths=1.2,
        origin="lower"
    )

    ax.set_title(
        f"{case_id} | DIRECT | slice {z_direct}"
    )
    ax.axis("off")

    plt.tight_layout()

    out = OUT / f"{case_id}_DIRECT.png"

    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Z-FLIPPED interpretation
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        img[:, :, z_flip].T,
        cmap="gray",
        origin="lower",
        vmin=-200,
        vmax=400
    )

    ax.contour(
        lab_flip[:, :, z_flip].T.astype(float),
        levels=[0.5],
        linewidths=1.2,
        origin="lower"
    )

    ax.set_title(
        f"{case_id} | Z-FLIPPED | slice {z_flip}"
    )
    ax.axis("off")

    plt.tight_layout()

    out = OUT / f"{case_id}_ZFLIPPED.png"

    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

print("\nSaved to:")
print(OUT)
