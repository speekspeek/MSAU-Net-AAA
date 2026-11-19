#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AAA CTA - Training V4
Strict deduplicated 15-case patient-level 5-fold training.

V2:
1. Outer test fold is NEVER used for training/model selection.
2. Outer development cases are split patient-wise into train/validation.
3. Mixed foreground/context patch sampling.
4. Online augmentation.
5. AdamW + weight decay.
6. ReduceLROnPlateau.
7. Best checkpoint selected ONLY by validation Dice.
8. Early stopping.
"""

import argparse
import random
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "3_models")
)
from unet3d_gn import UNet3D_GN

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

PATCH_CSV = ROOT / "2_dataset/finetune_patch_index_v2.csv"
FOLD_ROOT = ROOT / "2_dataset/folds_dedup15_v2data"

CKPT_ROOT = ROOT / "5_checkpoints/folds_dedup15_gn_test"
RESULT_ROOT = ROOT / "4_results/folds_dedup15_gn_test"

CKPT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Hyperparameters
# ============================================================

PATCH_SIZE = (96, 96, 96)

EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4

VAL_CASES = 2

EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 6
LR_FACTOR = 0.5
MIN_LR = 1e-6

NUM_WORKERS = 2

# Whole-volume validation
SW_PATCH_SIZE = (96, 96, 96)
SW_STRIDE = (48, 48, 48)


# ============================================================
# Models
# ============================================================

MODEL_CONFIG = {
    "unet3d_gn": (
        "unet3d_gn.py",
        "UNet3D_GN"
    ),
    "unet3d": (
        "unet3d.py",
        "UNet3D"
    ),
    "attention_unet3d": (
        "attention_unet3d.py",
        "AttentionUNet3D"
    ),
    "msau_no_multiscale": (
        "msau_unet3d_no_multiscale.py",
        "MSAUNet3D_NoMultiScale"
    ),
    "msau_no_msa": (
        "msau_unet3d_msa_only.py",
        "MSAUNet3DMSAOnly"
    ),
    "msau_unet3d_v2": (
        "msau_unet3d_v2.py",
        "MSAUNet3DV2"
    ),
}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Patient-level internal validation split
# ============================================================

def split_train_val_cases(dev_df, seed):

    dev_df = dev_df.copy()

    case_col = "case_id"

    # Try to retain AAA/Normal representation
    class_col = None

    for c in ["class", "group", "Group", "label"]:
        if c in dev_df.columns:
            class_col = c
            break

    rng = np.random.RandomState(seed)

    if class_col is not None:

        val_cases = []

        groups = dev_df.groupby(class_col)

        for _, g in groups:

            cases = g[case_col].astype(str).unique().tolist()

            rng.shuffle(cases)

            if len(cases) > 0:
                val_cases.append(cases[0])

        # Exactly two validation patients
        val_cases = list(dict.fromkeys(val_cases))

        if len(val_cases) > VAL_CASES:
            val_cases = val_cases[:VAL_CASES]

        if len(val_cases) < VAL_CASES:

            remaining = [
                x for x in
                dev_df[case_col].astype(str).unique()
                if x not in val_cases
            ]

            rng.shuffle(remaining)

            val_cases += remaining[
                :VAL_CASES-len(val_cases)
            ]

    else:

        cases = (
            dev_df[case_col]
            .astype(str)
            .unique()
            .tolist()
        )

        rng.shuffle(cases)

        val_cases = cases[:VAL_CASES]

    val_cases = set(val_cases)

    train_cases = set(
        dev_df[case_col].astype(str)
    ) - val_cases

    return (
        sorted(train_cases),
        sorted(val_cases)
    )



# ============================================================
# V4 Dynamic balanced patch sampling
# ============================================================

TRAIN_PATCHES_PER_CASE = 32
FG_PATCH_FRACTION = 0.70
CONTEXT_CANDIDATES = 32
MAX_SAMPLE_TRIES = 50


def _safe_start(center, patch, size):
    """
    Convert a desired patch center to a legal start coordinate.
    """
    if size <= patch:
        return 0

    start = int(center - patch // 2)

    return max(
        0,
        min(
            start,
            size - patch
        )
    )


def _random_start(size, patch):
    """
    Random legal patch start.
    """
    if size <= patch:
        return 0

    return random.randint(
        0,
        size - patch
    )


def _extract_fixed_patch(
    image,
    label,
    z0,
    y0,
    x0
):
    """
    Extract patch and pad boundary dimensions to PATCH_SIZE.
    """

    dz, dy, dx = PATCH_SIZE

    image_patch = image[
        z0:z0+dz,
        y0:y0+dy,
        x0:x0+dx
    ]

    label_patch = label[
        z0:z0+dz,
        y0:y0+dy,
        x0:x0+dx
    ]

    pad_z = max(
        0,
        dz - image_patch.shape[0]
    )

    pad_y = max(
        0,
        dy - image_patch.shape[1]
    )

    pad_x = max(
        0,
        dx - image_patch.shape[2]
    )

    if pad_z or pad_y or pad_x:

        image_patch = np.pad(
            image_patch,
            (
                (0, pad_z),
                (0, pad_y),
                (0, pad_x)
            ),
            mode="constant",
            constant_values=0
        )

        label_patch = np.pad(
            label_patch,
            (
                (0, pad_z),
                (0, pad_y),
                (0, pad_x)
            ),
            mode="constant",
            constant_values=0
        )

    image_patch = image_patch[
        :dz,
        :dy,
        :dx
    ]

    label_patch = label_patch[
        :dz,
        :dy,
        :dx
    ]

    return (
        image_patch,
        label_patch
    )


class DynamicBalancedPatchDataset(Dataset):
    """
    Patient-balanced dynamic patch sampler.

    Every case contributes exactly TRAIN_PATCHES_PER_CASE
    samples per epoch.

    Sampling:
        70% foreground-centered
        30% random context/background

    No global patch index is used.
    """

    def __init__(
        self,
        case_df,
        patches_per_case=TRAIN_PATCHES_PER_CASE,
        fg_fraction=FG_PATCH_FRACTION,
        augment=True
    ):

        self.case_df = (
            case_df
            .drop_duplicates(
                subset=["case_id"]
            )
            .reset_index(drop=True)
            .copy()
        )

        self.patches_per_case = int(
            patches_per_case
        )

        self.fg_fraction = float(
            fg_fraction
        )

        self.augment = augment

        self.cache = {}

        if len(self.case_df) == 0:
            raise RuntimeError(
                "Dynamic training dataset "
                "contains zero cases."
            )

    def __len__(self):

        return (
            len(self.case_df)
            *
            self.patches_per_case
        )

    def _load_case(self, case_index):

        row = self.case_df.iloc[
            case_index
        ]

        case_id = str(
            row["case_id"]
        )

        if case_id not in self.cache:

            image = np.load(
                ROOT /
                row["image_npy"]
            ).astype(
                np.float32
            )

            label = np.load(
                ROOT /
                row["label_npy"]
            )

            label = (
                label > 0
            ).astype(
                np.uint8
            )

            if image.shape != label.shape:
                raise RuntimeError(
                    f"{case_id}: image/label "
                    f"shape mismatch "
                    f"{image.shape} vs "
                    f"{label.shape}"
                )

            fg_coordinates = np.argwhere(
                label > 0
            )

            self.cache[
                case_id
            ] = (
                image,
                label,
                fg_coordinates
            )

        return (
            case_id,
            *self.cache[
                case_id
            ]
        )

    def _foreground_patch(
        self,
        image,
        label,
        fg_coordinates
    ):

        D, H, W = image.shape
        dz, dy, dx = PATCH_SIZE

        # Choose an actual foreground voxel
        index = random.randrange(
            len(fg_coordinates)
        )

        cz, cy, cx = (
            fg_coordinates[index]
        )

        # Jitter center so model does not always
        # see the vessel exactly at patch center.
        cz = int(
            cz +
            random.randint(
                -dz // 4,
                dz // 4
            )
        )

        cy = int(
            cy +
            random.randint(
                -dy // 4,
                dy // 4
            )
        )

        cx = int(
            cx +
            random.randint(
                -dx // 4,
                dx // 4
            )
        )

        z0 = _safe_start(
            cz,
            dz,
            D
        )

        y0 = _safe_start(
            cy,
            dy,
            H
        )

        x0 = _safe_start(
            cx,
            dx,
            W
        )

        return _extract_fixed_patch(
            image,
            label,
            z0,
            y0,
            x0
        )

    def _context_patch(
        self,
        image,
        label
    ):
        """
        Low-foreground context sampling.

        The cropped CTA volumes are tight ROIs, therefore a
        96^3 patch usually contains some vessel foreground.
        Instead of requiring a pure-background patch, randomly
        generate multiple candidate windows and select the one
        with the lowest foreground burden.
        """

        D, H, W = image.shape
        dz, dy, dx = PATCH_SIZE

        best_image = None
        best_label = None
        best_fg = None

        for _ in range(
            CONTEXT_CANDIDATES
        ):

            z0 = _random_start(
                D,
                dz
            )

            y0 = _random_start(
                H,
                dy
            )

            x0 = _random_start(
                W,
                dx
            )

            image_patch, label_patch = (
                _extract_fixed_patch(
                    image,
                    label,
                    z0,
                    y0,
                    x0
                )
            )

            fg = int(
                label_patch.sum()
            )

            if (
                best_fg is None
                or fg < best_fg
            ):
                best_fg = fg
                best_image = image_patch
                best_label = label_patch

        return (
            best_image,
            best_label
        )


    def __getitem__(
        self,
        index
    ):

        case_index = (
            index
            //
            self.patches_per_case
        )

        _, image, label, fg_coordinates = (
            self._load_case(
                case_index
            )
        )

        use_fg = (
            random.random()
            <
            self.fg_fraction
        )

        if (
            use_fg
            and
            len(fg_coordinates) > 0
        ):

            image_patch, label_patch = (
                self._foreground_patch(
                    image,
                    label,
                    fg_coordinates
                )
            )

        else:

            image_patch, label_patch = (
                self._context_patch(
                    image,
                    label
                )
            )

        label_patch = (
            label_patch > 0
        ).astype(
            np.int64
        )

        if self.augment:

            image_patch, label_patch = (
                augment_patch(
                    image_patch,
                    label_patch
                )
            )

        image_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    image_patch
                )
            )
            .float()
            .unsqueeze(0)
        )

        label_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    label_patch
                )
            )
            .long()
        )

        return (
            image_tensor,
            label_tensor
        )


# ============================================================
# Augmentation
# ============================================================

def augment_patch(image, label):

    # Random flips
    for axis in range(3):

        if random.random() < 0.5:

            image = np.flip(
                image,
                axis=axis
            ).copy()

            label = np.flip(
                label,
                axis=axis
            ).copy()

    # Random 90-degree rotation in axial plane
    if random.random() < 0.5:

        k = random.randint(0, 3)

        image = np.rot90(
            image,
            k=k,
            axes=(1, 2)
        ).copy()

        label = np.rot90(
            label,
            k=k,
            axes=(1, 2)
        ).copy()

    # Intensity scaling
    if random.random() < 0.5:

        scale = random.uniform(
            0.90,
            1.10
        )

        image = image * scale

    # Intensity shift
    if random.random() < 0.5:

        shift = random.uniform(
            -0.10,
            0.10
        )

        image = image + shift

    # Gaussian noise
    if random.random() < 0.30:

        sigma = random.uniform(
            0.01,
            0.05
        )

        noise = np.random.normal(
            0.0,
            sigma,
            size=image.shape
        ).astype(np.float32)

        image = image + noise

    return (
        image.astype(np.float32),
        label.astype(np.int64)
    )


# ============================================================
# Patch Dataset
# ============================================================

class PatchDataset(Dataset):

    def __init__(
        self,
        dataframe,
        augment=False
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        r = self.df.iloc[index]

        image_path = ROOT / r["image_npy"]
        label_path = ROOT / r["label_npy"]

        image = np.load(
            image_path
        ).astype(np.float32)

        label = np.load(
            label_path
        )

        z0 = int(r["z0"])
        y0 = int(r["y0"])
        x0 = int(r["x0"])

        dz = int(r["dz"])
        dy = int(r["dy"])
        dx = int(r["dx"])

        image = image[
            z0:z0+dz,
            y0:y0+dy,
            x0:x0+dx
        ]

        label = label[
            z0:z0+dz,
            y0:y0+dy,
            x0:x0+dx
        ]

        # ----------------------------------------------------
        # Pad boundary patches to fixed 96 x 96 x 96
        # ----------------------------------------------------

        pad_z = max(0, PATCH_SIZE[0] - image.shape[0])
        pad_y = max(0, PATCH_SIZE[1] - image.shape[1])
        pad_x = max(0, PATCH_SIZE[2] - image.shape[2])

        if pad_z or pad_y or pad_x:

            image = np.pad(
                image,
                (
                    (0, pad_z),
                    (0, pad_y),
                    (0, pad_x)
                ),
                mode="constant",
                constant_values=0
            )

            label = np.pad(
                label,
                (
                    (0, pad_z),
                    (0, pad_y),
                    (0, pad_x)
                ),
                mode="constant",
                constant_values=0
            )

        # Safety crop in case any dimension exceeds patch size
        image = image[
            :PATCH_SIZE[0],
            :PATCH_SIZE[1],
            :PATCH_SIZE[2]
        ]

        label = label[
            :PATCH_SIZE[0],
            :PATCH_SIZE[1],
            :PATCH_SIZE[2]
        ]

        if image.shape != PATCH_SIZE or label.shape != PATCH_SIZE:
            raise RuntimeError(
                f"Patch shape error after padding: "
                f"image={image.shape}, label={label.shape}"
            )

        label = (
            label > 0
        ).astype(np.int64)

        if self.augment:

            image, label = augment_patch(
                image,
                label
            )

        image = torch.from_numpy(
            np.ascontiguousarray(image)
        ).float().unsqueeze(0)

        label = torch.from_numpy(
            np.ascontiguousarray(label)
        ).long()

        return image, label


# ============================================================
# Dice
# ============================================================

def dice_score_from_logits(
    logits,
    target,
    eps=1e-6
):

    pred = torch.argmax(
        logits,
        dim=1
    )

    pred = pred == 1
    target = target == 1

    intersection = (
        pred & target
    ).sum().float()

    denominator = (
        pred.sum().float()
        +
        target.sum().float()
    )

    if denominator.item() == 0:
        return 1.0

    return float(
        (
            2.0 * intersection + eps
        )
        /
        (
            denominator + eps
        )
    )


# ============================================================
# Soft Dice loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, eps=1e-6):

        super().__init__()
        self.eps = eps

    def forward(self, logits, target):

        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        target_fg = (
            target == 1
        ).float()

        dims = (
            1,
            2,
            3
        )

        intersection = (
            probs * target_fg
        ).sum(dims)

        denominator = (
            probs.sum(dims)
            +
            target_fg.sum(dims)
        )

        dice = (
            2.0 * intersection
            +
            self.eps
        ) / (
            denominator
            +
            self.eps
        )

        return 1.0 - dice.mean()


# ============================================================
# Combined loss
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        # Mild foreground emphasis
        weights = torch.tensor(
            [1.0, 2.0],
            dtype=torch.float32
        )

        self.register_buffer(
            "weights",
            weights
        )

        self.dice = DiceLoss()

    def forward(self, logits, target):

        ce = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weights
        )

        dice = self.dice(
            logits,
            target
        )

        return (
            0.5 * ce
            +
            0.5 * dice
        )


# ============================================================
# Build model
# ============================================================

def build_model(model_name):

    model_file, class_name = (
        MODEL_CONFIG[
            model_name
        ]
    )

    module = SourceFileLoader(
        f"{model_name}_gn_test",
        str(
            ROOT /
            "3_models" /
            model_file
        )
    ).load_module()

    ModelClass = getattr(
        module,
        class_name
    )

    return ModelClass(
        in_channels=1,
        num_classes=2,
        base_ch=16
    )


# ============================================================
# Epoch
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    dices = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for image, label in loader:

            image = image.to(
                device,
                non_blocking=True
            )

            label = label.to(
                device,
                non_blocking=True
            )

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(
                image
            )

            loss = criterion(
                logits,
                label
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

            dice = dice_score_from_logits(
                logits,
                label
            )

            losses.append(
                float(loss.item())
            )

            dices.append(
                float(dice)
            )

    return (
        float(np.mean(losses)),
        float(np.mean(dices))
    )



# ============================================================
# Whole-volume sliding-window validation
# ============================================================

def make_starts(size, patch, stride):

    if size <= patch:
        return [0]

    starts = list(
        range(
            0,
            size - patch + 1,
            stride
        )
    )

    last = size - patch

    if starts[-1] != last:
        starts.append(last)

    return starts


def pad_volume_to_patch(
    image,
    patch_size=SW_PATCH_SIZE
):

    original_shape = image.shape

    pads = []

    for dim, p in zip(
        original_shape,
        patch_size
    ):

        total = max(
            0,
            p - dim
        )

        before = total // 2
        after = total - before

        pads.append(
            (before, after)
        )

    if any(
        a > 0 or b > 0
        for a, b in pads
    ):

        image = np.pad(
            image,
            pads,
            mode="constant",
            constant_values=0
        )

    return (
        image,
        pads,
        original_shape
    )


def crop_padding(
    arr,
    pads,
    original_shape
):

    slices = []

    for (
        before,
        after
    ), size in zip(
        pads,
        original_shape
    ):

        slices.append(
            slice(
                before,
                before + size
            )
        )

    return arr[
        tuple(slices)
    ]


@torch.no_grad()
def sliding_window_predict(
    model,
    image,
    device,
    patch_size=SW_PATCH_SIZE,
    stride=SW_STRIDE
):

    model.eval()

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image, pads, original_shape = (
        pad_volume_to_patch(
            image,
            patch_size
        )
    )

    D, H, W = image.shape

    pz, py, px = patch_size
    sz, sy, sx = stride

    z_starts = make_starts(
        D,
        pz,
        sz
    )

    y_starts = make_starts(
        H,
        py,
        sy
    )

    x_starts = make_starts(
        W,
        px,
        sx
    )

    probability_sum = np.zeros(
        (D, H, W),
        dtype=np.float32
    )

    count_map = np.zeros(
        (D, H, W),
        dtype=np.float32
    )

    for z0 in z_starts:

        for y0 in y_starts:

            for x0 in x_starts:

                patch = image[
                    z0:z0+pz,
                    y0:y0+py,
                    x0:x0+px
                ]

                patch_tensor = (
                    torch.from_numpy(
                        np.ascontiguousarray(
                            patch
                        )
                    )
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(
                        device,
                        non_blocking=True
                    )
                )

                logits = model(
                    patch_tensor
                )

                probability = (
                    torch.softmax(
                        logits,
                        dim=1
                    )[0, 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

                probability_sum[
                    z0:z0+pz,
                    y0:y0+py,
                    x0:x0+px
                ] += probability

                count_map[
                    z0:z0+pz,
                    y0:y0+py,
                    x0:x0+px
                ] += 1.0

    probability_map = (
        probability_sum
        /
        np.maximum(
            count_map,
            1e-6
        )
    )

    probability_map = crop_padding(
        probability_map,
        pads,
        original_shape
    )

    prediction = (
        probability_map >= 0.5
    )

    return prediction


def binary_dice(
    prediction,
    target,
    eps=1e-6
):

    prediction = (
        np.asarray(
            prediction
        ) > 0
    )

    target = (
        np.asarray(
            target
        ) > 0
    )

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    denominator = (
        prediction.sum()
        +
        target.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        (
            2.0 * intersection
            +
            eps
        )
        /
        (
            denominator
            +
            eps
        )
    )


@torch.no_grad()
def validate_whole_volume(
    model,
    val_case_df,
    device
):

    model.eval()

    case_dices = []

    for _, row in val_case_df.iterrows():

        case_id = str(
            row["case_id"]
        )

        image = np.load(
            ROOT /
            row["image_npy"]
        ).astype(
            np.float32
        )

        label = np.load(
            ROOT /
            row["label_npy"]
        )

        label = (
            label > 0
        )

        prediction = (
            sliding_window_predict(
                model,
                image,
                device
            )
        )

        dice = binary_dice(
            prediction,
            label
        )

        pred_voxels = int(
            prediction.sum()
        )

        gt_voxels = int(
            label.sum()
        )

        case_dices.append(
            dice
        )

        print(
            f"    VAL {case_id:10s} "
            f"Dice={dice:.6f} "
            f"Pred={pred_voxels} "
            f"GT={gt_voxels}"
        )

    mean_dice = float(
        np.mean(
            case_dices
        )
    )

    return (
        mean_dice,
        case_dices
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(
            MODEL_CONFIG.keys()
        )
    )

    parser.add_argument(
        "--fold",
        required=True,
        type=int,
        choices=[1,2,3,4,5]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    args = parser.parse_args()

    model_name = args.model
    fold = args.fold

    seed = 3000 + fold

    seed_everything(
        seed
    )

    # --------------------------------------------------------
    # Outer development/test split
    # --------------------------------------------------------

    dev_csv = (
        FOLD_ROOT /
        f"fold{fold}_train.csv"
    )

    test_csv = (
        FOLD_ROOT /
        f"fold{fold}_test.csv"
    )

    dev_df = pd.read_csv(
        dev_csv
    )

    test_df = pd.read_csv(
        test_csv
    )

    dev_df["case_id"] = (
        dev_df["case_id"]
        .astype(str)
    )

    test_df["case_id"] = (
        test_df["case_id"]
        .astype(str)
    )

    outer_test_cases = set(
        test_df["case_id"]
    )

    # --------------------------------------------------------
    # Internal patient-level train/validation
    # --------------------------------------------------------

    train_cases, val_cases = (
        split_train_val_cases(
            dev_df,
            seed
        )
    )

    train_case_set = set(
        train_cases
    )

    val_case_set = set(
        val_cases
    )

    # Leakage audit
    assert not (
        train_case_set
        &
        val_case_set
    )

    assert not (
        train_case_set
        &
        outer_test_cases
    )

    assert not (
        val_case_set
        &
        outer_test_cases
    )

    # --------------------------------------------------------
    # Global patch index
    # --------------------------------------------------------

    patch_df = pd.read_csv(
        PATCH_CSV
    )

    patch_df["case_id"] = (
        patch_df["case_id"]
        .astype(str)
    )

    # V2 preprocessing paths.
    # Patch coordinates remain valid because crop geometry is unchanged;
    # only normalization order changed.
    patch_df["image_npy"] = patch_df["case_id"].apply(
        lambda x:
        f"3_preprocessed_v2/images/{x}_crop_img.npy"
    )

    patch_df["label_npy"] = patch_df["case_id"].apply(
        lambda x:
        f"3_preprocessed_v2/labels/{x}_crop_lab.npy"
    )

    train_patch_df = patch_df[
        patch_df["case_id"].isin(
            train_case_set
        )
    ].copy()

    val_patch_df = patch_df[
        patch_df["case_id"].isin(
            val_case_set
        )
    ].copy()

    # Patient-level validation cases for whole-volume inference.
    # Use one row per case from the outer development CSV.
    val_case_df = (
        dev_df[
            dev_df["case_id"].isin(
                val_case_set
            )
        ]
        .drop_duplicates(
            subset=["case_id"]
        )
        .copy()
    )

    if len(val_case_df) != len(val_case_set):
        raise RuntimeError(
            "Validation case dataframe mismatch: "
            f"expected={len(val_case_set)}, "
            f"found={len(val_case_df)}"
        )

    # Final leakage audit
    train_patch_cases = set(
        train_patch_df[
            "case_id"
        ].unique()
    )

    val_patch_cases = set(
        val_patch_df[
            "case_id"
        ].unique()
    )

    if train_patch_cases & outer_test_cases:

        raise RuntimeError(
            "Outer-test leakage in "
            "training patches."
        )

    if val_patch_cases & outer_test_cases:

        raise RuntimeError(
            "Outer-test leakage in "
            "validation patches."
        )

    if train_patch_cases & val_patch_cases:

        raise RuntimeError(
            "Train-validation leakage."
        )

    # --------------------------------------------------------
    # Datasets/loaders
    # --------------------------------------------------------

    # --------------------------------------------------------
    # V4 dynamic patient-balanced training dataset
    # --------------------------------------------------------

    train_case_df = (
        dev_df[
            dev_df["case_id"].isin(
                train_case_set
            )
        ]
        .drop_duplicates(
            subset=["case_id"]
        )
        .copy()
    )

    train_dataset = DynamicBalancedPatchDataset(
        train_case_df,
        patches_per_case=TRAIN_PATCHES_PER_CASE,
        fg_fraction=FG_PATCH_FRACTION,
        augment=True
    )

    # Patch-level validation is retained only as
    # a secondary monitoring metric.
    val_dataset = PatchDataset(
        val_patch_df,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Device/model
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        model_name
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    criterion = CombinedLoss().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    checkpoint = (
        CKPT_ROOT /
        f"fold{fold}_{model_name}_best.pth"
    )

    log_file = (
        RESULT_ROOT /
        f"fold{fold}_{model_name}_training_log.csv"
    )

    split_file = (
        RESULT_ROOT /
        f"fold{fold}_{model_name}_internal_split.csv"
    )

    split_rows = []

    for c in train_cases:
        split_rows.append({
            "case_id": c,
            "subset": "train"
        })

    for c in val_cases:
        split_rows.append({
            "case_id": c,
            "subset": "validation"
        })

    for c in sorted(
        outer_test_cases
    ):
        split_rows.append({
            "case_id": c,
            "subset": "outer_test"
        })

    pd.DataFrame(
        split_rows
    ).to_csv(
        split_file,
        index=False
    )

    print("=" * 90)
    print("DEDUP15 TRAINING V4")
    print("=" * 90)

    print("Model            :", model_name)
    print("Fold             :", fold)
    print("Seed             :", seed)

    print("\nOuter development:", len(dev_df))
    print("Outer test       :", len(test_df))

    print("\nInternal train cases:")
    print(train_cases)

    print("\nInternal validation cases:")
    print(val_cases)

    print("\nOuter test cases:")
    print(sorted(outer_test_cases))

    print("\nDynamic train patches/epoch :", len(train_dataset))
    print("Train patches per case      :", TRAIN_PATCHES_PER_CASE)
    print("Foreground sampling fraction:", FG_PATCH_FRACTION)
    print("Context sampling fraction   :", 1.0 - FG_PATCH_FRACTION)
    print("Validation cases            :", len(val_case_df))
    print("Validation mode  : whole-volume sliding window")
    print("SW patch size    :", SW_PATCH_SIZE)
    print("SW stride        :", SW_STRIDE)

    print("\nPatch size       :", PATCH_SIZE)
    print("Batch size       :", BATCH_SIZE)
    print("Epochs           :", args.epochs)

    print("LR               :", LR)
    print("Weight decay     :", WEIGHT_DECAY)

    print("Device           :", device)

    if torch.cuda.is_available():
        print(
            "GPU              :",
            torch.cuda.get_device_name(0)
        )

    print("Parameters       :", parameters)

    print("=" * 90)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_val_dice = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss, train_dice = (
            run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer
            )
        )

        # ----------------------------------------------------
        # Patch validation loss is retained only for monitoring.
        # Checkpoint selection uses whole-volume patient Dice.
        # ----------------------------------------------------

        val_loss, val_patch_dice = (
            run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None
            )
        )

        print(
            "  Whole-volume validation:"
        )

        val_dice, val_case_dices = (
            validate_whole_volume(
                model,
                val_case_df,
                device
            )
        )

        scheduler.step(
            val_dice
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        improved = (
            val_dice >
            best_val_dice + 1e-5
        )

        if improved:

            best_val_dice = val_dice
            best_epoch = epoch
            no_improve = 0

            torch.save(
                model.state_dict(),
                checkpoint
            )

            status = "SAVED BEST"

        else:

            no_improve += 1
            status = ""

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_patch_loss": val_loss,
            "val_patch_dice": val_patch_dice,
            "val_whole_dice": val_dice,
            "learning_rate": current_lr,
            "best_val_whole_dice": best_val_dice
        }

        for i, d in enumerate(
            val_case_dices,
            start=1
        ):
            history_row[
                f"val_case_{i}_dice"
            ] = d

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            log_file,
            index=False
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| Train Loss {train_loss:.4f} "
            f"| Train Dice {train_dice:.4f} "
            f"| Val Patch Dice {val_patch_dice:.4f} "
            f"| Val Whole Dice {val_dice:.4f} "
            f"| LR {current_lr:.2e} "
            f"| {status}"
        )

        if (
            no_improve >=
            EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping at "
                f"epoch {epoch}."
            )

            break

    print("\n" + "=" * 90)
    print("TRAINING V4 FINISHED")
    print("=" * 90)

    print("Model         :", model_name)
    print("Fold          :", fold)
    print("Best epoch    :", best_epoch)
    print(
        "Best Whole-volume Val Dice :",
        f"{best_val_dice:.6f}"
    )
    print("Checkpoint    :", checkpoint)
    print("Log           :", log_file)
    print("Split         :", split_file)

    print("=" * 90)


if __name__ == "__main__":
    main()
