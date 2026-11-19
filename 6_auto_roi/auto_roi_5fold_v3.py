#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V3: Recall-oriented 3D U-Net locator with internal OOF audit.

Fold 1:
    12 outer-training cases
    -> 3 internal folds
    -> 8 training / 4 held-out cases per fold
    -> 12 out-of-fold localization predictions
    -> select ROI parameters using internal OOF only

No outer-test case is used during training or calibration.

Commands:
    python -u 6_auto_roi/auto_roi_5fold_v3.py audit --fold 1
    python -u 6_auto_roi/auto_roi_5fold_v3.py final-train --fold 1
"""

import argparse
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.ndimage import (
    label as cc_label,
    find_objects,
)

ROOT = Path.home() / "AAA_CTA_Project"

V1_FILE = ROOT / "6_auto_roi/auto_roi_5fold.py"

OUT = ROOT / "9_results/auto_roi_5fold_v3"

OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261110

DEFAULT_EPOCHS = 60

LR = 2e-4

THRESHOLDS = [0.20, 0.30, 0.40, 0.50]

MARGINS = [24, 40, 56]

POLICIES = ["largest", "top3", "all"]

MAX_ROI_FRACTION = 0.65

TARGET_COVERAGE = 0.98


# ============================================================
# Import original project code
# ============================================================

spec = importlib.util.spec_from_file_location(
    "aaa_locator_v1",
    str(V1_FILE)
)

v1 = importlib.util.module_from_spec(spec)

spec.loader.exec_module(v1)


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# Internal fold splitting
# ============================================================

def internal_splits(outer_fold):

    cases = v1.verify_cases()

    outer_test = set(v1.FOLDS[outer_fold])

    pool = sorted(set(cases) - outer_test)

    assert len(pool) == 12

    # Stratify by AAA/normal class.
    # Each held-out case appears exactly once.

    aaa = sorted(
        c for c in pool if c.startswith("Anev")
    )

    normal = sorted(
        c for c in pool if c.startswith("Nat")
    )

    rng = random.Random(SEED + outer_fold)

    rng.shuffle(aaa)
    rng.shuffle(normal)

    aaa_parts = np.array_split(aaa, 3)
    normal_parts = np.array_split(normal, 3)

    splits = []

    for k in range(3):

        held_out = sorted(
            list(aaa_parts[k])
            + list(normal_parts[k])
        )

        train = sorted(
            set(pool) - set(held_out)
        )

        splits.append((train, held_out))

    seen = [
        c
        for _, held_out in splits
        for c in held_out
    ]

    assert sorted(seen) == sorted(pool)
    assert len(seen) == len(set(seen))

    return splits


# ============================================================
# Coarse dataset
# ============================================================

def prepare_cases(cases):

    data = {}

    for case in cases:

        print(
            f"Loading and resampling {case}",
            flush=True
        )

        image, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        x = v1.coarse_image(image)
        y = v1.coarse_gt(gt)

        data[case] = {
            "x": torch.from_numpy(
                x[None, None].copy()
            ),
            "y": torch.from_numpy(
                y[None, None].copy()
            ),
        }

        del image, gt, x, y

    return data


# ============================================================
# Recall-oriented training objective
# ============================================================

def recall_loss(logits, target):

    prob = torch.sigmoid(logits)

    intersection = (
        prob * target
    ).sum()

    dice = (
        2 * intersection + 1
    ) / (
        prob.sum() + target.sum() + 1
    )

    recall = (
        intersection + 1
    ) / (
        target.sum() + 1
    )

    # Positive voxels receive greater weight.
    # Fixed before observing internal OOF results.

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.tensor(
            3.0,
            device=logits.device
        )
    )

    return (
        0.30 * bce
        + 0.35 * (1 - dice)
        + 0.35 * (1 - recall)
    )


# ============================================================
# Train locator
# ============================================================

def train_locator(
    train_cases,
    checkpoint,
    epochs,
    seed
):

    seed_everything(seed)

    device = get_device()

    print(
        "Training cases:",
        train_cases,
        flush=True
    )

    data = prepare_cases(train_cases)

    model = v1.LocatorUNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    history = []

    for epoch in range(1, epochs + 1):

        model.train()

        order = train_cases[:]

        random.shuffle(order)

        losses = []

        for case in order:

            x = data[case]["x"].to(device)
            y = data[case]["y"].to(device)

            # Mild intensity augmentation only.
            # No spatial orientation changes.

            scale = random.uniform(0.90, 1.10)
            shift = random.uniform(-0.10, 0.10)

            x_aug = x * scale + shift

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x_aug)

            loss = recall_loss(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            losses.append(
                float(loss.detach().cpu())
            )

        mean_loss = float(
            np.mean(losses)
        )

        history.append({
            "epoch": epoch,
            "train_loss": mean_loss
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"loss={mean_loss:.5f}",
                flush=True
            )

    # Fixed epoch count.
    # Held-out cases are not used for checkpoint selection.

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        checkpoint
    )

    pd.DataFrame(history).to_csv(
        checkpoint.with_suffix(".csv"),
        index=False
    )

    del data

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        "Saved checkpoint:",
        checkpoint,
        flush=True
    )

    return model


# ============================================================
# Locator inference
# ============================================================

def predict_probability(
    model,
    image,
    device
):

    x = torch.from_numpy(
        v1.coarse_image(image)[None, None].copy()
    ).to(device)

    model.eval()

    with torch.no_grad():

        prob = torch.sigmoid(
            model(x)
        )[0, 0].cpu().numpy()

    return prob.astype(np.float32)


# ============================================================
# ROI generation
# ============================================================

def generate_roi(
    probability,
    image_shape,
    threshold,
    margin,
    policy
):

    mask = probability >= threshold

    components, n = cc_label(
        mask,
        structure=np.ones(
            (3, 3, 3),
            dtype=np.uint8
        )
    )

    shape = np.asarray(
        image_shape,
        dtype=int
    )

    if n == 0:

        return (
            np.zeros(3, dtype=int),
            shape.copy()
        )

    sizes = np.bincount(
        components.ravel()
    )

    sizes[0] = 0

    ranked = (
        np.argsort(sizes[1:])[::-1] + 1
    )

    if policy == "largest":
        selected = ranked[:1]

    elif policy == "top3":
        selected = ranked[:3]

    elif policy == "all":
        selected = ranked

    else:
        raise ValueError(policy)

    objects = find_objects(components)

    lows = []
    highs = []

    for component_id in selected:

        sl = objects[int(component_id) - 1]

        if sl is None:
            continue

        lows.append(
            [s.start for s in sl]
        )

        highs.append(
            [s.stop for s in sl]
        )

    if not lows:

        return (
            np.zeros(3, dtype=int),
            shape.copy()
        )

    low_coarse = np.min(
        lows,
        axis=0
    )

    high_coarse = np.max(
        highs,
        axis=0
    )

    coarse_shape = np.asarray(
        probability.shape,
        dtype=float
    )

    low = np.floor(
        low_coarse * shape / coarse_shape
    ).astype(int)

    high = np.ceil(
        high_coarse * shape / coarse_shape
    ).astype(int)

    low = np.maximum(
        low - margin,
        0
    )

    high = np.minimum(
        high + margin,
        shape
    )

    return low, high


def roi_coverage(gt, low, high):

    total = int(gt.sum())

    if total == 0:
        return float("nan")

    inside = int(
        gt[
            low[0]:high[0],
            low[1]:high[1],
            low[2]:high[2]
        ].sum()
    )

    return inside / total


def roi_fraction(shape, low, high):

    return float(
        np.prod(
            (high - low)
            /
            np.asarray(shape, dtype=float)
        )
    )


# ============================================================
# Save internal OOF probabilities
# ============================================================

def audit_fold(outer_fold, epochs):

    v1.verify_cases()

    seed_everything(SEED + outer_fold)

    device = get_device()

    fold_out = OUT / f"fold{outer_fold}"

    fold_out.mkdir(
        parents=True,
        exist_ok=True
    )

    splits = internal_splits(
        outer_fold
    )

    print("=" * 100)
    print("V3 INTERNAL OOF LOCATOR AUDIT")
    print("=" * 100)

    print("Outer fold:", outer_fold)
    print("Device:", device)
    print("Epochs:", epochs)
    print(
        "Outer-test cases (excluded):",
        v1.FOLDS[outer_fold]
    )

    split_records = []

    for k, (train, held_out) in enumerate(
        splits,
        start=1
    ):

        print(
            f"Internal fold {k}: "
            f"train={train} "
            f"held_out={held_out}",
            flush=True
        )

        split_records.append({
            "internal_fold": k,
            "train_cases": train,
            "held_out_cases": held_out
        })

    with open(
        fold_out / "internal_splits.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_records,
            f,
            indent=2,
            ensure_ascii=False
        )

    prob_dir = fold_out / "oof_probabilities"

    prob_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    meta = []

    for k, (train, held_out) in enumerate(
        splits,
        start=1
    ):

        print("\n" + "=" * 100)
        print(f"INTERNAL FOLD {k}/3")
        print("=" * 100)

        checkpoint = (
            fold_out /
            "internal_checkpoints" /
            f"internal{k}_locator.pth"
        )

        model = train_locator(
            train,
            checkpoint,
            epochs,
            SEED + outer_fold * 100 + k
        )

        for case in held_out:

            print(
                "Held-out prediction:",
                case,
                flush=True
            )

            # No held-out GT is loaded for inference.
            image, _, _, _ = v1.load_volume(
                case,
                need_gt=False
            )

            prob = predict_probability(
                model,
                image,
                device
            )

            np.save(
                prob_dir / f"{case}_prob.npy",
                prob
            )

            meta.append({
                "case_id": case,
                "internal_fold": k,
                "image_shape": list(image.shape)
            })

            del image, prob

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(
        fold_out / "oof_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            meta,
            f,
            indent=2
        )

    # Audit all candidate ROI configurations.
    # GT is now loaded only for evaluation/calibration.

    print("\nEvaluating 12 internal OOF cases")

    records = []

    for item in meta:

        case = item["case_id"]

        print(
            "ROI coverage audit:",
            case,
            flush=True
        )

        _, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        probability = np.load(
            prob_dir / f"{case}_prob.npy"
        )

        for threshold, margin, policy in itertools.product(
            THRESHOLDS,
            MARGINS,
            POLICIES
        ):

            low, high = generate_roi(
                probability,
                gt.shape,
                threshold,
                margin,
                policy
            )

            coverage = roi_coverage(
                gt,
                low,
                high
            )

            fraction = roi_fraction(
                gt.shape,
                low,
                high
            )

            records.append({
                "case_id": case,
                "internal_fold": item["internal_fold"],
                "threshold": threshold,
                "margin": margin,
                "policy": policy,
                "gt_coverage": coverage,
                "roi_fraction": fraction,
                "roi_start": str(low.tolist()),
                "roi_stop_exclusive": str(
                    high.tolist()
                )
            })

        del gt, probability

    audit = pd.DataFrame(records)

    audit.to_csv(
        fold_out / "internal_oof_roi_audit.csv",
        index=False
    )

    group_cols = [
        "threshold",
        "margin",
        "policy"
    ]

    summary = audit.groupby(
        group_cols
    ).agg(
        min_coverage=(
            "gt_coverage",
            "min"
        ),
        mean_coverage=(
            "gt_coverage",
            "mean"
        ),
        mean_roi_fraction=(
            "roi_fraction",
            "mean"
        ),
        max_roi_fraction=(
            "roi_fraction",
            "max"
        )
    ).reset_index()

    summary.to_csv(
        fold_out / "candidate_summary.csv",
        index=False
    )

    # Parameter selection from internal OOF only.

    feasible = summary[
        summary["max_roi_fraction"]
        <= MAX_ROI_FRACTION
    ].copy()

    if feasible.empty:

        # Explicitly report constraint failure.
        # Do not silently choose a full-volume ROI.

        selected_pool = summary.copy()

        constraint_status = (
            "volume_constraint_not_met"
        )

    else:

        selected_pool = feasible

        constraint_status = (
            "volume_constraint_met"
        )

    target_pool = selected_pool[
        selected_pool["min_coverage"]
        >= TARGET_COVERAGE
    ]

    if not target_pool.empty:

        selected = target_pool.sort_values(
            [
                "mean_roi_fraction",
                "min_coverage",
                "mean_coverage"
            ],
            ascending=[
                True,
                False,
                False
            ]
        ).iloc[0]

        selection_status = "target_met"

    else:

        selected = selected_pool.sort_values(
            [
                "min_coverage",
                "mean_coverage",
                "mean_roi_fraction"
            ],
            ascending=[
                False,
                False,
                True
            ]
        ).iloc[0]

        selection_status = (
            "best_available_not_target_met"
        )

    chosen = {
        "outer_fold": int(outer_fold),
        "threshold": float(
            selected["threshold"]
        ),
        "margin": int(
            selected["margin"]
        ),
        "policy": str(
            selected["policy"]
        ),
        "min_coverage": float(
            selected["min_coverage"]
        ),
        "mean_coverage": float(
            selected["mean_coverage"]
        ),
        "mean_roi_fraction": float(
            selected["mean_roi_fraction"]
        ),
        "max_roi_fraction": float(
            selected["max_roi_fraction"]
        ),
        "constraint_status": constraint_status,
        "selection_status": selection_status,
        "training_epochs": int(epochs),
        "method": "3fold_internal_oof"
    }

    with open(
        fold_out / "selected_roi_config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chosen,
            f,
            indent=2
        )

    chosen_rows = audit[
        (audit["threshold"] == chosen["threshold"])
        & (audit["margin"] == chosen["margin"])
        & (audit["policy"] == chosen["policy"])
    ].copy()

    chosen_rows.to_csv(
        fold_out / "selected_case_coverage.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("SELECTED ROI CONFIGURATION")
    print("=" * 100)

    print(
        json.dumps(
            chosen,
            indent=2
        )
    )

    print("\n12 CASES — SELECTED CONFIGURATION")

    print(
        chosen_rows[
            [
                "case_id",
                "gt_coverage",
                "roi_fraction"
            ]
        ].sort_values(
            "gt_coverage"
        ).to_string(index=False)
    )

    print("\nSaved results to:", fold_out)


# ============================================================
# Final locator training on all 12 outer-training cases
# ============================================================

def final_train(outer_fold):

    fold_out = OUT / f"fold{outer_fold}"

    config_path = (
        fold_out / "selected_roi_config.json"
    )

    if not config_path.exists():

        raise FileNotFoundError(
            "Run audit first: "
            + str(config_path)
        )

    with open(
        config_path,
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    cases = v1.verify_cases()

    train_cases = sorted(
        set(cases)
        - set(v1.FOLDS[outer_fold])
    )

    assert len(train_cases) == 12

    checkpoint = (
        fold_out /
        "final_locator_12cases.pth"
    )

    print("=" * 100)
    print("V3 FINAL LOCATOR TRAINING")
    print("=" * 100)

    print("Outer fold:", outer_fold)
    print("Training cases:", train_cases)
    print("Excluded outer-test:", v1.FOLDS[outer_fold])

    print("Selected ROI config:", config)

    model = train_locator(
        train_cases,
        checkpoint,
        int(config["training_epochs"]),
        SEED + outer_fold * 1000
    )

    del model

    print("\nFinal locator saved:", checkpoint)


# ============================================================
# Command line
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "audit",
            "final-train"
        ]
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 6)
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.command == "audit":

        audit_fold(
            args.fold,
            args.epochs
        )

    elif args.command == "final-train":

        final_train(args.fold)


if __name__ == "__main__":
    main()
