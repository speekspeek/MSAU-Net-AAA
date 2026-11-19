#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import auto_roi_5fold_v6_end2end as v6

CASE = "Anev_04"
FOLD = 1


def first_foreground_index(mask, axis):
    other_axes = tuple(dim for dim in range(3) if dim != axis)
    occupied = np.flatnonzero(mask.any(axis=other_axes))
    return int(occupied[0])


def check_gt_alignment(original_gt, full_gt):
    if original_gt.shape == full_gt.shape and np.array_equal(original_gt, full_gt):
        return {"status": "exact_full_grid_match", "offset": [0, 0, 0]}

    if original_gt.sum() != full_gt.sum():
        return {"status": "foreground_count_mismatch"}

    offset = [
        first_foreground_index(full_gt, axis)
        - first_foreground_index(original_gt, axis)
        for axis in range(3)
    ]
    end = [start + size for start, size in zip(offset, original_gt.shape)]
    if any(start < 0 or stop > size
           for start, stop, size in zip(offset, end, full_gt.shape)):
        return {"status": "crop_outside_full_grid"}

    region = tuple(slice(start, stop) for start, stop in zip(offset, end))
    if not np.array_equal(original_gt, full_gt[region]):
        return {"status": "ground_truth_masks_do_not_match"}

    return {"status": "exact_crop_match_by_gt_translation", "offset": offset}


def main():
    out_dir = Path(v6.ROOT) / "9_results/auto_roi_v6_audit_anev04/fold1"
    v6_result = json.loads(
        (out_dir / "Anev_04_audit.json").read_text(encoding="utf-8")
    )
    if v6_result.get("case", v6_result.get("case_id")) != CASE:
        raise RuntimeError("V6 audit case does not match Anev_04")
    if int(v6_result["fold"]) != FOLD:
        raise RuntimeError("V6 audit fold does not match Fold 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_mod = v6.load_fine_model(FOLD, device)

    test_csv = Path(train_mod.FOLD_ROOT) / f"fold{FOLD}_test.csv"
    test_df = pd.read_csv(test_csv, dtype={"case_id": str})
    matches = test_df.loc[test_df["case_id"] == CASE]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {CASE} row in {test_csv}, got {len(matches)}")

    row = matches.iloc[0]
    image_path = Path(train_mod.ROOT) / row["image_npy"]
    label_path = Path(train_mod.ROOT) / row["label_npy"]
    image = np.load(image_path, allow_pickle=False).astype(np.float32)
    gt = np.load(label_path, allow_pickle=False) > 0
    if image.ndim != 3 or image.shape != gt.shape or not gt.any():
        raise RuntimeError(f"Invalid original image or GT: {image.shape}, {gt.shape}")

    print(f"Original input: {image_path}", flush=True)
    print(f"Original shape: {image.shape}, device: {device}", flush=True)

    prediction = np.asarray(
        train_mod.sliding_window_predict(model, image, device), dtype=bool
    )
    if prediction.shape != gt.shape:
        raise RuntimeError("Original prediction and GT shapes differ")

    tp = int(np.count_nonzero(prediction & gt))
    original_dice = float(train_mod.binary_dice(prediction, gt))
    original_recall = tp / int(gt.sum())

    _, full_gt, _, _ = v6.v1.load_volume(CASE, need_gt=True)
    full_gt = np.asarray(full_gt, dtype=bool)
    if full_gt.ndim != 3 or not full_gt.any():
        raise RuntimeError("Invalid full CTA ground truth")
    alignment = check_gt_alignment(gt, full_gt)

    v6_dice = v6_result.get("final_dice")
    if v6_dice is None:
        v6_dice = v6_result["segmentation_metrics"]["Dice"]
    v6_recall = float(v6_result["recall_after_voting"])

    report = {
        "case": CASE,
        "fold": FOLD,
        "original_image": str(image_path),
        "original_gt": str(label_path),
        "original_shape": list(image.shape),
        "full_gt_shape": list(full_gt.shape),
        "original_gt_voxels": int(gt.sum()),
        "full_gt_voxels": int(full_gt.sum()),
        "gt_alignment": alignment,
        "original_dice": original_dice,
        "original_recall": original_recall,
        "v6_dice": float(v6_dice),
        "v6_recall": v6_recall,
    }

    if alignment["status"].startswith("exact_"):
        # The matched crop contains all full-grid GT foreground.
        # Zero-filling the original prediction outside it preserves TP, FP and FN.
        report["dice_difference_original_minus_v6"] = original_dice - float(v6_dice)
        report["recall_difference_original_minus_v6"] = original_recall - v6_recall

    output = out_dir / "Anev_04_original_vs_v6.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nAnev_04 PAIRED COMPARISON")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
