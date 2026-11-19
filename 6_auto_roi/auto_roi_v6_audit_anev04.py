#!/usr/bin/env python3
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import auto_roi_5fold_v6_end2end as v6

CASE = "Anev_04"
FOLD = 1
OUTPUT = (
    v6.ROOT / "9_results/auto_roi_v6_audit_anev04"
    / "fold1/Anev_04_audit.json"
)


def input_stats(array):
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p01_p50_p99": np.percentile(array, [1, 50, 99]).tolist(),
    }


@torch.no_grad()
def main():
    v6.v1.verify_cases()
    if CASE not in v6.v1.FOLDS[FOLD]:
        raise RuntimeError(f"{CASE} is absent from Fold {FOLD}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = v6.load_config(FOLD)
    locator = v6.load_locator(FOLD, device)
    fine_model, train_mod = v6.load_fine_model(FOLD, device)

    image, _, _, _ = v6.v1.load_volume(CASE, need_gt=False)
    probability = v6.locator_predict(locator, image, device)
    candidates = v6.v6.component_candidates(
        probability,
        float(config["threshold"]),
        int(config["axis"]),
        int(config["slab"]),
    )
    boxes = v6.v6.build_boxes(
        candidates, probability.shape, image.shape, config
    )
    v6.validate_boxes(boxes, image.shape)
    if not boxes:
        raise RuntimeError("V6 generated no ROI boxes")

    print(f"Case={CASE}, Fold={FOLD}, Device={device}, ROI boxes={len(boxes)}")

    local_union = np.zeros(image.shape, dtype=bool)
    roi_records = []
    original_predict = train_mod.sliding_window_predict

    def observe_roi(model, crop, inference_device):
        index = len(roi_records)
        box = boxes[index]
        region = tuple(
            slice(int(low), int(high))
            for low, high in zip(box["low"], box["high"])
        )
        first_window = None
        forward_calls = 0

        def capture_input(_model, inputs):
            nonlocal first_window, forward_calls
            forward_calls += 1
            if first_window is None:
                first_window = input_stats(
                    inputs[0].detach().float().cpu().numpy()
                )

        hook = model.register_forward_pre_hook(capture_input)
        try:
            prediction = original_predict(model, crop, inference_device)
        finally:
            hook.remove()

        local_positive = np.asarray(prediction) > 0
        if local_positive.shape != local_union[region].shape:
            raise RuntimeError(f"ROI {index + 1}: prediction and box shapes differ")
        local_union[region] |= local_positive

        roi_records.append({
            "roi": index + 1,
            "crop_before_zscore": input_stats(image[region]),
            "crop_after_zscore": input_stats(crop),
            "first_model_window": first_window,
            "forward_calls": forward_calls,
            "positive_voxels": int(local_positive.sum()),
        })
        return prediction

    with patch.object(train_mod, "sliding_window_predict", observe_roi):
        merged, covered, roi_fraction, processing_fraction = v6.segment_rois(
            image, boxes, fine_model, train_mod, device
        )

    if len(roi_records) != len(boxes):
        raise RuntimeError("Expected one sliding-window call per ROI box")

    # Load GT after image-only inference, as in V6.
    _, gt, _, _ = v6.v1.load_volume(CASE, need_gt=True)
    gt = np.asarray(gt, dtype=bool)
    merged = np.asarray(merged, dtype=bool)
    covered = np.asarray(covered, dtype=bool)
    if gt.shape != image.shape or not gt.any():
        raise RuntimeError("GT must share the CTA grid and contain foreground")
    if np.any(merged & ~local_union) or np.any(local_union & ~covered):
        raise RuntimeError("Masks violate V6 voting containment")

    gt_total = int(gt.sum())
    gt_covered = int(np.count_nonzero(gt & covered))
    tp_before = int(np.count_nonzero(gt & local_union))
    tp_after = int(np.count_nonzero(gt & merged))
    metrics = v6.evaluate_prediction(merged, covered, gt)

    report = {
        "case": CASE,
        "fold": FOLD,
        "roi_count": len(boxes),
        "roi_gt_coverage": gt_covered / gt_total,
        "roi_fraction": float(roi_fraction),
        "processing_fraction": float(processing_fraction),
        "recall_before_voting": tp_before / gt_total,
        "recall_after_voting": tp_after / gt_total,
        "gt_outside_roi_voxels": gt_total - gt_covered,
        "tp_missed_before_voting_inside_roi": gt_covered - tp_before,
        "tp_lost_during_voting": tp_before - tp_after,
        "final_dice": float(metrics["Dice"]),
        "roi_inputs": roi_records,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nAnev_04 AUDIT RESULT")
    print(f"ROI GT coverage:       {report['roi_gt_coverage']:.6f}")
    print(f"Recall before voting:  {report['recall_before_voting']:.6f}")
    print(f"Recall after voting:   {report['recall_after_voting']:.6f}")
    print(f"TP missed within ROI:  {report['tp_missed_before_voting_inside_roi']}")
    print(f"TP removed by voting:  {report['tp_lost_during_voting']}")
    print(f"Final Dice:            {report['final_dice']:.6f}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
