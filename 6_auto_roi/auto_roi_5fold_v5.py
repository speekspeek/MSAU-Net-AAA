#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V5: Multi-segment ROI feasibility audit.

Input:
  V3 internal OOF probability maps (12 cases)

Methods:
  - 2D connected-component tracking
  - Spatial continuity constraint
  - Non-overlapping slab-wise ROI boxes
  - Exact union volume of the boxes
  - GT coverage audit
  - Paired comparison with V4

No training.
No outer-test GT access.
No modification to V1-V4.

Run:
  python -u 6_auto_roi/auto_roi_5fold_v5.py audit --fold 1
"""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import label, find_objects

ROOT = Path.home() / "AAA_CTA_Project"

V1_PATH = ROOT / "6_auto_roi/auto_roi_5fold.py"

V3_ROOT = ROOT / "9_results/auto_roi_5fold_v3"
V4_ROOT = ROOT / "9_results/auto_roi_5fold_v4"
OUT = ROOT / "9_results/auto_roi_5fold_v5"

THRESHOLDS = [0.30, 0.40, 0.50]
SLAB_SIZES = [4, 8, 16]
MARGINS = [12, 24, 40]
AXES = [0, 1, 2]

TARGET_COVERAGE = 0.98
MAX_ROI_FRACTION = 0.65

# Maximum permitted displacement between
# consecutive 2D component centers.
MAX_CENTER_JUMP = 16.0

spec = importlib.util.spec_from_file_location(
    "aaa_v1",
    str(V1_PATH)
)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


def configurations():
    result = []
    for axis, slab, threshold, margin in itertools.product(
        AXES, SLAB_SIZES, THRESHOLDS, MARGINS
    ):
        result.append({
            "axis": axis,
            "slab": slab,
            "threshold": threshold,
            "margin": margin,
        })
    return result


def component_track(probability, axis, threshold):
    """
    Track a 2D component along a candidate axis.

    The first valid slice uses its largest component.
    Subsequent slices prefer spatially continuous
    components, with a center-distance constraint.

    GT is never used.
    """
    moved = np.moveaxis(probability, axis, 0)

    length = moved.shape[0]

    tracked = np.zeros(
        moved.shape,
        dtype=np.uint8
    )

    previous_center = None

    structure = np.ones((3, 3), dtype=np.uint8)

    for z in range(length):

        binary = moved[z] >= threshold

        components, n = label(
            binary,
            structure=structure
        )

        if n == 0:
            continue

        sizes = np.bincount(
            components.ravel()
        )
        sizes[0] = 0

        # Consider the largest five components.
        ranked = (
            np.argsort(sizes[1:])[::-1] + 1
        )[:5]

        candidates = []

        for component_id in ranked:

            if sizes[component_id] == 0:
                continue

            coords = np.argwhere(
                components == component_id
            )

            center = coords.mean(axis=0)

            candidates.append((
                int(component_id),
                center,
                int(sizes[component_id])
            ))

        if not candidates:
            continue

        if previous_center is None:
            selected = max(
                candidates,
                key=lambda x: x[2]
            )
        else:
            distances = [
                np.linalg.norm(
                    c[1] - previous_center
                )
                for c in candidates
            ]

            nearest_index = int(
                np.argmin(distances)
            )

            if distances[nearest_index] > MAX_CENTER_JUMP:
                # Reject a large spatial jump.
                # The previous center is retained.
                continue

            selected = candidates[nearest_index]

        component_id, center, _ = selected

        tracked[z] = (
            components == component_id
        ).astype(np.uint8)

        previous_center = center

    return tracked


def make_segment_boxes(
    probability,
    full_shape,
    config
):
    """
    Build disjoint slab-wise rectangular ROIs.

    Each box is confined to one slab along the
    selected axis. Therefore boxes do not overlap
    in the longitudinal dimension, allowing exact
    union-volume and GT-coverage calculations.
    """

    axis = int(config["axis"])
    slab = int(config["slab"])
    threshold = float(config["threshold"])
    margin = int(config["margin"])

    coarse_shape = np.asarray(
        probability.shape,
        dtype=int
    )

    full_shape = np.asarray(
        full_shape,
        dtype=int
    )

    tracked = component_track(
        probability,
        axis,
        threshold
    )

    other_axes = [
        a for a in range(3)
        if a != axis
    ]

    boxes = []

    n = tracked.shape[0]

    for start in range(0, n, slab):

        stop = min(start + slab, n)

        slab_mask = tracked[start:stop]

        coords = np.argwhere(
            slab_mask > 0
        )

        if len(coords) == 0:
            continue

        # Coordinates of the two transverse axes.
        low_2d = coords[:, 1:].min(axis=0)
        high_2d = coords[:, 1:].max(axis=0) + 1

        low = np.zeros(3, dtype=int)
        high = np.zeros(3, dtype=int)

        # Longitudinal slab boundaries.
        # Use the same floor mapping at every
        # boundary to prevent overlaps.

        low[axis] = int(
            np.floor(
                start * full_shape[axis]
                / coarse_shape[axis]
            )
        )

        high[axis] = int(
            np.floor(
                stop * full_shape[axis]
                / coarse_shape[axis]
            )
        )

        # Expand transverse directions only.
        for j, a in enumerate(other_axes):

            low[a] = int(
                np.floor(
                    low_2d[j] * full_shape[a]
                    / coarse_shape[a]
                )
            ) - margin

            high[a] = int(
                np.ceil(
                    high_2d[j] * full_shape[a]
                    / coarse_shape[a]
                )
            ) + margin

        low = np.maximum(low, 0)
        high = np.minimum(high, full_shape)

        if np.any(high <= low):
            continue

        boxes.append((
            low.tolist(),
            high.tolist()
        ))

    return boxes


def evaluate_boxes(gt, boxes):
    """
    Exact GT coverage and union volume.

    Boxes occupy disjoint longitudinal slabs.
    Therefore voxel counts can be summed without
    double-counting overlap.
    """

    total_gt = int(gt.sum())

    total_volume = int(gt.size)

    inside_gt = 0
    union_volume = 0

    for low, high in boxes:

        low = np.asarray(low, dtype=int)
        high = np.asarray(high, dtype=int)

        union_volume += int(
            np.prod(high - low)
        )

        inside_gt += int(
            gt[
                low[0]:high[0],
                low[1]:high[1],
                low[2]:high[2]
            ].sum()
        )

    if union_volume > total_volume:
        raise RuntimeError(
            "Overlapping boxes detected"
        )

    if inside_gt > total_gt:
        raise RuntimeError(
            "GT double counting detected"
        )

    coverage = (
        inside_gt / total_gt
        if total_gt
        else float("nan")
    )

    fraction = union_volume / total_volume

    return {
        "gt_coverage": float(coverage),
        "roi_fraction": float(fraction),
        "gt_inside": inside_gt,
        "gt_total": total_gt,
        "gt_outside": total_gt - inside_gt,
        "roi_voxels": union_volume,
        "number_of_boxes": len(boxes)
    }


def audit(fold):

    v1.verify_cases()

    v3_folder = V3_ROOT / f"fold{fold}"
    v4_folder = V4_ROOT / f"fold{fold}"
    fold_out = OUT / f"fold{fold}"

    fold_out.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = (
        v3_folder / "oof_metadata.json"
    )

    with open(
        metadata_file,
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    expected = (
        set(v1.verify_cases())
        - set(v1.FOLDS[fold])
    )

    actual = [
        str(item["case_id"])
        for item in metadata
    ]

    if (
        len(actual) != 12
        or len(set(actual)) != 12
        or set(actual) != expected
    ):
        raise RuntimeError(
            "OOF case verification failed"
        )

    configs = configurations()

    print("=" * 100)
    print("V5 SEGMENTED ROI FEASIBILITY AUDIT")
    print("=" * 100)
    print("Outer fold:", fold)
    print("Internal OOF cases:", len(metadata))
    print("Candidate configurations:", len(configs))
    print("Outer-test cases excluded:", v1.FOLDS[fold])

    records = []

    # Keep the selected boxes for each candidate
    # so that the chosen result is reproducible.
    box_records = {}

    for item in metadata:

        case = str(item["case_id"])

        print(
            "Evaluating:",
            case,
            flush=True
        )

        prob_path = (
            v3_folder /
            "oof_probabilities" /
            f"{case}_prob.npy"
        )

        probability = np.load(
            prob_path
        )

        # GT is used for audit only.
        _, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        for config_id, config in enumerate(configs):

            boxes = make_segment_boxes(
                probability,
                gt.shape,
                config
            )

            result = evaluate_boxes(
                gt,
                boxes
            )

            records.append({
                "case_id": case,
                "config_id": config_id,
                **config,
                **result
            })

            box_records[(case, config_id)] = boxes

        del probability, gt

    results = pd.DataFrame(records)

    results.to_csv(
        fold_out / "candidate_results.csv",
        index=False
    )

    summary = results.groupby(
        "config_id"
    ).agg(
        min_coverage=("gt_coverage", "min"),
        mean_coverage=("gt_coverage", "mean"),
        mean_roi_fraction=("roi_fraction", "mean"),
        max_roi_fraction=("roi_fraction", "max"),
        mean_boxes=("number_of_boxes", "mean"),
        failed_coverage_cases=(
            "gt_coverage",
            lambda x: int(
                (x < TARGET_COVERAGE).sum()
            )
        ),
        oversized_cases=(
            "roi_fraction",
            lambda x: int(
                (x > MAX_ROI_FRACTION).sum()
            )
        )
    ).reset_index()

    config_df = pd.DataFrame(configs)

    config_df.insert(
        0,
        "config_id",
        range(len(configs))
    )

    summary = summary.merge(
        config_df,
        on="config_id",
        validate="one_to_one"
    )

    summary["feasible"] = (
        (summary["min_coverage"] >= TARGET_COVERAGE)
        &
        (summary["max_roi_fraction"] <= MAX_ROI_FRACTION)
    )

    summary.to_csv(
        fold_out / "candidate_summary.csv",
        index=False
    )

    feasible = summary[
        summary["feasible"]
    ].copy()

    if len(feasible):

        selected = feasible.sort_values(
            [
                "mean_roi_fraction",
                "min_coverage"
            ],
            ascending=[True, False]
        ).iloc[0]

        status = "FEASIBLE_CONFIGURATION_FOUND"

    else:

        # Diagnostic selection only.
        # Do not treat this as a validated method.

        selected = summary.sort_values(
            [
                "failed_coverage_cases",
                "oversized_cases",
                "min_coverage",
                "mean_roi_fraction"
            ],
            ascending=[
                True, True, False, True
            ]
        ).iloc[0]

        status = "NO_FEASIBLE_CONFIGURATION"

    config_id = int(
        selected["config_id"]
    )

    chosen = {
        **configs[config_id],
        "config_id": config_id,
        "fold": fold,
        "status": status,
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
        "failed_coverage_cases": int(
            selected["failed_coverage_cases"]
        ),
        "oversized_cases": int(
            selected["oversized_cases"]
        ),
        "feasible_configurations": int(
            len(feasible)
        )
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

    selected_cases = results[
        results["config_id"] == config_id
    ].copy()

    # Save actual ROI coordinates.
    selected_boxes = {
        case: box_records[(case, config_id)]
        for case in actual
    }

    with open(
        fold_out / "selected_boxes.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            selected_boxes,
            f,
            indent=2
        )

    # V4 comparison.
    v4_file = (
        v4_folder / "selected_case_results.csv"
    )

    if v4_file.exists():

        v4 = pd.read_csv(v4_file)[
            [
                "case_id",
                "gt_coverage",
                "roi_fraction"
            ]
        ].rename(columns={
            "gt_coverage": "v4_gt_coverage",
            "roi_fraction": "v4_roi_fraction"
        })

        selected_cases = selected_cases.merge(
            v4,
            on="case_id",
            how="left",
            validate="one_to_one"
        )

        selected_cases["volume_reduction"] = (
            selected_cases["v4_roi_fraction"]
            - selected_cases["roi_fraction"]
        )

        selected_cases["coverage_difference"] = (
            selected_cases["gt_coverage"]
            - selected_cases["v4_gt_coverage"]
        )

    selected_cases.to_csv(
        fold_out / "selected_case_results.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("V5 SELECTED CONFIGURATION")
    print("=" * 100)

    print(
        json.dumps(
            chosen,
            indent=2
        )
    )

    print("\n12-CASE RESULTS")

    columns = [
        "case_id",
        "gt_coverage",
        "roi_fraction",
        "number_of_boxes"
    ]

    if "v4_roi_fraction" in selected_cases.columns:
        columns.extend([
            "v4_roi_fraction",
            "volume_reduction"
        ])

    print(
        selected_cases[columns]
        .sort_values("gt_coverage")
        .to_string(index=False)
    )

    print("\nFeasible configurations:", len(feasible))
    print("Saved:", fold_out)

    if not len(feasible):
        print(
            "\nWARNING: Coverage and ROI-volume "
            "constraints cannot both be satisfied."
        )
        print(
            "Do not proceed to final training."
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["audit"]
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 6)
    )

    args = parser.parse_args()

    audit(args.fold)


if __name__ == "__main__":
    main()
