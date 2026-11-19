#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V6: Multi-candidate segmented ROI feasibility audit.

Inputs:
  V3 internal OOF probability maps
  12 internal OOF cases per outer fold

No training.
No outer-test access.
No GT used for ROI generation.

Exact ROI union and GT coverage are calculated
with 2D integral images and inclusion-exclusion.

Run:
  python -u 6_auto_roi/auto_roi_5fold_v6.py audit --fold 1
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
OUT = ROOT / "9_results/auto_roi_5fold_v6"

# Small, prespecified candidate grid.
AXES = [0, 1, 2]
SLABS = [8, 16]
THRESHOLDS = [0.25, 0.30]
MARGINS = [12, 24]
TOP_K = [1, 3]

TARGET_COVERAGE = 0.98
MAX_ROI_FRACTION = 0.65

spec = importlib.util.spec_from_file_location(
    "aaa_v1",
    str(V1_PATH)
)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


def configurations():
    return [
        dict(
            axis=a,
            slab=s,
            threshold=t,
            margin=m,
            top_k=k
        )
        for a, s, t, m, k in itertools.product(
            AXES, SLABS, THRESHOLDS, MARGINS, TOP_K
        )
    ]


def component_candidates(prob, threshold, axis, slab):
    """
    Extract multiple 3D components in each slab.

    Each component has:
      probability mass,
      coarse bounding box,
      transverse center.

    No GT is accessed.
    """
    moved = np.moveaxis(prob, axis, 0)
    other = [a for a in range(3) if a != axis]

    all_slabs = []

    for start in range(0, moved.shape[0], slab):

        stop = min(start + slab, moved.shape[0])

        part = moved[start:stop]
        binary = part >= threshold

        components, n = label(
            binary,
            structure=np.ones(
                (3, 3, 3),
                dtype=np.uint8
            )
        )

        objects = find_objects(components)
        candidates = []

        for cid, obj in enumerate(objects, start=1):

            if obj is None:
                continue

            local_mask = (
                components[obj] == cid
            )

            if not local_mask.any():
                continue

            weights = part[obj][local_mask]
            mass = float(weights.sum())

            coords = np.argwhere(local_mask)

            # Bounding box in original coarse axes.
            lo = np.zeros(3, dtype=int)
            hi = np.zeros(3, dtype=int)

            lo[axis] = start + obj[0].start
            hi[axis] = start + obj[0].stop

            for j, original_axis in enumerate(other):
                lo[original_axis] = obj[j + 1].start
                hi[original_axis] = obj[j + 1].stop

            center = np.array([
                (lo[a] + hi[a]) / 2
                for a in other
            ], dtype=float)

            candidates.append({
                "mass": mass,
                "low": lo,
                "high": hi,
                "center": center
            })

        candidates.sort(
            key=lambda c: c["mass"],
            reverse=True
        )

        all_slabs.append({
            "start": start,
            "stop": stop,
            "candidates": candidates
        })

    return all_slabs


def build_boxes(
    slab_candidates,
    coarse_shape,
    full_shape,
    config
):
    """
    Select up to top_k candidates per slab.

    Candidate ordering uses probability mass,
    with a modest preference for continuity
    with the previous slab.

    Boxes remain confined to disjoint
    longitudinal slabs.

    Multiple boxes within a slab may overlap.
    Exact union is handled during evaluation.
    """
    axis = config["axis"]
    margin = config["margin"]
    top_k = config["top_k"]

    other = [a for a in range(3) if a != axis]

    coarse_shape = np.asarray(
        coarse_shape,
        dtype=float
    )

    full_shape = np.asarray(
        full_shape,
        dtype=int
    )

    previous_centers = []
    boxes = []

    for item in slab_candidates:

        candidates = item["candidates"]

        if not candidates:
            previous_centers = []
            continue

        ranked = []

        for candidate in candidates:

            mass = candidate["mass"]

            if previous_centers:
                distance = min(
                    np.linalg.norm(
                        candidate["center"] - c
                    )
                    for c in previous_centers
                )

                # Soft continuity preference.
                # Never rejects a candidate solely
                # because it is distant.
                continuity = 1.0 + 0.5 * np.exp(
                    -distance / 8.0
                )
            else:
                continuity = 1.0

            ranked.append((
                mass * continuity,
                candidate
            ))

        ranked.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected = [
            c for _, c in ranked[:top_k]
        ]

        previous_centers = [
            c["center"]
            for c in selected
        ]

        for candidate in selected:

            lo = np.zeros(3, dtype=int)
            hi = np.zeros(3, dtype=int)

            # Disjoint longitudinal slab bounds.
            lo[axis] = int(np.floor(
                item["start"]
                * full_shape[axis]
                / coarse_shape[axis]
            ))

            hi[axis] = int(np.floor(
                item["stop"]
                * full_shape[axis]
                / coarse_shape[axis]
            ))

            for a in other:

                lo[a] = int(np.floor(
                    candidate["low"][a]
                    * full_shape[a]
                    / coarse_shape[a]
                )) - margin

                hi[a] = int(np.ceil(
                    candidate["high"][a]
                    * full_shape[a]
                    / coarse_shape[a]
                )) + margin

            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, full_shape)

            if np.any(hi <= lo):
                continue

            boxes.append({
                "slab_start": item["start"],
                "slab_stop": item["stop"],
                "low": lo.tolist(),
                "high": hi.tolist()
            })

    return boxes


def integral_image(array):
    """
    Summed-area table with zero padding.
    Uses int64 for GT counts and ROI area.
    """
    a = array.astype(np.int64, copy=False)

    out = np.pad(
        a,
        ((1, 0), (1, 0)),
        mode="constant"
    )

    return out.cumsum(axis=0).cumsum(axis=1)


def rectangle_sum(ii, x0, y0, x1, y1):
    return int(
        ii[x1, y1]
        - ii[x0, y1]
        - ii[x1, y0]
        + ii[x0, y0]
    )


def union_rectangle_sum(ii, rectangles):
    """
    Exact union of up to 3 rectangles,
    using inclusion-exclusion.
    """
    result = 0

    for r in range(1, len(rectangles) + 1):

        sign = 1 if r % 2 else -1

        for subset in itertools.combinations(
            rectangles, r
        ):

            x0 = max(s[0] for s in subset)
            y0 = max(s[1] for s in subset)
            x1 = min(s[2] for s in subset)
            y1 = min(s[3] for s in subset)

            if x1 <= x0 or y1 <= y0:
                continue

            result += sign * rectangle_sum(
                ii, x0, y0, x1, y1
            )

    return int(result)


def evaluate_boxes(gt, boxes, axis):
    """
    Exact union-volume and GT coverage.

    Slabs do not overlap along axis.
    Rectangles within each slab are combined
    by inclusion-exclusion.
    """
    total_gt = int(gt.sum())
    total_volume = int(gt.size)

    other = [a for a in range(3) if a != axis]

    grouped = {}

    for box in boxes:
        key = (
            box["slab_start"],
            box["slab_stop"]
        )
        grouped.setdefault(key, []).append(box)

    inside_gt = 0
    union_volume = 0

    for _, group in grouped.items():

        first = group[0]

        lo = first["low"][axis]
        hi = first["high"][axis]

        if hi <= lo:
            continue

        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)

        # Sum GT over the longitudinal slab.
        projection = gt[tuple(sl)].sum(
            axis=axis,
            dtype=np.int64
        )

        gt_ii = integral_image(projection)

        area_ii = integral_image(
            np.ones(
                projection.shape,
                dtype=np.int64
            )
        )

        rectangles = []

        for box in group:

            low = box["low"]
            high = box["high"]

            rectangles.append((
                low[other[0]],
                low[other[1]],
                high[other[0]],
                high[other[1]]
            ))

        inside_gt += union_rectangle_sum(
            gt_ii,
            rectangles
        )

        area = union_rectangle_sum(
            area_ii,
            rectangles
        )

        union_volume += area * (hi - lo)

    if inside_gt > total_gt:
        raise RuntimeError(
            "GT coverage double-counting"
        )

    if union_volume > total_volume:
        raise RuntimeError(
            "ROI union volume exceeds CTA volume"
        )

    return {
        "gt_coverage": (
            inside_gt / total_gt
            if total_gt else float("nan")
        ),
        "roi_fraction": (
            union_volume / total_volume
        ),
        "gt_inside": inside_gt,
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

    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)

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
            "Internal OOF case verification failed"
        )

    configs = configurations()

    print("=" * 100)
    print("V6 MULTI-CANDIDATE SEGMENTED ROI AUDIT")
    print("=" * 100)
    print("Outer fold:", fold)
    print("Internal OOF cases:", len(actual))
    print("Configurations:", len(configs))
    print("Excluded outer test:", v1.FOLDS[fold])

    records = []
    selected_box_cache = {}

    for item in metadata:

        case = str(item["case_id"])

        print(
            "Evaluating:",
            case,
            flush=True
        )

        probability = np.load(
            v3_folder /
            "oof_probabilities" /
            f"{case}_prob.npy"
        )

        # GT is used only for audit.
        _, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        # Reuse component extraction across
        # margin and top_k configurations.
        candidate_cache = {}

        for axis, slab, threshold in itertools.product(
            AXES, SLABS, THRESHOLDS
        ):
            candidate_cache[
                (axis, slab, threshold)
            ] = component_candidates(
                probability,
                threshold,
                axis,
                slab
            )

        for config_id, config in enumerate(configs):

            key = (
                config["axis"],
                config["slab"],
                config["threshold"]
            )

            boxes = build_boxes(
                candidate_cache[key],
                probability.shape,
                gt.shape,
                config
            )

            result = evaluate_boxes(
                gt,
                boxes,
                config["axis"]
            )

            records.append({
                "case_id": case,
                "config_id": config_id,
                **config,
                **result
            })

            selected_box_cache[
                (case, config_id)
            ] = boxes

        del gt, probability, candidate_cache

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
            ["mean_roi_fraction", "min_coverage"],
            ascending=[True, False]
        ).iloc[0]

        status = "FEASIBLE_CONFIGURATION_FOUND"

    else:

        selected = summary.sort_values(
            [
                "failed_coverage_cases",
                "oversized_cases",
                "min_coverage",
                "mean_roi_fraction"
            ],
            ascending=[True, True, False, True]
        ).iloc[0]

        status = "NO_FEASIBLE_CONFIGURATION"

    selected_id = int(
        selected["config_id"]
    )

    chosen = {
        **configs[selected_id],
        "config_id": selected_id,
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
        json.dump(chosen, f, indent=2)

    chosen_cases = results[
        results["config_id"] == selected_id
    ].copy()

    chosen_boxes = {
        case: selected_box_cache[
            (case, selected_id)
        ]
        for case in actual
    }

    with open(
        fold_out / "selected_boxes.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chosen_boxes,
            f,
            indent=2
        )

    v4_file = (
        v4_folder / "selected_case_results.csv"
    )

    if v4_file.exists():

        v4 = pd.read_csv(v4_file)[
            ["case_id", "gt_coverage", "roi_fraction"]
        ].rename(columns={
            "gt_coverage": "v4_gt_coverage",
            "roi_fraction": "v4_roi_fraction"
        })

        chosen_cases = chosen_cases.merge(
            v4,
            on="case_id",
            how="left",
            validate="one_to_one"
        )

        chosen_cases["volume_reduction"] = (
            chosen_cases["v4_roi_fraction"]
            - chosen_cases["roi_fraction"]
        )

        chosen_cases["coverage_difference"] = (
            chosen_cases["gt_coverage"]
            - chosen_cases["v4_gt_coverage"]
        )

    chosen_cases.to_csv(
        fold_out / "selected_case_results.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("V6 SELECTED CONFIGURATION")
    print("=" * 100)

    print(json.dumps(chosen, indent=2))

    columns = [
        "case_id",
        "gt_coverage",
        "roi_fraction",
        "number_of_boxes"
    ]

    if "volume_reduction" in chosen_cases:
        columns.append("volume_reduction")

    print("\n12-CASE RESULTS")

    print(
        chosen_cases[columns]
        .sort_values("gt_coverage")
        .to_string(index=False)
    )

    print("\nFeasible configurations:", len(feasible))
    print("Saved:", fold_out)

    if not len(feasible):
        print(
            "\nWARNING: No configuration meets "
            "both prespecified constraints."
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
