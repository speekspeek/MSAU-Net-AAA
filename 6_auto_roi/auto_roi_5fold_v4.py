#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V4: Spatially constrained automatic ROI localization.

Input:
    V3 internal OOF probability maps
    12 outer-training cases

Methods:
    1. Largest connected-component bounding box
    2. Probability-weighted quantile bounding box
    3. Slice-wise continuity constrained bounding box

Outputs:
    candidate_results.csv
    candidate_summary.csv
    selected_roi_config.json
    selected_case_results.csv

No network retraining.
No outer-test cases are accessed.

Usage:
    python -u 6_auto_roi/auto_roi_5fold_v4.py audit --fold 1
"""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import (
    label,
    find_objects,
    gaussian_filter1d,
)

ROOT = Path.home() / "AAA_CTA_Project"

V1_FILE = ROOT / "6_auto_roi/auto_roi_5fold.py"

V3_ROOT = ROOT / "9_results/auto_roi_5fold_v3"

OUT = ROOT / "9_results/auto_roi_5fold_v4"

OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.20, 0.30, 0.40, 0.50]

MARGINS = [12, 24, 40]

QUANTILES = [0.001, 0.005, 0.01]

SMOOTH_SIGMAS = [1.0, 2.0]

TARGET_COVERAGE = 0.98

MAX_ROI_FRACTION = 0.65


# ============================================================
# Import existing V1 preprocessing
# ============================================================

spec = importlib.util.spec_from_file_location(
    "aaa_v1",
    str(V1_FILE)
)

v1 = importlib.util.module_from_spec(spec)

spec.loader.exec_module(v1)


# ============================================================
# General utilities
# ============================================================

def roi_fraction(shape, low, high):

    shape = np.asarray(shape, dtype=float)

    return float(
        np.prod(
            (high - low) / shape
        )
    )


def gt_coverage(gt, low, high):

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


def coarse_to_full(
    low_coarse,
    high_coarse,
    coarse_shape,
    full_shape,
    margin
):

    full_shape = np.asarray(
        full_shape,
        dtype=int
    )

    coarse_shape = np.asarray(
        coarse_shape,
        dtype=float
    )

    low = np.floor(
        np.asarray(low_coarse)
        * full_shape
        / coarse_shape
    ).astype(int)

    high = np.ceil(
        np.asarray(high_coarse)
        * full_shape
        / coarse_shape
    ).astype(int)

    low = np.maximum(
        low - margin,
        0
    )

    high = np.minimum(
        high + margin,
        full_shape
    )

    if np.any(high <= low):
        raise RuntimeError("Invalid ROI")

    return low, high


def full_roi(shape):

    return (
        np.zeros(3, dtype=int),
        np.asarray(shape, dtype=int)
    )


# ============================================================
# Method 1: Largest connected component
# ============================================================

def roi_largest(
    probability,
    full_shape,
    threshold,
    margin
):

    mask = probability >= threshold

    components, n = label(
        mask,
        structure=np.ones(
            (3, 3, 3),
            dtype=np.uint8
        )
    )

    if n == 0:
        return full_roi(full_shape)

    sizes = np.bincount(
        components.ravel()
    )

    sizes[0] = 0

    largest_id = int(
        np.argmax(sizes)
    )

    objects = find_objects(components)

    sl = objects[largest_id - 1]

    low_coarse = [
        s.start for s in sl
    ]

    high_coarse = [
        s.stop for s in sl
    ]

    return coarse_to_full(
        low_coarse,
        high_coarse,
        probability.shape,
        full_shape,
        margin
    )


# ============================================================
# Method 2: Probability-weighted quantile ROI
# ============================================================

def weighted_quantile(
    values,
    weights,
    q
):

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)

    total = cumulative[-1]

    if total <= 0:
        return float(np.median(values))

    return float(
        np.interp(
            q * total,
            cumulative,
            values
        )
    )


def roi_weighted_quantile(
    probability,
    full_shape,
    threshold,
    margin,
    quantile
):

    mask = probability >= threshold

    coords = np.argwhere(mask)

    if len(coords) == 0:
        return full_roi(full_shape)

    weights = probability[
        mask
    ].astype(np.float64)

    low_coarse = []
    high_coarse = []

    for axis in range(3):

        values = coords[:, axis].astype(
            np.float64
        )

        lo = weighted_quantile(
            values,
            weights,
            quantile
        )

        hi = weighted_quantile(
            values,
            weights,
            1.0 - quantile
        )

        low_coarse.append(
            int(np.floor(lo))
        )

        high_coarse.append(
            int(np.ceil(hi)) + 1
        )

    return coarse_to_full(
        low_coarse,
        high_coarse,
        probability.shape,
        full_shape,
        margin
    )


# ============================================================
# Method 3: Slice-wise continuity constrained ROI
# ============================================================

def roi_slice_continuity(
    probability,
    full_shape,
    threshold,
    margin,
    sigma
):

    mask = probability >= threshold

    if not mask.any():
        return full_roi(full_shape)

    # Infer the long anatomical axis from the
    # distribution of positive voxels.

    coords = np.argwhere(mask)

    spans = (
        coords.max(axis=0)
        - coords.min(axis=0)
    )

    long_axis = int(
        np.argmax(spans)
    )

    other_axes = [
        a for a in range(3)
        if a != long_axis
    ]

    # Positive mass per slice along the long axis.

    mass = mask.sum(
        axis=tuple(other_axes)
    ).astype(np.float64)

    smooth_mass = gaussian_filter1d(
        mass,
        sigma=sigma
    )

    active = smooth_mass > max(
        1.0,
        0.05 * smooth_mass.max()
    )

    if not active.any():
        return full_roi(full_shape)

    # Keep the longest continuous active interval.

    intervals, n = label(
        active.astype(np.uint8)
    )

    sizes = np.bincount(
        intervals
    )

    sizes[0] = 0

    selected = (
        intervals == np.argmax(sizes)
    )

    indices = np.flatnonzero(selected)

    start = int(indices.min())
    stop = int(indices.max()) + 1

    # Restrict mask to selected slice interval.

    constrained = np.zeros_like(
        mask,
        dtype=bool
    )

    sl = [slice(None)] * 3

    sl[long_axis] = slice(
        start,
        stop
    )

    constrained[tuple(sl)] = (
        mask[tuple(sl)]
    )

    if not constrained.any():
        return full_roi(full_shape)

    # Compute transverse bounds from robust quantiles
    # to reduce isolated far-away false positives.

    coords = np.argwhere(constrained)

    low_coarse = coords.min(axis=0)
    high_coarse = coords.max(axis=0) + 1

    for axis in other_axes:

        low_coarse[axis] = int(
            np.floor(
                np.quantile(
                    coords[:, axis],
                    0.005
                )
            )
        )

        high_coarse[axis] = int(
            np.ceil(
                np.quantile(
                    coords[:, axis],
                    0.995
                )
            )
        ) + 1

    return coarse_to_full(
        low_coarse,
        high_coarse,
        probability.shape,
        full_shape,
        margin
    )


# ============================================================
# Generate candidate parameter configurations
# ============================================================

def configurations():

    configs = []

    for threshold, margin in itertools.product(
        THRESHOLDS,
        MARGINS
    ):

        configs.append({
            "method": "largest",
            "threshold": threshold,
            "margin": margin,
            "quantile": None,
            "sigma": None
        })

        for q in QUANTILES:

            configs.append({
                "method": "weighted_quantile",
                "threshold": threshold,
                "margin": margin,
                "quantile": q,
                "sigma": None
            })

        for sigma in SMOOTH_SIGMAS:

            configs.append({
                "method": "slice_continuity",
                "threshold": threshold,
                "margin": margin,
                "quantile": None,
                "sigma": sigma
            })

    return configs


def predict_roi(
    probability,
    full_shape,
    config
):

    method = config["method"]

    threshold = float(
        config["threshold"]
    )

    margin = int(
        config["margin"]
    )

    if method == "largest":

        return roi_largest(
            probability,
            full_shape,
            threshold,
            margin
        )

    if method == "weighted_quantile":

        return roi_weighted_quantile(
            probability,
            full_shape,
            threshold,
            margin,
            float(config["quantile"])
        )

    if method == "slice_continuity":

        return roi_slice_continuity(
            probability,
            full_shape,
            threshold,
            margin,
            float(config["sigma"])
        )

    raise ValueError(method)


# ============================================================
# Internal OOF audit
# ============================================================

def audit(outer_fold):

    v1.verify_cases()

    v3_folder = (
        V3_ROOT / f"fold{outer_fold}"
    )

    prob_folder = (
        v3_folder / "oof_probabilities"
    )

    metadata_file = (
        v3_folder / "oof_metadata.json"
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            metadata_file
        )

    with open(
        metadata_file,
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    expected = (
        set(v1.verify_cases())
        - set(v1.FOLDS[outer_fold])
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
            "Internal OOF metadata mismatch"
        )

    configs = configurations()

    fold_out = (
        OUT / f"fold{outer_fold}"
    )

    fold_out.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 100)
    print("V4 SPATIAL ROI OPTIMIZATION")
    print("=" * 100)

    print("Outer fold:", outer_fold)
    print("OOF cases:", len(metadata))
    print("Candidate configurations:", len(configs))
    print(
        "Excluded outer-test cases:",
        v1.FOLDS[outer_fold]
    )

    records = []

    for item in metadata:

        case = str(item["case_id"])

        print(
            "Evaluating:",
            case,
            flush=True
        )

        prob_path = (
            prob_folder / f"{case}_prob.npy"
        )

        probability = np.load(
            prob_path
        )

        # GT is used only to evaluate candidate ROIs.
        # It is never used to construct a predicted ROI.

        _, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        for config_id, config in enumerate(
            configs
        ):

            low, high = predict_roi(
                probability,
                gt.shape,
                config
            )

            coverage = gt_coverage(
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
                "config_id": config_id,
                "method": config["method"],
                "threshold": config["threshold"],
                "margin": config["margin"],
                "quantile": config["quantile"],
                "sigma": config["sigma"],
                "gt_coverage": coverage,
                "roi_fraction": fraction,
                "roi_start": str(low.tolist()),
                "roi_stop_exclusive": str(
                    high.tolist()
                )
            })

        del gt, probability

    results = pd.DataFrame(records)

    results.to_csv(
        fold_out / "candidate_results.csv",
        index=False
    )

    summary = results.groupby(
        "config_id"
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
        ),
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
        how="left",
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

    if feasible.empty:

        status = "NO_FEASIBLE_CONFIGURATION"

        # Diagnostic only. Not approved for final training.

        ranked = summary.sort_values(
            [
                "failed_coverage_cases",
                "oversized_cases",
                "min_coverage",
                "mean_roi_fraction"
            ],
            ascending=[
                True,
                True,
                False,
                True
            ]
        )

        selected = ranked.iloc[0]

    else:

        status = "FEASIBLE_CONFIGURATION_FOUND"

        selected = feasible.sort_values(
            [
                "mean_roi_fraction",
                "min_coverage"
            ],
            ascending=[
                True,
                False
            ]
        ).iloc[0]

    selected_config = configs[
        int(selected["config_id"])
    ].copy()

    selected_config.update({
        "outer_fold": int(outer_fold),
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
        )
    })

    with open(
        fold_out / "selected_roi_config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selected_config,
            f,
            indent=2,
            ensure_ascii=False
        )

    selected_cases = results[
        results["config_id"]
        == int(selected["config_id"])
    ].copy()

    selected_cases.to_csv(
        fold_out / "selected_case_results.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("V4 SELECTED CONFIGURATION")
    print("=" * 100)

    print(
        json.dumps(
            selected_config,
            indent=2
        )
    )

    print("\n12 CASES")

    print(
        selected_cases[
            [
                "case_id",
                "gt_coverage",
                "roi_fraction"
            ]
        ].sort_values(
            "gt_coverage"
        ).to_string(index=False)
    )

    print("\nFeasible configurations:", len(feasible))

    print("\nSaved:", fold_out)

    if feasible.empty:

        print(
            "\nWARNING: No configuration meets "
            "both coverage and ROI-volume constraints."
        )

        print(
            "Do not proceed to final locator training."
        )


# ============================================================
# CLI
# ============================================================

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

    if args.command == "audit":

        audit(args.fold)


if __name__ == "__main__":
    main()
