#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Morphology Risk Score V2
========================

Input:
    9_results/true_centerline_diameter/
        AAA_true_centerline_diameter_v3.csv

Primary morphology variables:
    1. max_feret_diameter_mm
    2. mesh_volume_mm3
    3. tortuosity
    4. sphericity
    5. max_section_area_mm2

Risk direction:
    Larger diameter          -> higher score
    Larger volume            -> higher score
    Greater tortuosity       -> higher score
    Lower sphericity         -> higher score
    Larger cross-section     -> higher score

The resulting score is an exploratory morphology-based index,
NOT a clinically validated rupture-risk probability.

Output:
    9_results/morphology_risk_v2/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    percentileofscore,
    spearmanr
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project"
)

INPUT = (
    ROOT /
    "9_results/true_centerline_diameter/"
    "AAA_true_centerline_diameter_v3.csv"
)

OUT_DIR = (
    ROOT /
    "9_results/morphology_risk_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Feature definition
# ============================================================

FEATURES = {

    "max_feret_diameter_mm": {
        "label":
            "Maximum Feret diameter",
        "weight":
            0.35,
        "direction":
            1
    },

    "mesh_volume_mm3": {
        "label":
            "AAA volume",
        "weight":
            0.25,
        "direction":
            1
    },

    "tortuosity": {
        "label":
            "Tortuosity",
        "weight":
            0.15,
        "direction":
            1
    },

    "sphericity": {
        "label":
            "Sphericity",
        "weight":
            0.15,
        # lower sphericity -> larger morphology risk contribution
        "direction":
            -1
    },

    "max_section_area_mm2": {
        "label":
            "Maximum cross-sectional area",
        "weight":
            0.10,
        "direction":
            1
    },
}


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    INPUT
)


print("=" * 90)
print("MORPHOLOGY RISK SCORE V2")
print("=" * 90)

print(
    "Input:",
    INPUT
)

print(
    "Cases:",
    len(df)
)


# ============================================================
# Validate input
# ============================================================

required = (
    ["case_id"]
    +
    list(
        FEATURES.keys()
    )
)


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


if len(df) != 100:

    print(
        f"[WARNING] Expected 100 cases, "
        f"but found {len(df)}"
    )


# ============================================================
# Keep valid rows only
# ============================================================

for feature in FEATURES:

    df[
        feature
    ] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


missing_count = (
    df[
        list(
            FEATURES.keys()
        )
    ]
    .isna()
    .sum()
)


print("\nMissing values:")
print(
    missing_count
)


if (
    missing_count.sum()
    >
    0
):

    raise ValueError(
        "Missing morphology values detected. "
        "Do not construct risk score before QC."
    )


# ============================================================
# Winsorization
# ============================================================
# Robust against isolated extreme morphology values.
# Original raw variables are preserved.
# ============================================================

def winsorize_series(
        x,
        low=0.01,
        high=0.99
):

    lo = x.quantile(
        low
    )

    hi = x.quantile(
        high
    )

    return x.clip(
        lower=lo,
        upper=hi
    )


for feature in FEATURES:

    df[
        feature +
        "_winsor"
    ] = winsorize_series(
        df[
            feature
        ]
    )


# ============================================================
# Percentile transformation
# ============================================================
# Percentile score:
# 0 = lowest morphology contribution
# 100 = highest morphology contribution
#
# This avoids assuming normally distributed morphology.
# ============================================================

def percentile_rank(
        series
):

    return (
        series.rank(
            method="average",
            pct=True
        )
        *
        100.0
    )


contribution_columns = []


for feature, config in FEATURES.items():

    x = df[
        feature +
        "_winsor"
    ]


    percentile = percentile_rank(
        x
    )


    # Reverse protective direction
    if (
        config[
            "direction"
        ]
        ==
        -1
    ):

        percentile = (
            100.0
            -
            percentile
        )


    percentile_col = (
        feature
        +
        "_risk_percentile"
    )


    contribution_col = (
        feature
        +
        "_weighted_contribution"
    )


    df[
        percentile_col
    ] = percentile


    df[
        contribution_col
    ] = (
        percentile
        *
        config[
            "weight"
        ]
    )


    contribution_columns.append(
        contribution_col
    )


# ============================================================
# Composite score
# ============================================================

df[
    "morphology_risk_score_v2"
] = (
    df[
        contribution_columns
    ]
    .sum(
        axis=1
    )
)


# ============================================================
# Normalize exactly to 0–100
# ============================================================

score = df[
    "morphology_risk_score_v2"
]


score_min = score.min()
score_max = score.max()


if (
    score_max
    >
    score_min
):

    df[
        "morphology_risk_score_v2"
    ] = (
        (
            score
            -
            score_min
        )
        /
        (
            score_max
            -
            score_min
        )
        *
        100.0
    )


# ============================================================
# Cohort-relative risk strata
# ============================================================
# IMPORTANT:
# These are cohort-relative morphology strata,
# NOT validated clinical rupture-risk thresholds.
# ============================================================

q33 = df[
    "morphology_risk_score_v2"
].quantile(
    1/3
)

q67 = df[
    "morphology_risk_score_v2"
].quantile(
    2/3
)


def assign_risk_group(
        score
):

    if score < q33:

        return "Low"

    elif score < q67:

        return "Intermediate"

    else:

        return "High"


df[
    "morphology_risk_group_v2"
] = (
    df[
        "morphology_risk_score_v2"
    ]
    .apply(
        assign_risk_group
    )
)


# ============================================================
# Diameter size category
# ============================================================
# Kept separate from the composite morphology risk score.
#
# These bands are descriptive size categories only.
# ============================================================

def diameter_size_band(
        diameter
):

    if diameter < 40:

        return "<40 mm"

    elif diameter < 55:

        return "40–54.9 mm"

    else:

        return ">=55 mm"


df[
    "diameter_size_band"
] = (
    df[
        "max_feret_diameter_mm"
    ]
    .apply(
        diameter_size_band
    )
)


# ============================================================
# Rank cases
# ============================================================

df[
    "morphology_risk_rank"
] = (
    df[
        "morphology_risk_score_v2"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ============================================================
# Sort by score
# ============================================================

df = df.sort_values(
    "morphology_risk_score_v2",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# Save complete dataset
# ============================================================

output_file = (
    OUT_DIR /
    "AAA100_morphology_risk_score_v2.csv"
)


df.to_csv(
    output_file,
    index=False
)


# ============================================================
# Compact paper-ready table
# ============================================================

paper_columns = [

    "case_id",

    "max_feret_diameter_mm",

    "max_section_area_mm2",

    "mesh_volume_mm3",

    "tortuosity",

    "sphericity",

    "centerline_length_mm",

    "morphology_risk_score_v2",

    "morphology_risk_group_v2",

    "morphology_risk_rank",

    "diameter_size_band"
]


paper_columns = [
    x
    for x in paper_columns
    if x in df.columns
]


paper_df = df[
    paper_columns
].copy()


paper_file = (
    OUT_DIR /
    "AAA100_morphology_risk_table_v2.csv"
)


paper_df.to_csv(
    paper_file,
    index=False
)


# ============================================================
# Feature weights
# ============================================================

weight_rows = []


for feature, config in FEATURES.items():

    weight_rows.append(
        {
            "feature":
                feature,

            "label":
                config[
                    "label"
                ],

            "weight":
                config[
                    "weight"
                ],

            "risk_direction":
                (
                    "Higher = higher score"
                    if config[
                        "direction"
                    ] == 1
                    else
                    "Lower = higher score"
                )
        }
    )


weight_df = pd.DataFrame(
    weight_rows
)


weight_file = (
    OUT_DIR /
    "morphology_risk_v2_weights.csv"
)


weight_df.to_csv(
    weight_file,
    index=False
)


# ============================================================
# Group summaries
# ============================================================

summary_features = [

    "morphology_risk_score_v2",

    "max_feret_diameter_mm",

    "max_section_area_mm2",

    "mesh_volume_mm3",

    "tortuosity",

    "sphericity"
]


group_summary = (
    df
    .groupby(
        "morphology_risk_group_v2"
    )[
        summary_features
    ]
    .agg(
        [
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max"
        ]
    )
)


group_summary_file = (
    OUT_DIR /
    "morphology_risk_v2_group_summary.csv"
)


group_summary.to_csv(
    group_summary_file
)


# ============================================================
# Diameter band summary
# ============================================================

diameter_order = [
    "<40 mm",
    "40–54.9 mm",
    ">=55 mm"
]


diameter_summary = (
    df[
        "diameter_size_band"
    ]
    .value_counts()
    .reindex(
        diameter_order,
        fill_value=0
    )
)


diameter_summary_file = (
    OUT_DIR /
    "diameter_size_band_distribution.csv"
)


diameter_summary.rename(
    "count"
).to_csv(
    diameter_summary_file
)


# ============================================================
# Correlation of score with morphology features
# ============================================================

correlation_rows = []


for feature, config in FEATURES.items():

    rho, p = spearmanr(

        df[
            feature
        ],

        df[
            "morphology_risk_score_v2"
        ]
    )


    correlation_rows.append(
        {
            "feature":
                feature,

            "label":
                config[
                    "label"
                ],

            "spearman_rho":
                rho,

            "p_value":
                p
        }
    )


correlation_df = pd.DataFrame(
    correlation_rows
)


correlation_file = (
    OUT_DIR /
    "morphology_risk_v2_correlations.csv"
)


correlation_df.to_csv(
    correlation_file,
    index=False
)


# ============================================================
# Top / bottom cases
# ============================================================

top10 = df[
    paper_columns
].head(
    10
)


bottom10 = df[
    paper_columns
].tail(
    10
)


top10.to_csv(
    OUT_DIR /
    "morphology_risk_v2_top10.csv",
    index=False
)


bottom10.to_csv(
    OUT_DIR /
    "morphology_risk_v2_bottom10.csv",
    index=False
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 90)

print(
    "MORPHOLOGY RISK SCORE V2 COMPLETED"
)

print(
    "=" * 90
)


print(
    "\nScore cut-offs "
    "(cohort-relative tertiles):"
)

print(
    f"Low < {q33:.2f}"
)

print(
    f"Intermediate: "
    f"{q33:.2f}–{q67:.2f}"
)

print(
    f"High >= {q67:.2f}"
)


print(
    "\nRisk-group distribution:"
)

print(
    df[
        "morphology_risk_group_v2"
    ]
    .value_counts()
    .reindex(
        [
            "Low",
            "Intermediate",
            "High"
        ]
    )
)


print(
    "\nDiameter-size distribution:"
)

print(
    diameter_summary
)


print(
    "\nTop 10 morphology-risk cases:"
)

print(
    top10[
        [
            "case_id",
            "max_feret_diameter_mm",
            "mesh_volume_mm3",
            "tortuosity",
            "sphericity",
            "morphology_risk_score_v2",
            "morphology_risk_group_v2"
        ]
    ]
    .round(
        3
    )
    .to_string(
        index=False
    )
)


print(
    "\nFeature-score correlations:"
)

print(
    correlation_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    output_file
)

print(
    paper_file
)

print(
    weight_file
)

print(
    group_summary_file
)

print(
    correlation_file
)

print(
    diameter_summary_file
)

