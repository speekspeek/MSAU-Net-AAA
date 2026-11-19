from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

INFILE = (
    ROOT /
    "9_results/corrected_v5_oof/"
    "FINAL_5MODEL_RESULTS/"
    "final_5model_per_case.csv"
)

OUTDIR = (
    ROOT /
    "9_results/corrected_v5_oof/"
    "FINAL_5MODEL_RESULTS"
)

FULL = "Full MSAU-Net"

COMPARATORS = [
    "3D U-Net",
    "Attention U-Net",
    "MSAU-Net w/o Multi-Scale",
    "MSA-only",
]

METRICS = {
    "dice": True,
    "iou": True,
    "precision": True,
    "recall": True,
    "hd95_mm": False,
    "assd_mm": False,
}

N_BOOT = 10000
SEED = 2026


def bootstrap_mean_ci(diff, n_boot=10000, seed=2026):
    """
    Percentile bootstrap 95% CI for paired mean difference.
    """
    diff = np.asarray(diff, dtype=float)

    rng = np.random.default_rng(seed)

    n = len(diff)

    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        boot[i] = diff[idx].mean()

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5]
    )

    return float(lo), float(hi)


df = pd.read_csv(INFILE)

required = {
    "model",
    "case_id",
    *METRICS.keys()
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


# ---------------------------------------------------------
# Ensure every model has the same 15 independent cases
# ---------------------------------------------------------

full_df = (
    df[df["model"] == FULL]
    .copy()
    .sort_values("case_id")
    .reset_index(drop=True)
)

if len(full_df) != 15:
    raise RuntimeError(
        f"{FULL}: expected 15 cases, got {len(full_df)}"
    )

if full_df["case_id"].nunique() != 15:
    raise RuntimeError(
        f"{FULL}: duplicate case IDs"
    )

reference_cases = full_df["case_id"].tolist()


rows = []

print("=" * 120)
print("CORRECTED V5 — PAIRED STATISTICAL COMPARISON")
print("Reference model:", FULL)
print("Independent OOF cases:", len(reference_cases))
print("Test: two-sided Wilcoxon signed-rank test")
print("CI: paired case-level bootstrap mean-difference 95% CI")
print("=" * 120)


for comparator in COMPARATORS:

    comp_df = (
        df[df["model"] == comparator]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    if len(comp_df) != 15:
        raise RuntimeError(
            f"{comparator}: expected 15 cases, got {len(comp_df)}"
        )

    if comp_df["case_id"].tolist() != reference_cases:
        raise RuntimeError(
            f"Case mismatch: {FULL} vs {comparator}"
        )

    print()
    print("-" * 120)
    print(f"{FULL} vs {comparator}")
    print("-" * 120)

    for metric, higher_is_better in METRICS.items():

        full_values = full_df[metric].to_numpy(dtype=float)
        comp_values = comp_df[metric].to_numpy(dtype=float)

        # Always define raw difference as Full - Comparator.
        raw_diff = full_values - comp_values

        # Improvement-oriented difference:
        # overlap metrics: Full - comparator
        # distance metrics: comparator - Full
        if higher_is_better:
            improvement = raw_diff
        else:
            improvement = -raw_diff

        try:
            stat, p = wilcoxon(
                full_values,
                comp_values,
                alternative="two-sided",
                zero_method="wilcox"
            )
        except ValueError:
            stat = np.nan
            p = 1.0

        ci_low, ci_high = bootstrap_mean_ci(
            improvement,
            n_boot=N_BOOT,
            seed=SEED
        )

        mean_improvement = improvement.mean()
        median_improvement = np.median(improvement)

        better_cases = int(
            np.sum(improvement > 0)
        )

        worse_cases = int(
            np.sum(improvement < 0)
        )

        tied_cases = int(
            np.sum(np.isclose(improvement, 0))
        )

        rows.append({
            "reference_model": FULL,
            "comparator": comparator,
            "metric": metric,
            "n": len(improvement),
            "reference_mean": full_values.mean(),
            "comparator_mean": comp_values.mean(),
            "mean_improvement": mean_improvement,
            "median_improvement": median_improvement,
            "bootstrap_95ci_low": ci_low,
            "bootstrap_95ci_high": ci_high,
            "wilcoxon_W": stat,
            "p_value": p,
            "better_cases": better_cases,
            "worse_cases": worse_cases,
            "tied_cases": tied_cases,
        })

        print(
            f"{metric:10s} | "
            f"Full={full_values.mean():.6f} | "
            f"Comp={comp_values.mean():.6f} | "
            f"Improvement={mean_improvement:+.6f} | "
            f"95%CI [{ci_low:+.6f}, {ci_high:+.6f}] | "
            f"W={stat:.3f} | "
            f"P={p:.6f} | "
            f"Better/Worse={better_cases}/{worse_cases}"
        )


results = pd.DataFrame(rows)

outfile = (
    OUTDIR /
    "full_msau_vs_all_paired_statistics.csv"
)

results.to_csv(
    outfile,
    index=False
)

print()
print("=" * 120)
print("STATISTICAL TEST FINISHED")
print("=" * 120)
print("Saved:")
print(outfile)
