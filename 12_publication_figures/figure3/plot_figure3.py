from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FIGURE2_DIR = (
    ROOT
    / "12_publication_figures"
    / "figure2"
)

if not (FIGURE2_DIR / "plot_figure2.py").is_file():
    raise SystemExit(
        "未找到图2脚本：\n"
        f"{FIGURE2_DIR / 'plot_figure2.py'}"
    )

# Reuse the same font settings and 15-case validation.
sys.path.insert(0, str(FIGURE2_DIR))

from plot_figure2 import (
    configure_font,
    load_data,
    DEFAULT_CSV,
    MODELS,
)


FULL_MODEL = "Full MSAU-Net"

COMPARATORS = [
    "3D U-Net",
    "Attention U-Net",
    "MSAU-Net w/o Multi-Scale",
    "MSAU-Net w/o Attention",
]

DISPLAY_NAMES = [
    "3D U-Net",
    "Attention U-Net",
    "MSAU-Net w/o Multi-Scale",
    "MSAU-Net w/o Attention",
]

COLORS = [
    "#6588AA",
    "#609D9A",
    "#B299C6",
    "#D6A15D",
]

METRICS = [
    ("dice", "Dice", True),
    ("iou", "IoU", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("hd95_mm", "HD95 (mm)", False),
    ("assd_mm", "ASSD (mm)", False),
]

N_BOOT = 10000
SEED = 2026


# ============================================================
# Paired statistics
# ============================================================

def paired_statistics(df, cases):
    records = []

    # One common set of case-resampling indices preserves pairing
    # and provides reproducible estimates across comparisons.
    rng = np.random.default_rng(SEED)

    bootstrap_indices = rng.integers(
        0,
        len(cases),
        size=(N_BOOT, len(cases)),
    )

    for metric, title, higher_is_better in METRICS:
        if metric not in df.columns:
            raise ValueError(
                f"CSV缺少指标列：{metric}"
            )

        df[metric] = pd.to_numeric(
            df[metric],
            errors="raise",
        )

        table = df.pivot(
            index="case_id",
            columns="model",
            values=metric,
        ).reindex(
            index=cases,
            columns=MODELS,
        )

        values = table.to_numpy(dtype=float)

        # Never silently remove failed cases.
        if not np.isfinite(values).all():
            bad = table.stack(dropna=False)
            bad = bad[
                ~np.isfinite(
                    bad.to_numpy(dtype=float)
                )
            ]

            raise ValueError(
                f"{metric}存在非有限值，不能直接计算"
                "15例完整配对均值置信区间。\n"
                f"{bad.to_string()}\n"
                "已停止，未自动删除失败病例。"
            )

        full = table[
            FULL_MODEL
        ].to_numpy(dtype=float)

        for comparator in COMPARATORS:
            baseline = table[
                comparator
            ].to_numpy(dtype=float)

            # Positive values always favor full MSAU-Net.
            if higher_is_better:
                differences = full - baseline
            else:
                differences = baseline - full

            mean_difference = float(
                differences.mean()
            )

            bootstrap_means = differences[
                bootstrap_indices
            ].mean(axis=1)

            ci_low, ci_high = np.percentile(
                bootstrap_means,
                [2.5, 97.5],
            )

            if np.all(differences == 0):
                statistic = 0.0
                p_value = 1.0

            else:
                test = wilcoxon(
                    full,
                    baseline,
                    alternative="two-sided",
                    zero_method="wilcox",
                )

                statistic = float(
                    test.statistic
                )
                p_value = float(
                    test.pvalue
                )

            records.append({
                "metric": metric,
                "metric_label": title,
                "comparator": comparator,
                "n_pairs": len(cases),
                "full_model_mean": float(
                    full.mean()
                ),
                "comparator_mean": float(
                    baseline.mean()
                ),
                "mean_improvement": mean_difference,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "wilcoxon_W": statistic,
                "p_raw": p_value,
                "bootstrap_resamples": N_BOOT,
                "seed": SEED,
                "difference_definition": (
                    "Full minus comparator"
                    if higher_is_better
                    else "Comparator minus Full"
                ),
            })

    return pd.DataFrame(records)


# ============================================================
# Forest plot
# ============================================================

def draw_panel(
    ax,
    results,
    metric,
    title,
    letter,
    higher_is_better,
):
    panel = (
        results.loc[
            results["metric"] == metric
        ]
        .set_index("comparator")
        .reindex(COMPARATORS)
    )

    low = panel[
        "ci95_low"
    ].to_numpy(dtype=float)

    high = panel[
        "ci95_high"
    ].to_numpy(dtype=float)

    estimate = panel[
        "mean_improvement"
    ].to_numpy(dtype=float)

    # Include zero with padding on both sides.
    left = min(0.0, float(low.min()))
    right = max(0.0, float(high.max()))

    width = right - left

    if width < 1e-10:
        width = (
            0.02
            if higher_is_better
            else 1.0
        )

    xmin = left - 0.16 * width
    xmax = right + 0.16 * width

    # Alternating background bands.
    for row in range(len(COMPARATORS)):
        if row % 2 == 0:
            ax.axhspan(
                row - 0.40,
                row + 0.40,
                color="#F4F7FA",
                zorder=0,
            )

    ax.axvline(
        0,
        color="#697782",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        zorder=1,
    )

    for row, color in enumerate(COLORS):
        ax.plot(
            [low[row], high[row]],
            [row, row],
            color=color,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=3,
        )

        ax.plot(
            [low[row], low[row]],
            [row - 0.09, row + 0.09],
            color=color,
            linewidth=1.4,
            zorder=3,
        )

        ax.plot(
            [high[row], high[row]],
            [row - 0.09, row + 0.09],
            color=color,
            linewidth=1.4,
            zorder=3,
        )

        ax.scatter(
            estimate[row],
            row,
            marker="D",
            s=66,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

    ax.set_yticks(
        range(len(COMPARATORS)),
        DISPLAY_NAMES,
    )

    ax.set_ylim(
        len(COMPARATORS) - 0.5,
        -0.6,
    )

    ax.set_xlim(xmin, xmax)

    ax.tick_params(
        axis="y",
        length=0,
        pad=10,
    )

    ax.tick_params(
        axis="x",
        pad=6,
    )

    ax.locator_params(
        axis="x",
        nbins=5,
    )

    ax.set_title(
        f"({letter}) {title}",
        loc="left",
        pad=13,
        fontsize=14,
    )

    ax.set_xlabel(
        (
            "Mean paired difference"
            if higher_is_better
            else "Mean paired reduction (mm)"
        ),
        labelpad=8,
    )

    ax.grid(
        axis="x",
        color="#E4E9EE",
        linewidth=0.7,
    )

    ax.set_axisbelow(True)

    for side in [
        "top",
        "right",
        "left",
    ]:
        ax.spines[side].set_visible(False)

    ax.spines[
        "bottom"
    ].set_color("#9BA6B0")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
    )

    parser.add_argument(
        "--font",
        type=str,
        default=(
            "/usr/share/fonts/truetype/"
            "msttcorefonts/times.ttf"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=HERE / "outputs",
    )

    args = parser.parse_args()

    csv_path = (
        args.csv
        .expanduser()
        .resolve()
    )

    df, cases = load_data(csv_path)

    font_path = configure_font(
        args.font
    )

    results = paired_statistics(
        df.copy(),
        cases,
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14),
    )

    fig.subplots_adjust(
        left=0.19,
        right=0.98,
        bottom=0.20,
        top=0.95,
        wspace=0.90,
        hspace=0.82,
    )

    for ax, configuration, letter in zip(
        axes.flat,
        METRICS,
        "abcdef",
    ):
        metric, title, higher_is_better = (
            configuration
        )

        draw_panel(
            ax,
            results,
            metric,
            title,
            letter,
            higher_is_better,
        )

    fig.text(
        0.19,
        0.075,
        "Positive differences favor full MSAU-Net.",
        fontsize=14,
        ha="left",
    )

    fig.text(
        0.19,
        0.048,
        "Diamonds: mean paired differences; "
        "lines: 95% percentile bootstrap CIs "
        "(10,000 resamples; 15 paired cases).",
        fontsize=14,
        ha="left",
    )

    fig.text(
        0.19,
        0.021,
        "Overlap metrics: Full − comparator. "
        "Distance metrics: comparator − Full. "
        "Differences are shown in original metric units.",
        fontsize=14,
        ha="left",
    )

    outdir = (
        args.outdir
        .expanduser()
        .resolve()
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = (
        outdir
        / "Figure3_paired_difference_forest"
    )

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".svg"),
        facecolor="white",
    )

    results.to_csv(
        outdir
        / "Figure3_paired_statistics.csv",
        index=False,
    )

    df.to_csv(
        outdir
        / "Figure3_source_data.csv",
        index=False,
    )

    plt.close(fig)

    print("=" * 70)
    print("Figure 3 completed")
    print(f"Input : {csv_path}")
    print(f"Font  : {font_path}")
    print("Style : Times New Roman, 14 pt")
    print("PNG   : 600 dpi, 9600 × 8400 pixels")
    print("Pairs : 15 independent cases")
    print("CI    : 10,000 paired bootstrap resamples")
    print("P     : raw two-sided Wilcoxon P values; exploratory")
    print(f"Output: {outdir}")
    print("=" * 70)

    print(
        results[[
            "metric",
            "comparator",
            "mean_improvement",
            "ci95_low",
            "ci95_high",
            "p_raw",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()

    except (
        ValueError,
        KeyError,
        OSError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
