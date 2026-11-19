from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DEFAULT_CSV = (
    ROOT
    / "9_results/corrected_v5_oof"
    / "FINAL_5MODEL_RESULTS"
    / "final_5model_per_case.csv"
)

MODELS = [
    "3D U-Net",
    "Attention U-Net",
    "MSAU-Net w/o Multi-Scale",
    "MSAU-Net w/o Attention",
    "Full MSAU-Net",
]

LABELS = [
    "3D U-Net",
    "Attention\nU-Net",
    "MSAU-Net\nw/o Multi-Scale",
    "MSAU-Net\nw/o Attention",
    "Full\nMSAU-Net",
]

COLORS = [
    "#6588AA",
    "#609D9A",
    "#B299C6",
    "#D6A15D",
    "#CB6676",
]

ALIASES = {
    "unet3d": MODELS[0],
    "attention_unet3d": MODELS[1],
    "msau_no_multiscale": MODELS[2],
    "msau_no_msa": MODELS[3],
    "MSA-only": MODELS[3],
    "msau_unet3d_v2": MODELS[4],
    "MSAU-Net": MODELS[4],
}


def configure_font(explicit=None):
    candidates = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.extend((HERE / "fonts").glob("*.ttf"))
    candidates.extend((HERE / "fonts").glob("*.TTF"))

    candidates.extend([
        Path.home() / ".local/share/fonts/times.ttf",
        Path.home() / ".local/share/fonts/timesbd.ttf",
        Path(
            "/usr/share/fonts/truetype/"
            "msttcorefonts/Times_New_Roman.ttf"
        ),
    ])

    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))

    try:
        font_path = font_manager.findfont(
            "Times New Roman",
            fallback_to_default=False,
        )
    except ValueError:
        raise ValueError(
            "未找到Times New Roman字体。\n"
            "请将Windows中的times.ttf上传到：\n"
            f"{HERE / 'fonts'}"
        )

    actual_name = font_manager.FontProperties(
        fname=font_path
    ).get_name()

    if actual_name != "Times New Roman":
        raise ValueError(
            f"字体不是Times New Roman：{actual_name}"
        )

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })

    return font_path


def load_data(path):
    if not path.is_file():
        raise ValueError(
            f"未找到逐病例CSV：\n{path}\n"
            "可通过 --csv 指定最终逐病例结果表。"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    required = {"model", "case_id", "dice", "hd95_mm"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"缺少列：{sorted(required - set(df.columns))}\n"
            f"实际列名：{df.columns.tolist()}"
        )

    df["model"] = (
        df["model"].astype(str).str.strip().replace(ALIASES)
    )
    df["case_id"] = df["case_id"].astype(str).str.strip()

    if set(df["model"]) != set(MODELS):
        raise ValueError(
            "模型名称不符合预期：\n"
            f"{sorted(df['model'].unique())}"
        )

    if df.duplicated(["model", "case_id"]).any():
        raise ValueError("同一模型存在重复病例记录。")

    cases = sorted(
        df.loc[df["model"] == MODELS[-1], "case_id"].tolist()
    )

    if len(cases) != 15:
        raise ValueError(
            f"需要15例独立病例，当前为{len(cases)}例。"
        )

    n_anev = sum(c.startswith("Anev_") for c in cases)
    n_nat = sum(c.startswith("Nat_") for c in cases)

    if (n_anev, n_nat) != (6, 9):
        raise ValueError(
            "病例构成应为Anev 6例、Nat 9例。\n"
            f"当前为Anev {n_anev}例、Nat {n_nat}例。"
        )

    for model in MODELS:
        selected = df.loc[df["model"] == model]

        if sorted(selected["case_id"]) != cases:
            raise ValueError(f"{model}的病例集合不一致。")

        if "fold" in df.columns:
            counts = selected.groupby("fold").size()

            if len(counts) != 5 or not counts.eq(3).all():
                raise ValueError(
                    f"{model}应为五折、每折3例测试病例。"
                )

    if "fold" in df.columns:
        if df.groupby("case_id")["fold"].nunique().ne(1).any():
            raise ValueError("不同模型间的测试折划分不一致。")

    for metric in ["dice", "hd95_mm"]:
        df[metric] = pd.to_numeric(
            df[metric], errors="raise"
        )

    if (
        not np.isfinite(df["dice"]).all()
        or not df["dice"].between(0, 1).all()
    ):
        raise ValueError("Dice必须为0—1之间的有限小数。")

    hd95 = df["hd95_mm"].to_numpy(dtype=float)

    if np.isnan(hd95).any() or (hd95 < 0).any():
        raise ValueError(
            "HD95存在缺失值或负值，请检查源数据。"
        )

    return df, cases


def draw_panel(ax, df, cases, metric, title):
    raw = (
        df.pivot(
            index="case_id",
            columns="model",
            values=metric,
        )
        .reindex(index=cases, columns=MODELS)
        .to_numpy(dtype=float)
    )

    values = raw.copy()

    if metric == "hd95_mm":
        values = np.log1p(values)

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError(f"{metric}没有有限值，无法绘图。")

    if metric == "dice":
        xmax = 1.035
    else:
        xmax = max(
            float(finite_values.max()) * 1.12,
            float(np.log1p(1.0)),
        )

    has_failure = not np.isfinite(values).all()
    failure_x = xmax * 1.12

    if has_failure:
        ax.axvspan(
            xmax,
            xmax * 1.20,
            color="#F4F1F1",
            zorder=0,
        )
        ax.text(
            failure_x, 4.85, "∞",
            ha="center", va="center",
        )

    rng = np.random.default_rng(2026)
    jitter = rng.uniform(-0.055, 0.055, len(cases))

    # Pair the same case across neighboring models.
    for k in range(len(cases)):
        for j in range(len(MODELS) - 1):
            pair = values[k, j:j + 2]

            if np.isfinite(pair).all():
                ax.plot(
                    pair,
                    np.array([j, j + 1]) - 0.19 + jitter[k],
                    color="#B4BBC1",
                    alpha=0.30,
                    linewidth=0.65,
                    zorder=1,
                )

    summaries = []

    for j, (model, color) in enumerate(zip(MODELS, COLORS)):
        column = values[:, j]
        valid = column[np.isfinite(column)]

        # Half violin: KDE on the displayed coordinate scale.
        if len(valid) > 1 and np.ptp(valid) > 1e-9:
            grid = np.linspace(
                max(0, valid.min() - 0.12 * xmax),
                min(xmax, valid.max() + 0.12 * xmax),
                240,
            )

            density = gaussian_kde(
                valid, bw_method=0.45
            )(grid)

            density = density / density.max() * 0.29
            baseline = j + 0.10

            ax.fill_between(
                grid,
                baseline,
                baseline + density,
                color=color,
                alpha=0.40,
                linewidth=0,
                zorder=2,
            )

            ax.plot(
                grid,
                baseline + density,
                color=color,
                linewidth=1.1,
                zorder=3,
            )

        if len(valid):
            ax.boxplot(
                [valid],
                positions=[j + 0.015],
                vert=False,
                widths=0.12,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 1.2,
                },
                whiskerprops={
                    "color": color,
                    "linewidth": 1.1,
                },
                capprops={
                    "color": color,
                    "linewidth": 1.1,
                },
                medianprops={
                    "color": "#26333E",
                    "linewidth": 1.6,
                },
            )

        # Display every case, including distance failures.
        for k, case in enumerate(cases):
            is_finite = np.isfinite(column[k])
            x = column[k] if is_finite else failure_x

            if is_finite:
                marker = "^" if case.startswith("Anev_") else "o"
                ax.scatter(
                    x,
                    j - 0.19 + jitter[k],
                    s=34,
                    marker=marker,
                    color=color,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=4,
                )
            else:
                ax.scatter(
                    x,
                    j - 0.19 + jitter[k],
                    s=34,
                    marker="x",
                    color=color,
                    linewidths=1.1,
                    zorder=4,
                )

        original = raw[:, j]
        all_finite = np.isfinite(original).all()

        summaries.append({
            "model": model,
            "metric": metric,
            "n_total": len(original),
            "n_finite": int(np.isfinite(original).sum()),
            "n_infinite": int(np.isinf(original).sum()),
            "mean_all_cases": (
                float(np.mean(original))
                if all_finite else np.inf
            ),
            "sample_sd_all_cases": (
                float(np.std(original, ddof=1))
                if all_finite else np.nan
            ),
            "median_all_cases": float(np.median(original)),
        })

    ax.set_yticks(range(len(MODELS)), LABELS)
    ax.set_ylim(-0.55, 5.05)
    ax.set_xlim(0, xmax * 1.20 if has_failure else xmax)
    ax.tick_params(axis="y", length=0, pad=10)

    if metric == "dice":
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Dice ↑", labelpad=10)

    else:
        ticks = np.array([
            0, 0.5, 1, 2, 5, 10,
            20, 50, 100, 200, 500, 1000,
        ])

        ticks = ticks[np.log1p(ticks) <= xmax]

        if len(ticks) > 8:
            ticks = ticks[::2]

        ax.set_xticks(
            np.log1p(ticks),
            [f"{value:g}" for value in ticks],
        )

        ax.set_xlabel(
            "HD95 (mm; log1p scale) ↓",
            labelpad=10,
        )

    ax.text(
        0, 1.025, title,
        transform=ax.transAxes,
        ha="left", va="bottom",
    )

    ax.grid(
        axis="x",
        color="#E5E9ED",
        linewidth=0.65,
    )
    ax.set_axisbelow(True)

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color("#9BA6B0")

    return summaries, has_failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--font", default=None)
    parser.add_argument(
        "--outdir", type=Path, default=HERE / "outputs"
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    df, cases = load_data(csv_path)
    font_path = configure_font(args.font)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8.2)
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.20,
        top=0.93,
        wspace=0.62,
    )

    summaries = []
    any_failure = False

    panels = [
        ("dice", "(a) Volumetric overlap"),
        ("hd95_mm", "(b) Surface discrepancy"),
    ]

    for ax, (metric, title) in zip(axes, panels):
        result, failed = draw_panel(
            ax, df, cases, metric, title
        )
        summaries.extend(result)
        any_failure |= failed

    handles = [
        Line2D(
            [], [], marker="^", linestyle="",
            color="#64717D", markersize=7,
            label="Anev (n = 6)",
        ),
        Line2D(
            [], [], marker="o", linestyle="",
            color="#64717D", markersize=7,
            label="Nat (n = 9)",
        ),
        Line2D(
            [], [], color="#AEB6BE", linewidth=1,
            label="Same case across models",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.075),
        ncol=3,
        frameon=False,
    )

    fig.text(
        0.18,
        0.065,
        "Points: individual cases; boxes: median and IQR; "
        "whiskers: 1.5 × IQR; shaded curves: KDE.",
        ha="left",
        va="top",
        fontsize=14,
    )

    if any_failure:
        fig.text(
            0.18,
            0.025,
            "∞ band: distance failures; "
            "KDE and boxes use finite distances only.",
            fontsize=14,
        )

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base = outdir / "Figure2_paired_raincloud"

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

    pd.DataFrame(summaries).to_csv(
        outdir / "Figure2_summary.csv",
        index=False,
    )

    df.to_csv(
        outdir / "Figure2_source_data.csv",
        index=False,
    )

    plt.close(fig)

    print("=" * 65)
    print("Figure 2 completed")
    print(f"Input : {csv_path}")
    print(f"Font  : {font_path}")
    print("Style : Times New Roman, 14 pt")
    print("PNG   : 600 dpi, 8400 × 4920 pixels")
    print(f"Output: {outdir}")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
