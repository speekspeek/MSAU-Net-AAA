#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "9_results" / "corrected_v5_oof"
OUTPUT = ROOT / "9_results" / "figures"

MODELS = {
    "unet3d": "3D U-Net",
    "attention_unet3d": "Attention\nU-Net",
    "msau_no_multiscale": "Without\nmultiscale",
    # Replace this display label after verifying the ablation architecture.
    "msau_no_msa": "Ablation 2",
    "msau_unet3d_v2": "Full\nMSAU-Net",
}
FULL = "msau_unet3d_v2"
GROUPS = {
    "AAA": ("#A74D17", "o"),
    "Normal": ("#21658A", "D"),
}
INK = "#26343F"
GRID = "#DFE6EB"


def load_results():
    results = {}
    reference = None

    for model in MODELS:
        path = INPUT / model / f"{model}_5fold_oof_per_case.csv"
        data = pd.read_csv(
            path, usecols=["fold", "case_id", "class", "dice", "hd95_mm"]
        )
        data["class"] = data["class"].str.strip().str.lower().map({
            "aaa": "AAA",
            "normal": "Normal",
        })
        if (
            len(data) != 15
            or data["case_id"].isna().any()
            or data["case_id"].duplicated().any()
            or data["fold"].value_counts().to_dict() != dict.fromkeys(range(1, 6), 3)
            or data["class"].value_counts().to_dict() != {"AAA": 6, "Normal": 9}
        ):
            raise ValueError(f"{path}: unexpected case, fold, or group counts")

        if (
            not np.isfinite(data[["dice", "hd95_mm"]].to_numpy()).all()
            or not data["dice"].between(0, 1).all()
            or (data["hd95_mm"] < 0).any()
        ):
            raise ValueError(f"{path}: invalid Dice or HD95 values")

        data = data.sort_values(["class", "fold", "case_id"])
        data = data.set_index(["fold", "case_id"])
        if reference is None:
            reference = data["class"]
        elif not data["class"].equals(reference):
            raise ValueError(f"{path}: case, fold, or group differs across models")
        results[model] = data

    return results


def main():
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError:
        raise SystemExit("Times New Roman is not installed on the SSH server.")
    results = load_results()
    full = results[FULL]
    groups = full["class"].to_numpy()
    case_ids = full.index.get_level_values("case_id")
    jitter = np.random.default_rng(18).uniform(-0.18, 0.18, len(full))
    labels = list(MODELS.values())

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    })

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, width_ratios=[1.14, 1], height_ratios=[1, 1.13])
    ax_a = fig.add_subplot(layout[0, 0])
    ax_b = fig.add_subplot(layout[0, 1])
    ax_c = fig.add_subplot(layout[1, :])

    # A. Case-level Dice and between-case SD.
    for x, model in enumerate(MODELS):
        dice = results[model]["dice"].to_numpy()
        for group, (color, marker) in GROUPS.items():
            mask = groups == group
            ax_a.scatter(x + jitter[mask], dice[mask], s=24, marker=marker,
                         color=color, edgecolors="white", linewidths=0.45, zorder=3)
        ax_a.errorbar(x + 0.27, dice.mean(), yerr=dice.std(ddof=1),
                      fmt="_", color=INK, linewidth=1, capsize=2.5, zorder=4)

    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=14)
    ax_a.set_xticks(range(len(labels)), labels, fontsize=14)
    ax_a.set_xlim(-0.4, len(labels) - 0.5)
    ax_a.set_ylabel("Dice")
    ax_a.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_a.grid(axis="y", color=GRID, linewidth=0.7)
    handles = [
        Line2D([], [], marker=marker, color=color, linestyle="none", label=group)
        for group, (color, marker) in GROUPS.items()
    ]
    handles.append(Line2D([], [], color=INK, marker="_", label="Mean ± SD"))
    ax_a.legend(handles=handles, loc="lower right", ncol=3, fontsize=14)
    low, high = ax_a.get_ylim()
    ax_a.set_ylim(low - 0.22 * (high - low), high)

    # B. Differences paired by case and fold.
    for y, baseline in enumerate(("unet3d", "attention_unet3d")):
        delta = (full["dice"] - results[baseline]["dice"]).to_numpy() * 100
        for group, (color, marker) in GROUPS.items():
            mask = groups == group
            ax_b.scatter(delta[mask], y + jitter[mask], s=25, marker=marker,
                         color=color, edgecolors="white", linewidths=0.45, zorder=3)
        mean = delta.mean()
        ax_b.plot(mean, y + 0.30, "|", color=INK, markersize=8, markeredgewidth=2)
        ax_b.annotate(f"Mean {mean:+.2f}", (mean, y + 0.30),
                      xytext=(5, 0), textcoords="offset points", va="center", fontsize=14)

    ax_b.axvline(0, color=INK, linestyle="--", linewidth=0.9)
    ax_b.set_yticks([0, 1], ["vs 3D U-Net", "vs Attention U-Net"], fontsize=14)
    ax_b.set_ylim(1.54, -0.4)
    ax_b.margins(x=0.16)
    ax_b.set_xlabel("Full MSAU-Net minus baseline (percentage points)")
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=14)
    ax_b.grid(axis="x", color=GRID, linewidth=0.7)

    # C. HD95 values on a common linear color scale.
    hd95 = np.vstack([results[model]["hd95_mm"].to_numpy() for model in MODELS])
    heatmap = ax_c.imshow(hd95, cmap="Blues", vmin=0, vmax=max(1, hd95.max()),
                         aspect="auto", interpolation="nearest")
    for (row, col), value in np.ndenumerate(hd95):
        ax_c.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=14,
                  color="white" if heatmap.norm(value) > 0.60 else INK)

    ax_c.axvline(np.count_nonzero(groups == "AAA") - 0.5, color="white", linewidth=2)
    ax_c.set_xticks(range(len(case_ids)), case_ids, rotation=40, ha="right", fontsize=14)
    ax_c.set_yticks(range(len(labels)), [label.replace("\n", " ") for label in labels])
    ax_c.set_xlabel("Outer-test case (6 AAA, 9 normal)")
    ax_c.set_title("C", loc="left", fontweight="bold", fontsize=14)
    ax_c.tick_params(length=0)
    fig.colorbar(heatmap, ax=ax_c, label="HD95 (mm)", fraction=0.035, pad=0.025)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        path = OUTPUT / f"Figure_2.{extension}"
        fig.savefig(path, dpi=600, facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
