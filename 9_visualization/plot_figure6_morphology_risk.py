from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

DATA_FILE = (
    ROOT /
    "9_results/AAA_morphology_dataset_v2.csv"
)

OUT_DIR = (
    ROOT /
    "9_results/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ROC_OUT = (
    ROOT /
    "9_results/Figure6_risk_roc_predictions.csv"
)

STAT_OUT = (
    ROOT /
    "9_results/Figure6_morphology_statistics.csv"
)


# ============================================================
# Publication style
# ============================================================

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Times",
    "DejaVu Serif"
]

plt.rcParams["font.size"] = 9

plt.rcParams["axes.linewidth"] = 0.8

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

df["label"] = (
    df["class"] == "AAA"
).astype(int)


FEATURES = [
    "volume_mm3",
    "diameter_mm",
    "surface_area_mm2",
    "sphericity",
    "volume_diameter_ratio"
]


FEATURE_LABELS = [
    "Volume",
    "Diameter",
    "Surface area",
    "Sphericity",
    "V/D ratio"
]


aaa = df[
    df["class"] == "AAA"
].copy()

normal = df[
    df["class"] == "Normal"
].copy()


print("AAA:", len(aaa))
print("Normal:", len(normal))


# ============================================================
# Morphology statistics
# ============================================================

stat_rows = []

for feature, label in zip(
        FEATURES,
        FEATURE_LABELS
):

    x = aaa[
        feature
    ].dropna()

    y = normal[
        feature
    ].dropna()

    u, p = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )

    stat_rows.append(
        {
            "Feature": label,
            "AAA_mean": x.mean(),
            "AAA_std": x.std(ddof=1),
            "Normal_mean": y.mean(),
            "Normal_std": y.std(ddof=1),
            "P_value": p
        }
    )


stat_df = pd.DataFrame(
    stat_rows
)

stat_df.to_csv(
    STAT_OUT,
    index=False
)


print("\nMorphology statistics:")
print(
    stat_df.to_string(
        index=False
    )
)


# ============================================================
# ROC via out-of-fold predictions
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    "label"
].copy()


models = {

    "Logistic Regression":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000
                    )
                )
            ]
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "SVM":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    SVC(
                        probability=True,
                        kernel="rbf"
                    )
                )
            ]
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            eval_metric="logloss"
        )
}


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


roc_rows = []

oof_predictions = {}


for model_name, model in models.items():

    probs = np.zeros(
        len(df),
        dtype=float
    )

    for train_idx, test_idx in skf.split(
        X,
        y
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        model.fit(
            X_train,
            y_train
        )

        probs[
            test_idx
        ] = model.predict_proba(
            X_test
        )[:, 1]


    oof_predictions[
        model_name
    ] = probs


    auc = roc_auc_score(
        y,
        probs
    )


    print(
        model_name,
        "OOF AUC =",
        round(auc, 4)
    )


for i in range(
    len(df)
):

    row = {
        "case_id":
            df.loc[
                i,
                "case_id"
            ],

        "label":
            y.iloc[i]
    }

    for model_name in models.keys():

        row[
            model_name
        ] = (
            oof_predictions[
                model_name
            ][i]
        )

    roc_rows.append(
        row
    )


pd.DataFrame(
    roc_rows
).to_csv(
    ROC_OUT,
    index=False
)


# ============================================================
# Helper functions
# ============================================================

def p_to_star(p):

    if p < 0.001:
        return "***"

    elif p < 0.01:
        return "**"

    elif p < 0.05:
        return "*"

    return "n.s."


def add_sig(
        ax,
        x1,
        x2,
        y,
        h,
        text
):

    ax.plot(
        [x1, x1, x2, x2],
        [y, y+h, y+h, y],
        color="black",
        linewidth=0.8
    )

    ax.text(
        (x1+x2)/2,
        y+h,
        text,
        ha="center",
        va="bottom",
        fontsize=8
    )


def draw_group_box(
        ax,
        feature,
        ylabel,
        panel_label
):

    data = [
        normal[
            feature
        ].values,

        aaa[
            feature
        ].values
    ]


    bp = ax.boxplot(
        data,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color="black",
            linewidth=1.1
        ),
        boxprops=dict(
            linewidth=0.9
        ),
        whiskerprops=dict(
            linewidth=0.8
        ),
        capprops=dict(
            linewidth=0.8
        )
    )


    colors = [
        "#A9CBE8",
        "#E99B8F"
    ]


    for patch, color in zip(
        bp["boxes"],
        colors
    ):

        patch.set_facecolor(
            color
        )

        patch.set_alpha(
            0.75
        )


    rng = np.random.default_rng(
        2026
    )


    for i, values in enumerate(
        data,
        start=1
    ):

        jitter = rng.normal(
            0,
            0.045,
            len(values)
        )

        ax.scatter(
            np.full(
                len(values),
                i
            ) + jitter,
            values,
            s=18,
            alpha=0.70,
            edgecolors="none"
        )


    ax.set_xticks(
        [1, 2]
    )

    ax.set_xticklabels(
        [
            "Normal",
            "AAA"
        ]
    )

    ax.set_ylabel(
        ylabel
    )


    p = stat_df.loc[
        stat_df[
            "Feature"
        ] == (
            "Diameter"
            if feature == "diameter_mm"
            else "Volume"
        ),
        "P_value"
    ].iloc[0]


    ymax = max(
        np.max(data[0]),
        np.max(data[1])
    )

    ymin = min(
        np.min(data[0]),
        np.min(data[1])
    )

    yrange = (
        ymax - ymin
    )

    add_sig(
        ax,
        1,
        2,
        ymax + 0.06 * yrange,
        0.02 * yrange,
        p_to_star(p)
    )


    ax.set_ylim(
        ymin - 0.08 * yrange,
        ymax + 0.20 * yrange
    )


    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.4,
        alpha=0.25
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    ax.text(
        -0.12,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )


# ============================================================
# Standardized morphology profile
# ============================================================

scaled = StandardScaler().fit_transform(
    df[
        FEATURES
    ]
)

scaled_df = pd.DataFrame(
    scaled,
    columns=FEATURE_LABELS
)

scaled_df[
    "class"
] = df[
    "class"
].values


profile = (
    scaled_df
    .groupby("class")[
        FEATURE_LABELS
    ]
    .mean()
)


# ============================================================
# Figure layout
# ============================================================

fig = plt.figure(
    figsize=(
        10.5,
        8.0
    ),
    dpi=300
)


gs = fig.add_gridspec(
    2,
    2,
    wspace=0.30,
    hspace=0.34
)


# ============================================================
# A: Diameter
# ============================================================

ax_a = fig.add_subplot(
    gs[0, 0]
)

draw_group_box(
    ax_a,
    "diameter_mm",
    "Maximum diameter (mm)",
    "(A)"
)


# ============================================================
# B: Volume
# ============================================================

ax_b = fig.add_subplot(
    gs[0, 1]
)

draw_group_box(
    ax_b,
    "volume_mm3",
    "Volume (mm³)",
    "(B)"
)


# ============================================================
# C: Standardized morphology profile
# ============================================================

ax_c = fig.add_subplot(
    gs[1, 0]
)


x = np.arange(
    len(FEATURE_LABELS)
)


normal_profile = profile.loc[
    "Normal"
].values

aaa_profile = profile.loc[
    "AAA"
].values


width = 0.36


ax_c.bar(
    x - width/2,
    normal_profile,
    width=width,
    label="Normal",
    alpha=0.80
)


ax_c.bar(
    x + width/2,
    aaa_profile,
    width=width,
    label="AAA",
    alpha=0.80
)


ax_c.axhline(
    0,
    color="0.4",
    linewidth=0.8
)


ax_c.set_xticks(
    x
)

ax_c.set_xticklabels(
    FEATURE_LABELS,
    rotation=25,
    ha="right",
    fontsize=8
)

ax_c.set_ylabel(
    "Standardized morphology (z-score)"
)


ax_c.legend(
    frameon=False,
    fontsize=8
)


ax_c.grid(
    axis="y",
    linestyle="--",
    linewidth=0.4,
    alpha=0.25
)


ax_c.spines[
    "top"
].set_visible(False)

ax_c.spines[
    "right"
].set_visible(False)


ax_c.text(
    -0.12,
    1.05,
    "(C)",
    transform=ax_c.transAxes,
    fontsize=11,
    fontweight="bold",
    va="top"
)


# ============================================================
# D: ROC curves
# ============================================================

ax_d = fig.add_subplot(
    gs[1, 1]
)


line_styles = {
    "Logistic Regression":
        "-",

    "Random Forest":
        "--",

    "SVM":
        "-.",

    "XGBoost":
        ":"
}


for model_name in models.keys():

    probs = oof_predictions[
        model_name
    ]

    fpr, tpr, _ = roc_curve(
        y,
        probs
    )

    auc = roc_auc_score(
        y,
        probs
    )


    ax_d.plot(
        fpr,
        tpr,
        linewidth=1.5,
        linestyle=line_styles[
            model_name
        ],
        label=(
            f"{model_name} "
            f"(AUC={auc:.2f})"
        )
    )


ax_d.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=0.8,
    color="0.5"
)


ax_d.set_xlim(
    0,
    1
)

ax_d.set_ylim(
    0,
    1.03
)


ax_d.set_xlabel(
    "False positive rate"
)

ax_d.set_ylabel(
    "True positive rate"
)


ax_d.legend(
    loc="lower right",
    frameon=False,
    fontsize=7.5
)


ax_d.grid(
    linestyle="--",
    linewidth=0.4,
    alpha=0.25
)


ax_d.spines[
    "top"
].set_visible(False)

ax_d.spines[
    "right"
].set_visible(False)


ax_d.text(
    -0.12,
    1.05,
    "(D)",
    transform=ax_d.transAxes,
    fontsize=11,
    fontweight="bold",
    va="top"
)


# ============================================================
# Save
# ============================================================

BASE = (
    OUT_DIR /
    "Figure6_morphology_risk_stratification"
)


PNG = BASE.with_suffix(
    ".png"
)

SVG = BASE.with_suffix(
    ".svg"
)

PDF = BASE.with_suffix(
    ".pdf"
)


fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.04
)


fig.savefig(
    SVG,
    format="svg",
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.04
)


fig.savefig(
    PDF,
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.04
)


print("\n============================================")
print("Figure 6 completed")
print("============================================")

print(PNG)
print(SVG)
print(PDF)

print("\nStatistics:")
print(STAT_OUT)

print("\nROC predictions:")
print(ROC_OUT)


plt.close(
    fig
)

