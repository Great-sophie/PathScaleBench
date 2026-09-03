#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 7 — Associations between representation-level scale-awareness
metrics and conventional cancer retrieval accuracy
====================================================================

Analysis unit
-------------
PFM (n = 8), NOT individual cases.

Outcome
-------
Cancer retrieval Recall@1 from the multiscale slide embedding analysis.

Predictors
----------
1. Global alignment (mean cross-scale CKA)
2. Relative retention
3. Cross-scale retrieval similarity
4. Neighborhood Preservation Score (NPS)

Statistics
----------
- Spearman rank correlation across 8 PFMs.
- Exact two-sided permutation p-value by exhaustive enumeration of 8! = 40,320
  permutations.
- Holm correction across the four planned associations.
- Leave-one-model-out Spearman sensitivity analysis.

Important
---------
This is an exploratory model-level association analysis.
It must not be interpreted as causal evidence.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
NATIVE_ROOT = SCRIPT.parents[1]

RESULT_ROOT = (
    NATIVE_ROOT
    / "reviewer_analysis"
    / "results"
)

FIG_ROOT = (
    NATIVE_ROOT
    / "reviewer_analysis"
    / "figures"
)

FIG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_raw.csv"
)

OUT_STATS = (
    RESULT_ROOT
    / "19_figure7_scale_awareness_vs_recall1_statistics.csv"
)

OUT_LOO = (
    RESULT_ROOT
    / "19_figure7_scale_awareness_vs_recall1_leave_one_model_out.csv"
)

OUT_DATA = (
    RESULT_ROOT
    / "19_figure7_scale_awareness_vs_recall1_plot_data.csv"
)

OUT_PNG = (
    FIG_ROOT
    / "Figure7_Scale_Awareness_vs_Cancer_Retrieval_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure7_Scale_Awareness_vs_Cancer_Retrieval_8PFM.pdf"
)


# ============================================================
# DEFINITIONS
# ============================================================

EXPECTED_MODELS = [
    "phikon",
    "uni",
    "virchow",
    "virchow2",
    "uni2",
    "midnight",
    "gigapath",
    "gigapath_flash",
]

MODEL_DISPLAY = {
    "phikon": "Phikon",
    "uni": "UNI",
    "virchow": "Virchow",
    "virchow2": "Virchow2",
    "uni2": "UNI2",
    "midnight": "Midnight",
    "gigapath": "GigaPath",
    "gigapath_flash": "GigaPath-Flash",
}

PREDICTORS = [
    "global_alignment",
    "relative_retention",
    "cross_scale_retrieval",
    "neighborhood_preservation",
]

PREDICTOR_LABELS = {
    "global_alignment":
        "Global alignment (CKA)",

    "relative_retention":
        "Relative retention",

    "cross_scale_retrieval":
        "Cross-scale retrieval similarity",

    "neighborhood_preservation":
        "Neighborhood Preservation Score",
}

PANEL_LABELS = {
    "global_alignment": "A",
    "relative_retention": "B",
    "cross_scale_retrieval": "C",
    "neighborhood_preservation": "D",
}

OUTCOME = "biological_identity"

OUTCOME_LABEL = (
    "Cancer retrieval Recall@1"
)


# ============================================================
# LOAD
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing input:\n{INPUT}"
    )

df = pd.read_csv(
    INPUT
)

# Figure 6 raw file was written with index_label="model"
if "model" not in df.columns:
    raise RuntimeError(
        f"'model' column missing.\n"
        f"Columns: {df.columns.tolist()}"
    )

required = set(
    PREDICTORS
    + [
        OUTCOME,
        "model_display",
    ]
)

missing = (
    required
    - set(df.columns)
)

if missing:
    raise RuntimeError(
        f"Missing columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# AUDIT
# ============================================================

df = (
    df.set_index(
        "model"
    )
    .reindex(
        EXPECTED_MODELS
    )
    .reset_index()
)

if len(df) != 8:
    raise RuntimeError(
        f"Expected 8 models, "
        f"got {len(df)}"
    )

if df[
    PREDICTORS
    + [OUTCOME]
].isna().any().any():

    raise RuntimeError(
        "Missing benchmark values."
    )

if set(
    df["model"]
) != set(
    EXPECTED_MODELS
):

    raise RuntimeError(
        "Unexpected model set."
    )


# ============================================================
# HELPERS
# ============================================================

def rankdata_average(x):
    """
    pandas rank provides average ranks for ties.
    """

    return (
        pd.Series(
            np.asarray(
                x,
                dtype=float,
            )
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )


def pearson_corr(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    x = (
        x
        - np.mean(x)
    )

    y = (
        y
        - np.mean(y)
    )

    denom = (
        np.sqrt(
            np.sum(
                x * x
            )
        )
        *
        np.sqrt(
            np.sum(
                y * y
            )
        )
    )

    if denom == 0:
        return np.nan

    return float(
        np.sum(
            x * y
        )
        / denom
    )


def exact_spearman_permutation(
    x,
    y,
):
    """
    Exact two-sided permutation p-value.

    With n=8:
        8! = 40,320 permutations.

    Spearman correlation is simply Pearson correlation
    of the rank vectors.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) != 8:
        raise RuntimeError(
            "Exact exhaustive implementation "
            "expects n=8."
        )

    rx = rankdata_average(
        x
    )

    ry = rankdata_average(
        y
    )

    rho_obs = pearson_corr(
        rx,
        ry,
    )

    abs_obs = abs(
        rho_obs
    )

    extreme = 0
    total = 0

    # Exact enumeration over all 8! permutations
    for perm in itertools.permutations(
        range(
            len(ry)
        )
    ):

        rho_perm = pearson_corr(
            rx,
            ry[
                list(
                    perm
                )
            ],
        )

        if (
            abs(
                rho_perm
            )
            >=
            abs_obs - 1e-12
        ):
            extreme += 1

        total += 1

    p_exact = (
        extreme
        / total
    )

    return (
        float(
            rho_obs
        ),
        float(
            p_exact
        ),
        int(
            extreme
        ),
        int(
            total
        ),
    )


def holm_adjust(
    pvalues,
):
    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(
        pvalues
    )

    order = np.argsort(
        pvalues
    )

    sorted_p = (
        pvalues[
            order
        ]
    )

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, p in enumerate(
        sorted_p
    ):

        adjusted = (
            (m - i)
            * p
        )

        running_max = max(
            running_max,
            adjusted,
        )

        adjusted_sorted[
            i
        ] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted


def p_text(
    p,
):
    if p < 0.001:
        return "p < 0.001"

    return f"p = {p:.3f}"


# ============================================================
# MAIN ASSOCIATIONS
# ============================================================

stats_rows = []

y = (
    df[
        OUTCOME
    ]
    .to_numpy(
        dtype=float
    )
)

for predictor in PREDICTORS:

    x = (
        df[
            predictor
        ]
        .to_numpy(
            dtype=float
        )
    )

    rho_scipy = float(
        spearmanr(
            x,
            y
        ).statistic
    )

    (
        rho_exact,
        p_exact,
        n_extreme,
        n_permutations,
    ) = exact_spearman_permutation(
        x,
        y,
    )

    if not np.isclose(
        rho_scipy,
        rho_exact,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Spearman mismatch for "
            f"{predictor}: "
            f"{rho_scipy} vs "
            f"{rho_exact}"
        )

    stats_rows.append({
        "predictor":
            predictor,

        "predictor_label":
            PREDICTOR_LABELS[
                predictor
            ],

        "n_models":
            len(df),

        "spearman_rho":
            rho_exact,

        "p_exact":
            p_exact,

        "n_extreme_permutations":
            n_extreme,

        "n_total_permutations":
            n_permutations,
    })


stats_df = pd.DataFrame(
    stats_rows
)

stats_df[
    "p_holm"
] = holm_adjust(
    stats_df[
        "p_exact"
    ].to_numpy(
        dtype=float
    )
)

stats_df[
    "significant_raw_0p05"
] = (
    stats_df[
        "p_exact"
    ]
    < 0.05
)

stats_df[
    "significant_holm_0p05"
] = (
    stats_df[
        "p_holm"
    ]
    < 0.05
)

stats_df.to_csv(
    OUT_STATS,
    index=False,
)


# ============================================================
# LEAVE-ONE-MODEL-OUT SENSITIVITY
# ============================================================

loo_rows = []

for predictor in PREDICTORS:

    for drop_model in EXPECTED_MODELS:

        sub = df[
            df["model"]
            != drop_model
        ]

        x_sub = (
            sub[
                predictor
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_sub = (
            sub[
                OUTCOME
            ]
            .to_numpy(
                dtype=float
            )
        )

        rho = float(
            spearmanr(
                x_sub,
                y_sub,
            ).statistic
        )

        loo_rows.append({
            "predictor":
                predictor,

            "predictor_label":
                PREDICTOR_LABELS[
                    predictor
                ],

            "excluded_model":
                drop_model,

            "excluded_model_display":
                MODEL_DISPLAY[
                    drop_model
                ],

            "n_models":
                len(sub),

            "spearman_rho":
                rho,
        })


loo_df = pd.DataFrame(
    loo_rows
)

loo_df.to_csv(
    OUT_LOO,
    index=False,
)


# Add sensitivity ranges to stats table
loo_summary = (
    loo_df.groupby(
        "predictor",
        observed=True,
    )[
        "spearman_rho"
    ]
    .agg([
        "min",
        "max",
        "median",
    ])
    .rename(
        columns={
            "min":
                "loo_rho_min",

            "max":
                "loo_rho_max",

            "median":
                "loo_rho_median",
        }
    )
)

stats_df = (
    stats_df
    .merge(
        loo_summary,
        left_on="predictor",
        right_index=True,
        how="left",
    )
)

stats_df.to_csv(
    OUT_STATS,
    index=False,
)


# ============================================================
# SAVE PLOT DATA
# ============================================================

plot_df = df[
    [
        "model",
        "model_display",
        OUTCOME,
    ]
    + PREDICTORS
].copy()

plot_df.to_csv(
    OUT_DATA,
    index=False,
)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family":
        "DejaVu Sans",

    "font.size":
        10,

    "axes.labelsize":
        10.5,

    "axes.titlesize":
        11,

    "axes.linewidth":
        1.0,

    "xtick.labelsize":
        9,

    "ytick.labelsize":
        9,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.3, 8.0),
)

axes = axes.ravel()


for ax, predictor in zip(
    axes,
    PREDICTORS,
):

    x = (
        df[
            predictor
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        df[
            OUTCOME
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Regression line for visual guidance only
    # --------------------------------------------------------

    coef = np.polyfit(
        x,
        y,
        deg=1,
    )

    x_line = np.linspace(
        np.min(x),
        np.max(x),
        200,
    )

    y_line = (
        coef[0]
        * x_line
        + coef[1]
    )

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.1,
        alpha=0.55,
        zorder=1,
    )


    # --------------------------------------------------------
    # Model points
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        s=58,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )


    # --------------------------------------------------------
    # Point labels
    # --------------------------------------------------------

    x_span = (
        np.max(x)
        - np.min(x)
    )

    if x_span == 0:
        x_span = 1.0

    y_span = (
        np.max(y)
        - np.min(y)
    )

    if y_span == 0:
        y_span = 1.0

    for _, row in df.iterrows():

        xx = float(
            row[
                predictor
            ]
        )

        yy = float(
            row[
                OUTCOME
            ]
        )

        name = str(
            row[
                "model_display"
            ]
        )

        # small model-specific offsets to reduce collisions
        dx = (
            0.012
            * x_span
        )

        dy = (
            0.012
            * y_span
        )

        ha = "left"
        va = "bottom"

        if name in [
            "UNI2",
            "Midnight",
        ]:
            va = "top"
            dy = (
                -0.015
                * y_span
            )

        if name in [
            "Virchow2",
            "GigaPath-Flash",
        ]:
            ha = "right"
            dx = (
                -0.012
                * x_span
            )

        ax.text(
            xx + dx,
            yy + dy,
            name,
            fontsize=8,
            ha=ha,
            va=va,
            zorder=4,
        )


    # --------------------------------------------------------
    # Stats annotation
    # --------------------------------------------------------

    stat_row = (
        stats_df.loc[
            stats_df[
                "predictor"
            ] == predictor
        ]
        .iloc[0]
    )

    rho = float(
        stat_row[
            "spearman_rho"
        ]
    )

    p_exact = float(
        stat_row[
            "p_exact"
        ]
    )

    p_holm = float(
        stat_row[
            "p_holm"
        ]
    )

    loo_min = float(
        stat_row[
            "loo_rho_min"
        ]
    )

    loo_max = float(
        stat_row[
            "loo_rho_max"
        ]
    )

    stats_text = (
        rf"Spearman $\rho={rho:.2f}$"
        "\n"
        rf"exact {p_text(p_exact)}"
        "\n"
        rf"Holm $p={p_holm:.3f}$"
        "\n"
        rf"LOO $\rho$: {loo_min:.2f} to {loo_max:.2f}"
    )

    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle":
                "round,pad=0.3",

            "facecolor":
                "white",

            "edgecolor":
                "0.75",

            "alpha":
                0.88,
        },
    )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlabel(
        PREDICTOR_LABELS[
            predictor
        ]
    )

    ax.set_ylabel(
        OUTCOME_LABEL
    )

    ax.set_title(
        f"{PANEL_LABELS[predictor]}   "
        f"{PREDICTOR_LABELS[predictor]}",
        loc="left",
        fontweight="bold",
    )

    ax.grid(
        linewidth=0.5,
        alpha=0.16,
    )

    ax.set_axisbelow(
        True
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


# ============================================================
# GLOBAL LABELS
# ============================================================

fig.suptitle(
    "Associations between representation-level scale-awareness metrics "
    "and conventional cancer retrieval accuracy",
    fontsize=13,
    fontweight="bold",
    y=0.985,
)

fig.text(
    0.015,
    0.012,
    (
        "Each point represents one pathology foundation model (n=8). "
        "Associations were assessed using Spearman rank correlation with "
        "exact two-sided permutation tests over all 8! model-label permutations; "
        "p-values were Holm-adjusted across four planned comparisons. "
        "Dashed lines are linear visual guides only and are not used for inference. "
        "LOO denotes leave-one-model-out sensitivity analysis."
    ),
    ha="left",
    va="bottom",
    fontsize=8,
)


fig.tight_layout(
    rect=[
        0,
        0.055,
        1,
        0.955,
    ]
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# REPORT
# ============================================================

print(
    "=" * 108
)

print(
    "FIGURE 7 — SCALE-AWARENESS METRICS VS CANCER RETRIEVAL ACCURACY"
)

print(
    "=" * 108
)

print(
    "\nMODEL-LEVEL DATA"
)

print(
    plot_df[
        [
            "model_display",
            "global_alignment",
            "relative_retention",
            "cross_scale_retrieval",
            "neighborhood_preservation",
            OUTCOME,
        ]
    ]
    .rename(
        columns={
            "global_alignment":
                "CKA",

            "relative_retention":
                "Retention",

            "cross_scale_retrieval":
                "Retrieval",

            "neighborhood_preservation":
                "NPS",

            OUTCOME:
                "Recall@1",
        }
    )
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nASSOCIATIONS"
)

report = stats_df[
    [
        "predictor_label",
        "spearman_rho",
        "p_exact",
        "p_holm",
        "loo_rho_min",
        "loo_rho_max",
        "loo_rho_median",
        "significant_holm_0p05",
    ]
].copy()

print(
    report.to_string(
        index=False,
        formatters={
            "spearman_rho":
                lambda x:
                    f"{x:.4f}",

            "p_exact":
                lambda x:
                    f"{x:.6f}",

            "p_holm":
                lambda x:
                    f"{x:.6f}",

            "loo_rho_min":
                lambda x:
                    f"{x:.4f}",

            "loo_rho_max":
                lambda x:
                    f"{x:.4f}",

            "loo_rho_median":
                lambda x:
                    f"{x:.4f}",
        },
    )
)


print(
    "\nOutputs:"
)

for p in [
    OUT_PNG,
    OUT_PDF,
    OUT_STATS,
    OUT_LOO,
    OUT_DATA,
]:
    print(p)
