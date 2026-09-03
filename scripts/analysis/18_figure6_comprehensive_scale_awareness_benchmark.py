#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 6 — Comprehensive benchmark summary of scale-awareness
==============================================================

Integrates five complementary representation-level dimensions:

1. Global alignment
      Mean cross-scale CKA
      Source: Figure 2

2. Relative retention
      Relative cross-scale representation retention
      Source: Figure 3

3. Cross-scale retrieval
      Mean bidirectional Top-5 cross-scale retrieval similarity
      Source: Figure 4

4. Biological identity
      Cancer-type leave-one-out Recall@1
      Source: Figure 5A

5. Neighborhood topology
      Neighborhood Preservation Score (NPS)
      Source: Figure 5D

Figure 6A
---------
8 PFMs × 5 dimensions.

Cell color:
    within-metric min-max normalization across the 8 PFMs.

Cell text:
    original raw metric value.

Figure 6B
---------
Descriptive mean rank across the five metrics.

For every metric:
    rank 1 = highest / best.

The five metric-specific ranks are equally weighted.

IMPORTANT:
The mean rank is a descriptive benchmark summary only.
It is NOT interpreted as a validated composite performance score.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

CKA_INPUT = (
    RESULT_ROOT
    / "01b_cross_scale_cka_8pfm_fullpatch_per_case.csv"
)

RETENTION_INPUT = (
    RESULT_ROOT
    / "02_relative_information_retention_8pfm_summary.csv"
)

RETRIEVAL_INPUT = (
    RESULT_ROOT
    / "07_cross_scale_retrieval_similarity_8pfm_summary_by_model.csv"
)

BIOLOGICAL_INPUT = (
    RESULT_ROOT
    / "12_cancer_retrieval_recall_8pfm_summary.csv"
)

NPS_INPUT = (
    RESULT_ROOT
    / "16_neighborhood_preservation_score_8pfm_summary.csv"
)


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

OUT_RAW = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_raw.csv"
)

OUT_NORMALIZED = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_normalized.csv"
)

OUT_RANKS = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_ranks.csv"
)

OUT_SUMMARY = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_summary.csv"
)

OUT_PNG = (
    FIG_ROOT
    / "Figure6_Comprehensive_Scale_Awareness_Benchmark_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure6_Comprehensive_Scale_Awareness_Benchmark_8PFM.pdf"
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODELS = [
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

DISPLAY_TO_MODEL = {
    v: k
    for k, v in MODEL_DISPLAY.items()
}


# ============================================================
# METRIC DEFINITIONS
# ============================================================

METRIC_KEYS = [
    "global_alignment",
    "relative_retention",
    "cross_scale_retrieval",
    "biological_identity",
    "neighborhood_preservation",
]

METRIC_LABELS = {
    "global_alignment":
        "Global\nalignment\n(CKA)",

    "relative_retention":
        "Relative\nretention",

    "cross_scale_retrieval":
        "Cross-scale\nretrieval",

    "biological_identity":
        "Biological\nidentity\n(Recall@1)",

    "neighborhood_preservation":
        "Neighborhood\npreservation\n(NPS)",
}

METRIC_SHORT = {
    "global_alignment":
        "CKA",

    "relative_retention":
        "Retention",

    "cross_scale_retrieval":
        "Retrieval",

    "biological_identity":
        "Recall@1",

    "neighborhood_preservation":
        "NPS",
}


# ============================================================
# HELPERS
# ============================================================

def check_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input:\n{path}"
        )


def canonicalize_model_column(df):
    """
    Ensure both:
        model
        model_display

    are available.
    """

    out = df.copy()

    if "model" in out.columns:

        out["model"] = (
            out["model"]
            .astype(str)
        )

    if "model_display" in out.columns:

        out["model_display"] = (
            out["model_display"]
            .astype(str)
        )

    elif "Model" in out.columns:

        out["model_display"] = (
            out["Model"]
            .astype(str)
        )

    # derive model from display
    if "model" not in out.columns:

        if "model_display" not in out.columns:
            raise RuntimeError(
                "Could not identify model column."
            )

        out["model"] = (
            out["model_display"]
            .map(DISPLAY_TO_MODEL)
        )

    # derive display from internal name
    if "model_display" not in out.columns:

        out["model_display"] = (
            out["model"]
            .map(MODEL_DISPLAY)
        )

    # handle older uppercase labels
    replacements = {
        "PHIKON": "Phikon",
        "VIRCHOW": "Virchow",
        "MIDNIGHT": "Midnight",
        "GIGAPATH": "GigaPath",
        "GIGAPATH-FLASH": "GigaPath-Flash",
    }

    out["model_display"] = (
        out["model_display"]
        .replace(replacements)
    )

    # re-derive internal where necessary
    mask = ~out[
        "model"
    ].isin(MODELS)

    if mask.any():

        out.loc[
            mask,
            "model"
        ] = (
            out.loc[
                mask,
                "model_display"
            ]
            .map(DISPLAY_TO_MODEL)
        )

    return out


def find_first_column(
    df,
    candidates,
):
    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# 1. GLOBAL ALIGNMENT — CKA
# ============================================================

def load_global_alignment():

    check_file(
        CKA_INPUT
    )

    df = pd.read_csv(
        CKA_INPUT
    )

    df = canonicalize_model_column(
        df
    )

    # --------------------------------------------------------
    # Possible long-format CKA
    # --------------------------------------------------------

    cka_value_col = find_first_column(
        df,
        [
            "cka",
            "CKA",
            "linear_cka",
            "cka_value",
        ],
    )

    if (
        cka_value_col is not None
        and "scale_pair" in df.columns
    ):

        result = (
            df.groupby(
                "model",
                observed=True,
            )[cka_value_col]
            .mean()
        )

        return result


    # --------------------------------------------------------
    # Possible explicit overall column
    # --------------------------------------------------------

    overall_col = find_first_column(
        df,
        [
            "mean_cka",
            "overall_cka",
            "mean_cross_scale_cka",
            "cross_scale_cka",
        ],
    )

    if overall_col is not None:

        result = (
            df.groupby(
                "model",
                observed=True,
            )[overall_col]
            .mean()
        )

        return result


    # --------------------------------------------------------
    # Wide format:
    # detect pair-specific CKA columns
    # --------------------------------------------------------

    numeric_cols = (
        df.select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
        .tolist()
    )

    pair_cols = []

    for col in numeric_cols:

        name = col.lower()

        if "cka" not in name:
            continue

        if any(
            token in name
            for token in [
                "mean",
                "overall",
                "std",
                "sem",
                "ci",
                "n_",
                "count",
            ]
        ):
            continue

        pair_cols.append(
            col
        )

    if len(pair_cols) < 3:

        raise RuntimeError(
            "Could not identify three "
            "pair-specific CKA columns in:\n"
            f"{CKA_INPUT}\n"
            f"Columns = {df.columns.tolist()}"
        )

    # exactly the representation intended for Fig. 6:
    # equal average across all cross-scale CKA pairs
    temp = df.copy()

    temp[
        "_overall_cka"
    ] = (
        temp[
            pair_cols
        ]
        .mean(
            axis=1
        )
    )

    result = (
        temp.groupby(
            "model",
            observed=True,
        )[
            "_overall_cka"
        ]
        .mean()
    )

    print(
        "CKA pair columns detected:",
        pair_cols,
    )

    return result


# ============================================================
# 2. RELATIVE RETENTION
# ============================================================

def load_relative_retention():

    check_file(
        RETENTION_INPUT
    )

    df = pd.read_csv(
        RETENTION_INPUT
    )

    df = canonicalize_model_column(
        df
    )

    value_col = find_first_column(
        df,
        [
            "mean_relative_retention",
            "mean_retention",
            "relative_retention_mean",
            "relative_retention",
            "mean",
        ],
    )

    if value_col is None:

        raise RuntimeError(
            "Cannot identify retention "
            f"mean column in {RETENTION_INPUT}\n"
            f"{df.columns.tolist()}"
        )

    return (
        df.set_index(
            "model"
        )[value_col]
    )


# ============================================================
# 3. CROSS-SCALE RETRIEVAL
# ============================================================

def load_cross_scale_retrieval():

    check_file(
        RETRIEVAL_INPUT
    )

    df = pd.read_csv(
        RETRIEVAL_INPUT
    )

    df = canonicalize_model_column(
        df
    )

    value_col = find_first_column(
        df,
        [
            "mean_retrieval_similarity",
            "overall_retrieval_similarity",
            "mean_similarity",
            "retrieval_similarity",
            "mean",
        ],
    )

    if value_col is None:

        raise RuntimeError(
            "Cannot identify retrieval "
            f"column in {RETRIEVAL_INPUT}\n"
            f"{df.columns.tolist()}"
        )

    return (
        df.set_index(
            "model"
        )[value_col]
    )


# ============================================================
# 4. BIOLOGICAL IDENTITY — RECALL@1
# ============================================================

def load_biological_identity():

    check_file(
        BIOLOGICAL_INPUT
    )

    df = pd.read_csv(
        BIOLOGICAL_INPUT
    )

    df = canonicalize_model_column(
        df
    )

    # filter strict nearest-neighbor endpoint
    if "k" in df.columns:

        df = df[
            df["k"].astype(int)
            == 1
        ].copy()

    elif "metric" in df.columns:

        df = df[
            df["metric"].astype(str)
            == "Recall@1"
        ].copy()

    else:

        raise RuntimeError(
            "Biological identity table "
            "contains neither k nor metric."
        )

    value_col = find_first_column(
        df,
        [
            "recall",
            "Recall",
            "accuracy",
            "Accuracy",
            "mean",
        ],
    )

    if value_col is None:

        raise RuntimeError(
            "Cannot identify Recall@1 "
            f"column in {BIOLOGICAL_INPUT}\n"
            f"{df.columns.tolist()}"
        )

    return (
        df.set_index(
            "model"
        )[value_col]
    )


# ============================================================
# 5. NEIGHBORHOOD PRESERVATION
# ============================================================

def load_nps():

    check_file(
        NPS_INPUT
    )

    df = pd.read_csv(
        NPS_INPUT
    )

    df = canonicalize_model_column(
        df
    )

    value_col = find_first_column(
        df,
        [
            "mean_nps",
            "mean_NPS",
            "neighborhood_preservation_score",
            "mean",
        ],
    )

    if value_col is None:

        raise RuntimeError(
            "Cannot identify NPS "
            f"column in {NPS_INPUT}\n"
            f"{df.columns.tolist()}"
        )

    return (
        df.set_index(
            "model"
        )[value_col]
    )


# ============================================================
# LOAD ALL FIVE DIMENSIONS
# ============================================================

print(
    "=" * 104
)

print(
    "FIGURE 6 — COMPREHENSIVE "
    "SCALE-AWARENESS BENCHMARK"
)

print(
    "=" * 104
)


global_alignment = (
    load_global_alignment()
)

relative_retention = (
    load_relative_retention()
)

cross_scale_retrieval = (
    load_cross_scale_retrieval()
)

biological_identity = (
    load_biological_identity()
)

neighborhood_preservation = (
    load_nps()
)


# ============================================================
# BUILD RAW MATRIX
# ============================================================

raw = pd.DataFrame(
    index=MODELS
)

raw[
    "global_alignment"
] = global_alignment

raw[
    "relative_retention"
] = relative_retention

raw[
    "cross_scale_retrieval"
] = cross_scale_retrieval

raw[
    "biological_identity"
] = biological_identity

raw[
    "neighborhood_preservation"
] = neighborhood_preservation


# hard audit
if raw.isna().any().any():

    print(
        "\nRAW TABLE WITH MISSING VALUES:"
    )

    print(
        raw.to_string()
    )

    raise RuntimeError(
        "Figure 6 benchmark contains "
        "missing metric values."
    )


if raw.shape != (
    8,
    5,
):

    raise RuntimeError(
        f"Unexpected raw matrix shape: "
        f"{raw.shape}"
    )


raw.insert(
    0,
    "model_display",
    [
        MODEL_DISPLAY[
            model
        ]
        for model in raw.index
    ],
)


raw.to_csv(
    OUT_RAW,
    index_label="model",
)


# ============================================================
# WITHIN-METRIC MIN-MAX NORMALIZATION
# ============================================================

raw_numeric = (
    raw[
        METRIC_KEYS
    ]
    .copy()
)

normalized = pd.DataFrame(
    index=raw_numeric.index
)

for metric in METRIC_KEYS:

    values = (
        raw_numeric[
            metric
        ]
        .astype(float)
    )

    lo = float(
        values.min()
    )

    hi = float(
        values.max()
    )

    if np.isclose(
        hi,
        lo,
    ):

        warnings.warn(
            f"{metric}: all models have "
            f"the same value; assigning "
            f"normalized score 0.5."
        )

        normalized[
            metric
        ] = 0.5

    else:

        normalized[
            metric
        ] = (
            values - lo
        ) / (
            hi - lo
        )


normalized.insert(
    0,
    "model_display",
    raw[
        "model_display"
    ],
)

normalized.to_csv(
    OUT_NORMALIZED,
    index_label="model",
)


# ============================================================
# WITHIN-METRIC RANKS
# ============================================================

# Higher raw score is better for all five metrics.
ranks = (
    raw_numeric.rank(
        axis=0,
        method="average",
        ascending=False,
    )
)

ranks[
    "mean_rank"
] = (
    ranks[
        METRIC_KEYS
    ]
    .mean(
        axis=1
    )
)

ranks[
    "rank_sd"
] = (
    ranks[
        METRIC_KEYS
    ]
    .std(
        axis=1,
        ddof=0,
    )
)

# Ranking of the descriptive mean rank itself
ranks[
    "descriptive_overall_rank"
] = (
    ranks[
        "mean_rank"
    ]
    .rank(
        method="min",
        ascending=True,
    )
    .astype(int)
)

ranks.insert(
    0,
    "model_display",
    raw[
        "model_display"
    ],
)

ranks.to_csv(
    OUT_RANKS,
    index_label="model",
)


# ============================================================
# COMBINED SUMMARY
# ============================================================

summary = pd.DataFrame(
    index=MODELS
)

summary[
    "model_display"
] = raw[
    "model_display"
]

for metric in METRIC_KEYS:

    summary[
        f"{metric}_raw"
    ] = raw_numeric[
        metric
    ]

    summary[
        f"{metric}_normalized"
    ] = normalized[
        metric
    ]

    summary[
        f"{metric}_rank"
    ] = ranks[
        metric
    ]


summary[
    "mean_rank"
] = ranks[
    "mean_rank"
]

summary[
    "rank_sd"
] = ranks[
    "rank_sd"
]

summary[
    "descriptive_overall_rank"
] = ranks[
    "descriptive_overall_rank"
]

summary = (
    summary.sort_values(
        [
            "mean_rank",
            "model_display",
        ]
    )
)

summary.to_csv(
    OUT_SUMMARY,
    index_label="model",
)


# ============================================================
# REPORT RAW VALUES
# ============================================================

print(
    "\nRAW METRICS"
)

report_raw = (
    raw_numeric.copy()
)

report_raw.index = [
    MODEL_DISPLAY[
        model
    ]
    for model in report_raw.index
]

report_raw.columns = [
    METRIC_SHORT[
        metric
    ]
    for metric in report_raw.columns
]

print(
    report_raw
    .round(4)
    .to_string()
)


print(
    "\n"
    + "=" * 104
)

print(
    "DESCRIPTIVE MEAN RANK"
)

print(
    "=" * 104
)

report_rank = (
    summary[
        [
            "model_display",
            "mean_rank",
            "rank_sd",
            "descriptive_overall_rank",
        ]
    ]
    .copy()
)

print(
    report_rank
    .round(3)
    .to_string(
        index=False
    )
)


# ============================================================
# ORDER FOR FIGURE 6
# ============================================================

# Heatmap and rank plot use the same ordering:
# descriptive mean rank, best at top.
FIG_MODEL_ORDER = (
    summary[
        "model_display"
    ]
    .tolist()
)

fig_model_internal = [
    DISPLAY_TO_MODEL[
        x
    ]
    for x in FIG_MODEL_ORDER
]


heat_norm = (
    normalized.loc[
        fig_model_internal,
        METRIC_KEYS,
    ]
    .to_numpy(
        dtype=float
    )
)

heat_raw = (
    raw_numeric.loc[
        fig_model_internal,
        METRIC_KEYS,
    ]
    .to_numpy(
        dtype=float
    )
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
        11,

    "axes.titlesize":
        12,

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
# FIGURE LAYOUT
# ============================================================

fig = plt.figure(
    figsize=(12.6, 6.7)
)

gs = GridSpec(
    nrows=1,
    ncols=2,
    width_ratios=[
        1.55,
        0.82,
    ],
    wspace=0.34,
    figure=fig,
)

ax_heat = fig.add_subplot(
    gs[0, 0]
)

ax_rank = fig.add_subplot(
    gs[0, 1]
)


# ============================================================
# PANEL A — MULTIDIMENSIONAL HEATMAP
# ============================================================

im = ax_heat.imshow(
    heat_norm,
    cmap="YlGnBu",
    vmin=0.0,
    vmax=1.0,
    aspect="auto",
    interpolation="nearest",
)


ax_heat.set_xticks(
    np.arange(
        len(METRIC_KEYS)
    )
)

ax_heat.set_xticklabels(
    [
        METRIC_LABELS[
            metric
        ]
        for metric in METRIC_KEYS
    ],
    rotation=0,
)


ax_heat.set_yticks(
    np.arange(
        len(FIG_MODEL_ORDER)
    )
)

ax_heat.set_yticklabels(
    FIG_MODEL_ORDER
)


ax_heat.set_xlabel(
    "Scale-awareness dimension"
)

ax_heat.set_ylabel(
    "Pathology foundation model"
)

ax_heat.set_title(
    "A   Multidimensional scale-awareness profile",
    loc="left",
    fontweight="bold",
)


# ------------------------------------------------------------
# RAW VALUE ANNOTATIONS
# ------------------------------------------------------------

for i in range(
    heat_raw.shape[0]
):

    for j in range(
        heat_raw.shape[1]
    ):

        raw_value = float(
            heat_raw[
                i,
                j
            ]
        )

        norm_value = float(
            heat_norm[
                i,
                j
            ]
        )

        text_color = (
            "white"
            if norm_value >= 0.58
            else "black"
        )

        ax_heat.text(
            j,
            i,
            f"{raw_value:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
            fontweight="medium",
        )


# ------------------------------------------------------------
# HEATMAP GRID
# ------------------------------------------------------------

ax_heat.set_xticks(
    np.arange(
        -0.5,
        len(METRIC_KEYS),
        1,
    ),
    minor=True,
)

ax_heat.set_yticks(
    np.arange(
        -0.5,
        len(FIG_MODEL_ORDER),
        1,
    ),
    minor=True,
)

ax_heat.grid(
    which="minor",
    color="white",
    linewidth=1.0,
)

ax_heat.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

for spine in (
    ax_heat.spines.values()
):

    spine.set_visible(
        False
    )


# ------------------------------------------------------------
# COLORBAR
# ------------------------------------------------------------

cbar = fig.colorbar(
    im,
    ax=ax_heat,
    fraction=0.035,
    pad=0.025,
)

cbar.set_label(
    "Within-metric normalized score"
)

cbar.set_ticks([
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
])

cbar.outline.set_linewidth(
    0.8
)


# ============================================================
# PANEL B — DESCRIPTIVE MEAN RANK
# ============================================================

rank_plot = (
    summary.set_index(
        "model_display"
    )
    .loc[
        FIG_MODEL_ORDER
    ]
)


y = np.arange(
    len(FIG_MODEL_ORDER)
)

mean_rank = (
    rank_plot[
        "mean_rank"
    ]
    .to_numpy(
        dtype=float
    )
)

rank_sd = (
    rank_plot[
        "rank_sd"
    ]
    .to_numpy(
        dtype=float
    )
)


bars = ax_rank.barh(
    y,
    mean_rank,
    height=0.62,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.78,
)


ax_rank.set_yticks(
    y
)

ax_rank.set_yticklabels(
    FIG_MODEL_ORDER
)

ax_rank.invert_yaxis()


ax_rank.set_xlabel(
    "Mean rank across five dimensions"
)

ax_rank.set_ylabel(
    ""
)

ax_rank.set_xlim(
    0.8,
    8.35,
)

ax_rank.set_xticks(
    np.arange(
        1,
        9,
        1,
    )
)

ax_rank.set_title(
    "B   Descriptive multidimensional rank",
    loc="left",
    fontweight="bold",
)


ax_rank.axvline(
    1.0,
    linestyle="--",
    linewidth=0.8,
    alpha=0.35,
)


ax_rank.grid(
    axis="x",
    linewidth=0.5,
    alpha=0.18,
)

ax_rank.set_axisbelow(
    True
)

ax_rank.spines[
    "top"
].set_visible(
    False
)

ax_rank.spines[
    "right"
].set_visible(
    False
)


# ------------------------------------------------------------
# Mean rank annotation
# ------------------------------------------------------------

for i, value in enumerate(
    mean_rank
):

    ax_rank.text(
        value + 0.08,
        i,
        f"{value:.2f}",
        ha="left",
        va="center",
        fontsize=9,
    )


# ============================================================
# GLOBAL TITLE / FOOTNOTE
# ============================================================

fig.suptitle(
    "Comprehensive benchmark summary of scale-awareness "
    "in pathology foundation models",
    fontsize=14,
    fontweight="bold",
    y=0.985,
)


fig.text(
    0.01,
    0.012,
    (
        "Heatmap colors show min–max normalization performed independently "
        "within each metric; cell annotations show raw values. "
        "All five metrics are oriented so that higher values indicate stronger "
        "preservation. Mean rank assigns rank 1 to the highest-performing model "
        "within each dimension and is presented as a descriptive summary only, "
        "not as a validated composite performance score."
    ),
    ha="left",
    va="bottom",
    fontsize=8,
)


# ============================================================
# SAVE
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.975,
    bottom=0.17,
    top=0.90,
)

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
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 104
)

print(
    "FIGURE 6 COMPLETE"
)

print(
    "=" * 104
)


print(
    "\nMetric winners:"
)

for metric in METRIC_KEYS:

    winner_model = (
        raw_numeric[
            metric
        ]
        .idxmax()
    )

    winner_value = float(
        raw_numeric.loc[
            winner_model,
            metric
        ]
    )

    print(
        f"  {METRIC_SHORT[metric]:12s}: "
        f"{MODEL_DISPLAY[winner_model]:15s} "
        f"{winner_value:.4f}"
    )


print(
    "\nDescriptive overall ordering:"
)

for _, row in (
    summary
    .sort_values(
        "mean_rank"
    )
    .iterrows()
):

    print(
        f"  {int(row['descriptive_overall_rank'])}. "
        f"{row['model_display']:15s} "
        f"mean rank = "
        f"{row['mean_rank']:.3f}"
    )


print(
    "\nOutputs:"
)

for p in [
    OUT_PNG,
    OUT_PDF,
    OUT_RAW,
    OUT_NORMALIZED,
    OUT_RANKS,
    OUT_SUMMARY,
]:
    print(p)
