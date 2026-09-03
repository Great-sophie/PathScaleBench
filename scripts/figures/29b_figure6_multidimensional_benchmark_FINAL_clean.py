#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 6
Multidimensional cross-scale representation benchmark
=====================================================

A. Eight PFMs × five complementary benchmark dimensions
   - color = within-metric min-max normalized score
   - annotation = original raw metric value

B. Metric-specific rank trajectories
   - rank 1 = highest metric value
   - highlights model-specific trade-offs

C. Descriptive mean rank
   - summary only
   - NOT an inferential composite score

Inputs are frozen Analysis-18 outputs.
No metric recomputation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


RAW_FILE = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_raw.csv"
)

NORMALIZED_FILE = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_normalized.csv"
)

RANK_FILE = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_ranks.csv"
)


OUT_AUDIT = (
    RESULT_ROOT
    / "29_figure6_multidimensional_benchmark_audit.txt"
)

OUT_PNG = (
    FIG_ROOT
    / "Figure6_Multidimensional_Benchmark_FINAL_clean_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure6_Multidimensional_Benchmark_FINAL_clean_8PFM.pdf"
)

OUT_TIFF = (
    FIG_ROOT
    / "Figure6_Multidimensional_Benchmark_FINAL_clean_8PFM_600dpi.tiff"
)


# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "Phikon",
    "UNI",
    "Virchow",
    "Virchow2",
    "UNI2",
    "Midnight",
    "GigaPath",
    "GigaPath-Flash",
]


METRICS = [
    "CKA",
    "Retention",
    "Retrieval",
    "Recall@1",
    "NPS",
]


METRIC_DISPLAY = {
    "CKA":
        "Global\nalignment",

    "Retention":
        "Relative\nretention",

    "Retrieval":
        "Cross-scale\nretrieval",

    "Recall@1":
        "Biological\nidentity",

    "NPS":
        "Neighborhood\npreservation",
}


# Frozen audited raw values
EXPECTED_RAW = {
    "Phikon": {
        "CKA": 0.7135,
        "Retention": 0.7742,
        "Retrieval": 0.4946,
        "Recall@1": 0.7042,
        "NPS": 0.2476,
    },

    "UNI": {
        "CKA": 0.6545,
        "Retention": 0.7264,
        "Retrieval": 0.3311,
        "Recall@1": 0.6197,
        "NPS": 0.2193,
    },

    "Virchow": {
        "CKA": 0.5590,
        "Retention": 0.7091,
        "Retrieval": 0.5206,
        "Recall@1": 0.6056,
        "NPS": 0.1862,
    },

    "Virchow2": {
        "CKA": 0.4147,
        "Retention": 0.6140,
        "Retrieval": 0.6426,
        "Recall@1": 0.7606,
        "NPS": 0.1785,
    },

    "UNI2": {
        "CKA": 0.6856,
        "Retention": 0.7580,
        "Retrieval": 0.3923,
        "Recall@1": 0.8028,
        "NPS": 0.2171,
    },

    "Midnight": {
        "CKA": 0.6899,
        "Retention": 0.7242,
        "Retrieval": 0.3602,
        "Recall@1": 0.8451,
        "NPS": 0.1799,
    },

    "GigaPath": {
        "CKA": 0.5884,
        "Retention": 0.6850,
        "Retrieval": 0.3705,
        "Recall@1": 0.6901,
        "NPS": 0.1852,
    },

    "GigaPath-Flash": {
        "CKA": 0.5897,
        "Retention": 0.6637,
        "Retrieval": 0.5302,
        "Recall@1": 0.6479,
        "NPS": 0.1794,
    },
}


EXPECTED_MEAN_RANK = {
    "Phikon": 2.2,
    "UNI2": 3.0,
    "Midnight": 4.0,
    "UNI": 4.8,
    "GigaPath-Flash": 5.4,
    "Virchow": 5.4,
    "GigaPath": 5.6,
    "Virchow2": 5.6,
}


# ============================================================
# HELPERS
# ============================================================

def resolve_model_column(df):

    for candidate in [
        "model",
        "model_display",
        "Model",
    ]:

        if candidate in df.columns:
            return candidate

    raise RuntimeError(
        f"No model column found.\n"
        f"Columns: {list(df.columns)}"
    )


def clean_model_name(x):

    s = str(x).strip()

    aliases = {
        "phikon": "Phikon",
        "uni": "UNI",
        "virchow": "Virchow",
        "virchow2": "Virchow2",
        "uni2": "UNI2",
        "midnight": "Midnight",
        "gigapath": "GigaPath",
        "gigapath-flash": "GigaPath-Flash",
        "gigapath_flash": "GigaPath-Flash",
    }

    return aliases.get(
        s.lower(),
        s,
    )


# ============================================================
# LOAD RAW
# ============================================================

if not RAW_FILE.exists():

    raise FileNotFoundError(
        f"Missing:\n{RAW_FILE}"
    )


raw = pd.read_csv(
    RAW_FILE
)


# ============================================================
# STANDARDIZE ANALYSIS-18 METRIC COLUMN NAMES
# ============================================================

METRIC_COLUMN_MAP = {
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


raw = raw.rename(
    columns=METRIC_COLUMN_MAP
)


model_col = resolve_model_column(
    raw
)


raw[
    "model"
] = raw[
    model_col
].map(
    clean_model_name
)


for metric in METRICS:

    if metric not in raw.columns:

        raise RuntimeError(
            f"RAW file missing metric: {metric}\n"
            f"Columns: {list(raw.columns)}"
        )


raw = (
    raw[
        [
            "model"
        ]
        + METRICS
    ]
    .drop_duplicates(
        subset=[
            "model"
        ]
    )
    .set_index(
        "model"
    )
    .reindex(
        MODELS
    )
)


if raw.isna().any().any():

    raise RuntimeError(
        "Raw Figure-6 matrix contains NaN."
    )


# ============================================================
# FROZEN VALUE AUDIT
# ============================================================

audit_lines = []

audit_lines.append(
    f"Raw input: {RAW_FILE}"
)

audit_lines.append(
    ""
)

audit_lines.append(
    "Frozen raw-value audit:"
)


max_delta = 0.0


for model in MODELS:

    for metric in METRICS:

        observed = float(
            raw.loc[
                model,
                metric
            ]
        )

        expected = (
            EXPECTED_RAW[
                model
            ][
                metric
            ]
        )

        delta = (
            observed
            -
            expected
        )

        max_delta = max(
            max_delta,
            abs(
                delta
            ),
        )


audit_lines.append(
    f"Max absolute raw-value delta = "
    f"{max_delta:.8f}"
)


if max_delta > 5e-4:

    raise RuntimeError(
        f"Frozen Figure-6 audit failed: "
        f"max delta={max_delta}"
    )


# ============================================================
# NORMALIZE WITHIN EACH METRIC
# ============================================================

normalized = pd.DataFrame(
    index=MODELS,
    columns=METRICS,
    dtype=float,
)


for metric in METRICS:

    values = raw[
        metric
    ].to_numpy(
        dtype=float
    )

    lo = float(
        values.min()
    )

    hi = float(
        values.max()
    )


    if hi <= lo:

        normalized[
            metric
        ] = 1.0

    else:

        normalized[
            metric
        ] = (
            (
                raw[
                    metric
                ]
                -
                lo
            )
            /
            (
                hi
                -
                lo
            )
        )


# ============================================================
# RANKS
# ============================================================

rank_matrix = pd.DataFrame(
    index=MODELS,
    columns=METRICS,
    dtype=float,
)


for metric in METRICS:

    rank_matrix[
        metric
    ] = (
        raw[
            metric
        ]
        .rank(
            method="average",
            ascending=False,
        )
    )


mean_rank = (
    rank_matrix
    .mean(
        axis=1
    )
)


rank_sd = (
    rank_matrix
    .std(
        axis=1,
        ddof=0,
    )
)


mean_rank_df = pd.DataFrame({
    "model":
        MODELS,

    "mean_rank":
        [
            float(
                mean_rank[
                    m
                ]
            )
            for m in MODELS
        ],

    "rank_sd":
        [
            float(
                rank_sd[
                    m
                ]
            )
            for m in MODELS
        ],
})


mean_rank_df = (
    mean_rank_df
    .sort_values(
        [
            "mean_rank",
            "model",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# audit mean ranks
max_rank_delta = 0.0

for _, row in mean_rank_df.iterrows():

    model = row[
        "model"
    ]

    observed = float(
        row[
            "mean_rank"
        ]
    )

    expected = (
        EXPECTED_MEAN_RANK[
            model
        ]
    )

    max_rank_delta = max(
        max_rank_delta,
        abs(
            observed
            -
            expected
        ),
    )


audit_lines.append(
    f"Max mean-rank audit delta = "
    f"{max_rank_delta:.8f}"
)


if max_rank_delta > 1e-6:

    raise RuntimeError(
        f"Mean-rank audit failed: "
        f"{max_rank_delta}"
    )


# ============================================================
# LEADERS BY METRIC
# ============================================================

leaders = {}

for metric in METRICS:

    leaders[
        metric
    ] = raw[
        metric
    ].idxmax()


audit_lines.append(
    ""
)

audit_lines.append(
    "Metric leaders:"
)

for metric in METRICS:

    audit_lines.append(
        f"{metric:12s}: "
        f"{leaders[metric]}"
    )


n_unique_leaders = len(
    set(
        leaders.values()
    )
)


audit_lines.append(
    ""
)

audit_lines.append(
    f"Unique metric leaders = "
    f"{n_unique_leaders}"
)


OUT_AUDIT.write_text(
    "\n".join(
        audit_lines
    ),
    encoding="utf-8",
)


print(
    "\n"
    + "\n".join(
        audit_lines
    )
)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family":
        "DejaVu Sans",

    "font.size":
        9.2,

    "axes.titlesize":
        11,

    "axes.labelsize":
        9.6,

    "xtick.labelsize":
        8.6,

    "ytick.labelsize":
        8.6,

    "legend.fontsize":
        7.8,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# restrained model colors for rank trajectories
tab = plt.get_cmap(
    "tab10"
)

MODEL_COLOR = {
    model:
        tab(i)
    for i, model in enumerate(
        MODELS
    )
}


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        13.4,
        7.6,
    ),
    facecolor="white",
)


outer = fig.add_gridspec(
    2,
    1,

    height_ratios=[
        1.18,
        0.92,
    ],

    hspace=0.31,
)


# ============================================================
# PANEL A — MULTIDIMENSIONAL HEATMAP
# ============================================================

ax_a = fig.add_subplot(
    outer[
        0
    ]
)


arr_norm = normalized.to_numpy(
    dtype=float
)

arr_raw = raw.to_numpy(
    dtype=float
)


im = ax_a.imshow(
    arr_norm,

    cmap="Blues",

    vmin=0.0,

    vmax=1.0,

    aspect="auto",

    interpolation="nearest",
)


ax_a.set_xticks(
    np.arange(
        len(
            METRICS
        )
    )
)

ax_a.set_xticklabels(
    [
        METRIC_DISPLAY[
            m
        ]
        for m in METRICS
    ]
)


ax_a.set_yticks(
    np.arange(
        len(
            MODELS
        )
    )
)

ax_a.set_yticklabels(
    MODELS
)


# raw values inside normalized-color cells
for r in range(
    len(
        MODELS
    )
):

    for c in range(
        len(
            METRICS
        )
    ):

        norm_val = arr_norm[
            r,
            c
        ]

        raw_val = arr_raw[
            r,
            c
        ]


        ax_a.text(
            c,
            r,

            f"{raw_val:.3f}",

            ha="center",

            va="center",

            fontsize=8.2,

            fontweight=(
                "bold"
                if (
                    leaders[
                        METRICS[c]
                    ]
                    == MODELS[r]
                )
                else "normal"
            ),

            color=(
                "white"
                if norm_val >= 0.56
                else "black"
            ),
        )


# white cell separators
ax_a.set_xticks(
    np.arange(
        -0.5,
        len(
            METRICS
        ),
        1,
    ),
    minor=True,
)

ax_a.set_yticks(
    np.arange(
        -0.5,
        len(
            MODELS
        ),
        1,
    ),
    minor=True,
)


ax_a.grid(
    which="minor",

    color="white",

    linewidth=0.85,

    alpha=0.95,
)


ax_a.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

ax_a.tick_params(
    which="major",
    length=0,
)


for spine in ax_a.spines.values():

    spine.set_visible(
        False
    )


ax_a.set_title(
    "A   Multidimensional benchmark profiles",

    loc="left",

    fontweight="bold",

    pad=9,
)


cbar = fig.colorbar(
    im,

    ax=ax_a,

    fraction=0.015,

    pad=0.018,
)

cbar.set_label(
    "Within-metric normalized score",

    fontsize=8.7,
)

cbar.ax.tick_params(
    labelsize=8,
)


# ============================================================
# LOWER LAYOUT
# ============================================================

lower = outer[
    1
].subgridspec(
    1,
    2,

    width_ratios=[
        1.52,
        0.92,
    ],

    wspace=0.29,
)


# ============================================================
# PANEL B — RANK TRAJECTORIES
# ============================================================

ax_b = fig.add_subplot(
    lower[
        0,
        0,
    ]
)


x = np.arange(
    len(
        METRICS
    )
)


for model in MODELS:

    model_ranks = rank_matrix.loc[
        model,
        METRICS
    ].to_numpy(
        dtype=float
    )


    ax_b.plot(
        x,
        model_ranks,

        marker="o",

        markersize=3.9,

        linewidth=1.05,

        alpha=0.73,

        color=MODEL_COLOR[
            model
        ],

        zorder=2,
    )


    # Direct model label at final metric.
    # NPS ranks span 1–8, so labels are naturally separated.
    final_y = float(
        model_ranks[
            -1
        ]
    )


    ax_b.text(
        x[
            -1
        ] + 0.12,

        final_y,

        model,

        fontsize=7.7,

        fontweight="bold",

        color=MODEL_COLOR[
            model
        ],

        ha="left",

        va="center",
    )


ax_b.set_xticks(
    x
)

ax_b.set_xticklabels(
    [
        METRIC_DISPLAY[
            m
        ]
        .replace(
            "\n",
            " "
        )
        for m in METRICS
    ],

    rotation=18,

    ha="right",
)


ax_b.set_yticks(
    np.arange(
        1,
        9,
    )
)

ax_b.set_ylim(
    8.35,
    0.65,
)


# Extra horizontal room for direct model labels.
ax_b.set_xlim(
    -0.15,
    5.05,
)


ax_b.set_ylabel(
    "PFM rank (1 = highest)"
)


ax_b.set_title(
    "B   Metric-specific PFM ranks",

    loc="left",

    fontweight="bold",

    pad=8,
)


ax_b.grid(
    axis="y",

    alpha=0.10,

    linewidth=0.7,
)


ax_b.spines[
    "top"
].set_visible(False)

ax_b.spines[
    "right"
].set_visible(False)




# ============================================================
# PANEL C — DESCRIPTIVE MEAN RANK
# ============================================================

ax_c = fig.add_subplot(
    lower[
        0,
        1,
    ]
)


ranked = mean_rank_df.copy()

y = np.arange(
    len(
        ranked
    )
)


for yi, (_, row) in enumerate(
    ranked.iterrows()
):

    model = row[
        "model"
    ]

    value = float(
        row[
            "mean_rank"
        ]
    )

    sd_value = float(
        row[
            "rank_sd"
        ]
    )


    # light horizontal guide
    ax_c.hlines(
        yi,

        1.0,

        value,

        color="0.84",

        linewidth=1.1,

        zorder=1,
    )


    ax_c.scatter(
        value,
        yi,

        s=56,

        facecolor=MODEL_COLOR[
            model
        ],

        edgecolor="white",

        linewidth=0.6,

        zorder=3,
    )


    ax_c.text(
        value + 0.10,
        yi,

        f"{value:.1f}",

        ha="left",

        va="center",

        fontsize=8.0,

        fontweight="bold",
    )


ax_c.set_yticks(
    y
)

ax_c.set_yticklabels(
    ranked[
        "model"
    ]
)

ax_c.invert_yaxis()


ax_c.set_xlim(
    1.0,
    6.25,
)


ax_c.set_xlabel(
    "Mean rank across five dimensions"
)


ax_c.set_title(
    "C   Descriptive mean rank",

    loc="left",

    fontweight="bold",

    pad=8,
)


ax_c.grid(
    axis="x",

    alpha=0.10,

    linewidth=0.7,
)


ax_c.spines[
    "top"
].set_visible(False)

ax_c.spines[
    "right"
].set_visible(False)

ax_c.spines[
    "left"
].set_visible(False)

ax_c.tick_params(
    axis="y",
    length=0,
)


# Minimal direction cue only.
ax_c.text(
    0.98,
    0.97,

    "Lower is better",

    transform=ax_c.transAxes,

    ha="right",

    va="top",

    fontsize=7.9,

    color="0.45",

    style="italic",
)


# ============================================================
# GLOBAL LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.94,
    top=0.955,
    bottom=0.085,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,

    dpi=600,

    bbox_inches="tight",

    facecolor="white",
)


fig.savefig(
    OUT_PDF,

    bbox_inches="tight",

    facecolor="white",
)


fig.savefig(
    OUT_TIFF,

    dpi=600,

    bbox_inches="tight",

    facecolor="white",

    pil_kwargs={
        "compression":
            "tiff_lzw",
    },
)


plt.close(
    fig
)


# ============================================================
# REPORT
# ============================================================

print(
    "=" * 105
)

print(
    "FIGURE 6 — MULTIDIMENSIONAL BENCHMARK"
)

print(
    "=" * 105
)


print(
    "\nRaw matrix:"
)

print(
    raw
    .round(4)
    .to_string()
)


print(
    "\nMetric leaders:"
)

for metric in METRICS:

    print(
        f"{metric:12s} "
        f"{leaders[metric]}"
    )


print(
    "\nDescriptive mean rank:"
)

print(
    mean_rank_df[
        [
            "model",
            "mean_rank",
            "rank_sd",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\nOutputs:"
)

for path in [
    OUT_AUDIT,
    OUT_PNG,
    OUT_PDF,
    OUT_TIFF,
]:

    print(
        path
    )
