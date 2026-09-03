#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 7 — FINAL CLEAN VERSION
Scale-awareness metrics vs conventional cancer retrieval
==========================================================

Analysis unit:
    PFM, n = 8

Outcome:
    Cancer retrieval Recall@1

Predictors:
    A. Global alignment (CKA)
    B. Relative retention
    C. Cross-scale retrieval similarity
    D. Neighborhood Preservation Score (NPS)

Statistics:
    Frozen exact Spearman permutation results from Analysis 19.

Design:
    - no linear regression/trend line
    - direct PFM labels
    - statistics outside plotting region
    - shared y-axis limits
    - no redundant legend

No benchmark metric is recomputed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


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


OUT_DATA = (
    RESULT_ROOT
    / "30b_figure7_scale_awareness_vs_retrieval_plot_data.csv"
)

OUT_STATS = (
    RESULT_ROOT
    / "30b_figure7_scale_awareness_vs_retrieval_statistics.csv"
)

OUT_AUDIT = (
    RESULT_ROOT
    / "30b_figure7_scale_awareness_vs_retrieval_audit.txt"
)


OUT_PNG = (
    FIG_ROOT
    / "Figure7_ScaleAwareness_vs_Retrieval_FINAL_clean_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure7_ScaleAwareness_vs_Retrieval_FINAL_clean_8PFM.pdf"
)

OUT_TIFF = (
    FIG_ROOT
    / "Figure7_ScaleAwareness_vs_Retrieval_FINAL_clean_8PFM_600dpi.tiff"
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


COLUMN_MAP = {
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


METRICS = [
    "CKA",
    "Retention",
    "Retrieval",
    "NPS",
]


PANEL_INFO = {
    "CKA": {
        "panel": "A",
        "title": "Global alignment (CKA)",
        "xlabel": "Global alignment (CKA)",
    },

    "Retention": {
        "panel": "B",
        "title": "Relative retention",
        "xlabel": "Relative retention",
    },

    "Retrieval": {
        "panel": "C",
        "title": "Cross-scale retrieval similarity",
        "xlabel": "Cross-scale retrieval similarity",
    },

    "NPS": {
        "panel": "D",
        "title": "Neighborhood preservation (NPS)",
        "xlabel": "Neighborhood Preservation Score",
    },
}


# ============================================================
# FROZEN ANALYSIS-19 STATISTICS
# ============================================================

STATS = {
    "CKA": {
        "rho": 0.428571,
        "p_exact": 0.299206,
        "p_holm": 1.000000,
    },

    "Retention": {
        "rho": 0.166667,
        "p_exact": 0.703323,
        "p_holm": 1.000000,
    },

    "Retrieval": {
        "rho": -0.095238,
        "p_exact": 0.840129,
        "p_holm": 1.000000,
    },

    "NPS": {
        "rho": -0.214286,
        "p_exact": 0.619097,
        "p_holm": 1.000000,
    },
}


# ============================================================
# FROZEN RAW VALUES
# ============================================================

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


# ============================================================
# COLORS
# ============================================================

MODEL_COLOR = {
    "Phikon": "#4C78A8",
    "UNI": "#F58518",
    "Virchow": "#54A24B",
    "Virchow2": "#E45756",
    "UNI2": "#8F63B8",
    "Midnight": "#9D755D",
    "GigaPath": "#E377C2",
    "GigaPath-Flash": "#7F7F7F",
}


# ============================================================
# DIRECT-LABEL OFFSETS
#
# Separate offsets for each panel because relative
# point positions differ across metrics.
#
# values are in matplotlib "offset points"
# ============================================================

LABEL_OFFSET = {

    "CKA": {
        "Phikon": (7, 7),
        "UNI": (7, -11),
        "Virchow": (7, -11),
        "Virchow2": (7, 7),
        "UNI2": (7, 7),
        "Midnight": (7, 7),
        "GigaPath": (7, 7),
        "GigaPath-Flash": (7, -11),
    },

    "Retention": {
        "Phikon": (7, 7),
        "UNI": (7, -11),
        "Virchow": (7, -11),
        "Virchow2": (7, 7),
        "UNI2": (7, 7),
        "Midnight": (7, 7),
        "GigaPath": (7, 7),
        "GigaPath-Flash": (7, -11),
    },

    "Retrieval": {
        "Phikon": (7, 7),
        "UNI": (7, -11),
        "Virchow": (7, -11),
        "Virchow2": (7, 7),
        "UNI2": (7, 7),

        # Move rightward rather than above the
        # statistics region.
        "Midnight": (8, -2),

        "GigaPath": (7, 7),
        "GigaPath-Flash": (7, -11),
    },

    "NPS": {
        "Phikon": (7, 7),
        "UNI": (7, -11),
        "Virchow": (7, -11),
        "Virchow2": (7, 7),
        "UNI2": (7, 7),

        # avoid subtitle area
        "Midnight": (8, -2),

        "GigaPath": (7, 7),
        "GigaPath-Flash": (7, -11),
    },
}


# ============================================================
# HELPERS
# ============================================================

def clean_model(x):

    s = (
        str(x)
        .strip()
        .lower()
        .replace("_", "-")
    )

    aliases = {
        "phikon":
            "Phikon",

        "uni":
            "UNI",

        "virchow":
            "Virchow",

        "virchow2":
            "Virchow2",

        "uni2":
            "UNI2",

        "midnight":
            "Midnight",

        "gigapath":
            "GigaPath",

        "gigapath-flash":
            "GigaPath-Flash",
    }

    return aliases.get(
        s,
        str(x).strip(),
    )


# ============================================================
# LOAD
# ============================================================

if not RAW_FILE.exists():

    raise FileNotFoundError(
        f"Missing:\n{RAW_FILE}"
    )


df = pd.read_csv(
    RAW_FILE
)


df = df.rename(
    columns=COLUMN_MAP
)


if "model_display" in df.columns:

    model_col = "model_display"

elif "model" in df.columns:

    model_col = "model"

else:

    raise RuntimeError(
        "Could not find model column."
    )


df[
    "model_clean"
] = df[
    model_col
].map(
    clean_model
)


required_metrics = [
    "CKA",
    "Retention",
    "Retrieval",
    "Recall@1",
    "NPS",
]


for metric in required_metrics:

    if metric not in df.columns:

        raise RuntimeError(
            f"Missing metric: {metric}\n"
            f"Columns: {list(df.columns)}"
        )


data = (
    df[
        [
            "model_clean"
        ]
        +
        required_metrics
    ]
    .drop_duplicates(
        subset=[
            "model_clean"
        ]
    )
    .rename(
        columns={
            "model_clean":
                "model"
        }
    )
    .set_index(
        "model"
    )
    .reindex(
        MODELS
    )
)


if data.isna().any().any():

    raise RuntimeError(
        "Figure-7 matrix contains NaN."
    )


data.to_csv(
    OUT_DATA
)


# ============================================================
# FROZEN VALUE AUDIT
# ============================================================

audit_lines = []

audit_lines.append(
    f"Input: {RAW_FILE}"
)

audit_lines.append(
    ""
)

audit_lines.append(
    "Frozen raw-value audit:"
)


max_delta = 0.0


for model in MODELS:

    for metric in required_metrics:

        observed = float(
            data.loc[
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

        delta = abs(
            observed
            -
            expected
        )

        max_delta = max(
            max_delta,
            delta,
        )


audit_lines.append(
    f"Max |raw delta| = "
    f"{max_delta:.8f}"
)


if max_delta > 5e-4:

    raise RuntimeError(
        f"Frozen raw-value audit failed: "
        f"{max_delta}"
    )


# ============================================================
# SAVE FROZEN STATISTICS
# ============================================================

stats_rows = []


for metric in METRICS:

    stats_rows.append({
        "metric":
            metric,

        "n_pfms":
            8,

        "spearman_rho":
            STATS[
                metric
            ][
                "rho"
            ],

        "exact_two_sided_p":
            STATS[
                metric
            ][
                "p_exact"
            ],

        "holm_p":
            STATS[
                metric
            ][
                "p_holm"
            ],

        "significant_holm_0_05":
            False,
    })


stats_df = pd.DataFrame(
    stats_rows
)


stats_df.to_csv(
    OUT_STATS,
    index=False,
)


audit_lines.append(
    ""
)

audit_lines.append(
    "Frozen Analysis-19 statistics:"
)


for _, row in stats_df.iterrows():

    audit_lines.append(
        f"{row['metric']:10s} "
        f"rho={row['spearman_rho']:+.6f} "
        f"exact_p={row['exact_two_sided_p']:.6f} "
        f"Holm={row['holm_p']:.6f}"
    )


OUT_AUDIT.write_text(
    "\n".join(
        audit_lines
    ),
    encoding="utf-8",
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
        8.5,

    "ytick.labelsize":
        8.5,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        12.9,
        7.25,
    ),
    facecolor="white",
)


gs = fig.add_gridspec(
    2,
    2,

    hspace=0.45,

    wspace=0.28,
)


axes = {
    "CKA":
        fig.add_subplot(
            gs[
                0,
                0,
            ]
        ),

    "Retention":
        fig.add_subplot(
            gs[
                0,
                1,
            ]
        ),

    "Retrieval":
        fig.add_subplot(
            gs[
                1,
                0,
            ]
        ),

    "NPS":
        fig.add_subplot(
            gs[
                1,
                1,
            ]
        ),
}


# ============================================================
# COMMON Y RANGE
# ============================================================

YMIN = 0.58
YMAX = 0.87

YTICKS = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]


# ============================================================
# DRAW PANELS
# ============================================================

for metric in METRICS:

    ax = axes[
        metric
    ]


    info = PANEL_INFO[
        metric
    ]


    x = data[
        metric
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Main panel title
    # --------------------------------------------------------

    ax.set_title(
        (
            f"{info['panel']}   "
            f"{info['title']}"
        ),

        loc="left",

        fontweight="bold",

        pad=23,
    )


    # --------------------------------------------------------
    # Statistical subtitle OUTSIDE plotting region
    # --------------------------------------------------------

    rho = STATS[
        metric
    ][
        "rho"
    ]

    p_exact = STATS[
        metric
    ][
        "p_exact"
    ]

    p_holm = STATS[
        metric
    ][
        "p_holm"
    ]


    subtitle = (
        f"ρ = {rho:.3f}"
        f"  ·  exact p = {p_exact:.3f}"
        f"  ·  Holm p = {p_holm:.3f}"
    )


    ax.text(
        0.00,
        1.015,

        subtitle,

        transform=ax.transAxes,

        ha="left",

        va="bottom",

        fontsize=8.3,

        color="0.38",
    )


    # --------------------------------------------------------
    # PFM points + direct labels
    # --------------------------------------------------------

    for model in MODELS:

        xv = float(
            data.loc[
                model,
                metric
            ]
        )

        yv = float(
            data.loc[
                model,
                "Recall@1"
            ]
        )


        ax.scatter(
            xv,
            yv,

            s=59,

            facecolor=MODEL_COLOR[
                model
            ],

            edgecolor="white",

            linewidth=0.7,

            zorder=3,
        )


        dx, dy = (
            LABEL_OFFSET[
                metric
            ][
                model
            ]
        )


        txt = ax.annotate(
            model,

            (
                xv,
                yv,
            ),

            xytext=(
                dx,
                dy,
            ),

            textcoords="offset points",

            ha="left",

            va="center",

            fontsize=7.7,

            fontweight="bold",

            color=MODEL_COLOR[
                model
            ],

            zorder=4,
        )


        # White halo improves legibility over grid lines.
        txt.set_path_effects([
            pe.withStroke(
                linewidth=2.2,
                foreground="white",
            )
        ])


    # --------------------------------------------------------
    # X range with label room
    # --------------------------------------------------------

    xmin = float(
        x.min()
    )

    xmax = float(
        x.max()
    )

    xrange = (
        xmax
        -
        xmin
    )


    ax.set_xlim(
        xmin
        -
        0.08
        *
        xrange,

        xmax
        +
        0.18
        *
        xrange,
    )


    # --------------------------------------------------------
    # Shared Y
    # --------------------------------------------------------

    ax.set_ylim(
        YMIN,
        YMAX,
    )

    ax.set_yticks(
        YTICKS
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        info[
            "xlabel"
        ]
    )


    # Only left column gets repeated y title.
    if metric in [
        "CKA",
        "Retrieval",
    ]:

        ax.set_ylabel(
            "Cancer retrieval Recall@1"
        )

    else:

        ax.set_ylabel(
            ""
        )


    # --------------------------------------------------------
    # publication-style grid/spines
    # --------------------------------------------------------

    ax.grid(
        alpha=0.075,

        linewidth=0.65,
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


# ============================================================
# SMALL GLOBAL ANALYSIS-UNIT NOTE
# ============================================================

fig.text(
    0.965,
    0.018,

    "Analysis unit: PFM (n = 8)",

    ha="right",

    va="bottom",

    fontsize=7.8,

    color="0.45",
)


# ============================================================
# GLOBAL LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.92,
    bottom=0.09,
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
    "FIGURE 7 FINAL CLEAN"
)

print(
    "=" * 105
)


print(
    "\nModel-level input:"
)

print(
    data
    .round(4)
    .to_string()
)


print(
    "\nFrozen Analysis-19 associations:"
)

print(
    stats_df[
        [
            "metric",
            "spearman_rho",
            "exact_two_sided_p",
            "holm_p",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print(
    f"\nMax frozen raw-value audit delta: "
    f"{max_delta:.8f}"
)


print(
    "\nConclusion:"
)

print(
    "None of the four representation-level "
    "scale-awareness metrics showed a statistically "
    "supported monotonic association with conventional "
    "cancer retrieval Recall@1."
)


print(
    "\nOutputs:"
)

for path in [
    OUT_DATA,
    OUT_STATS,
    OUT_AUDIT,
    OUT_PNG,
    OUT_PDF,
    OUT_TIFF,
]:

    print(
        path
    )
