#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4
Cross-scale retrieval similarity + GEE interaction + cancer-wise profiles
==========================================================================

A. Model-wise cross-scale retrieval similarity
   - 40x <-> 10x
   - 10x <-> 2.5x
   - 40x <-> 2.5x
   - black diamond = overall mean across 3 scale pairs

B. Cancer-wise retrieval similarity
   - case-level mean across 3 scale pairs
   - 8 cancers x 8 PFMs
   - descriptive heatmap

C. Formal GEE interaction analysis
   - Gaussian identity
   - exchangeable working correlation
   - case-clustered robust sandwich inference
   - PFM
   - Scale pair
   - PFM x Scale pair

No embedding recomputation.
"""

from __future__ import annotations

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


PREFIX = "27_figure4_retrieval_gee_cancerwise_FINAL"

OUT_MODEL_PAIR = (
    RESULT_ROOT
    / f"{PREFIX}_model_pair_summary.csv"
)

OUT_MODEL_OVERALL = (
    RESULT_ROOT
    / f"{PREFIX}_model_overall_summary.csv"
)

OUT_CANCER_MATRIX = (
    RESULT_ROOT
    / f"{PREFIX}_cancer_model_matrix.csv"
)

OUT_AUDIT = (
    RESULT_ROOT
    / f"{PREFIX}_audit.txt"
)

OUT_PNG = (
    FIG_ROOT
    / "Figure4_Retrieval_GEE_Cancerwise_OPTIMIZED_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure4_Retrieval_GEE_Cancerwise_OPTIMIZED_8PFM.pdf"
)

OUT_TIFF = (
    FIG_ROOT
    / "Figure4_Retrieval_GEE_Cancerwise_OPTIMIZED_8PFM_600dpi.tiff"
)


# ============================================================
# CONFIG
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

CANCERS = [
    "BLCA",
    "BRCA",
    "COAD",
    "HNSC",
    "KIRC",
    "LUAD",
    "STAD",
    "UCEC",
]

PAIR_ORDER = [
    "40×↔10×",
    "10×↔2.5×",
    "40×↔2.5×",
]

PAIR_COLOR = {
    "40×↔10×": "#D55E00",
    "10×↔2.5×": "#0072B2",
    "40×↔2.5×": "#009E73",
}

PAIR_MARKER = {
    "40×↔10×": "o",
    "10×↔2.5×": "s",
    "40×↔2.5×": "^",
}


# Frozen audited overall values
EXPECTED_OVERALL = {
    "phikon": 0.4946,
    "uni": 0.3311,
    "virchow": 0.5206,
    "virchow2": 0.6426,
    "uni2": 0.3923,
    "midnight": 0.3602,
    "gigapath": 0.3705,
    "gigapath_flash": 0.5302,
}


# Frozen pair means
EXPECTED_PAIR = {
    ("phikon", "40×↔10×"): 0.6514,
    ("phikon", "10×↔2.5×"): 0.5266,
    ("phikon", "40×↔2.5×"): 0.3059,

    ("uni", "40×↔10×"): 0.4157,
    ("uni", "10×↔2.5×"): 0.3621,
    ("uni", "40×↔2.5×"): 0.2154,

    ("virchow", "40×↔10×"): 0.7134,
    ("virchow", "10×↔2.5×"): 0.4681,
    ("virchow", "40×↔2.5×"): 0.3801,

    ("virchow2", "40×↔10×"): 0.7136,
    ("virchow2", "10×↔2.5×"): 0.6873,
    ("virchow2", "40×↔2.5×"): 0.5269,

    ("uni2", "40×↔10×"): 0.4151,
    ("uni2", "10×↔2.5×"): 0.5339,
    ("uni2", "40×↔2.5×"): 0.2278,

    ("midnight", "40×↔10×"): 0.3546,
    ("midnight", "10×↔2.5×"): 0.4327,
    ("midnight", "40×↔2.5×"): 0.2933,

    ("gigapath", "40×↔10×"): 0.3921,
    ("gigapath", "10×↔2.5×"): 0.5010,
    ("gigapath", "40×↔2.5×"): 0.2184,

    ("gigapath_flash", "40×↔10×"): 0.5619,
    ("gigapath_flash", "10×↔2.5×"): 0.5870,
    ("gigapath_flash", "40×↔2.5×"): 0.4417,
}


# Formal GEE robust Wald results from Analysis 08b
GEE_RESULTS = [
    {
        "effect": "PFM",
        "wald": 9855.1571,
        "df": 7,
        "p": "<0.001",
    },
    {
        "effect": "Scale pair",
        "wald": 3310.0633,
        "df": 2,
        "p": "<0.001",
    },
    {
        "effect": "PFM × scale pair",
        "wald": 4258.1805,
        "df": 14,
        "p": "<0.001",
    },
]

WORKING_CORRELATION = 0.324334


# ============================================================
# HELPERS
# ============================================================

def normalize_model(x):

    s = (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "phikon": "phikon",
        "uni": "uni",
        "virchow": "virchow",

        "virchow2": "virchow2",
        "virchow_2": "virchow2",

        "uni2": "uni2",
        "uni2_h": "uni2",
        "uni2h": "uni2",

        "midnight": "midnight",
        "midnight_12k": "midnight",

        "gigapath": "gigapath",

        "gigapath_flash": "gigapath_flash",
        "gigapathflash": "gigapath_flash",
    }

    return aliases.get(
        s,
        s,
    )


def normalize_pair(x):

    s = (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("×", "x")
        .replace("2.5x", "2p5x")
    )

    if (
        "40x" in s
        and
        "10x" in s
        and
        "2p5x" not in s
    ):
        return "40×↔10×"

    if (
        "10x" in s
        and
        "2p5x" in s
        and
        "40x" not in s
    ):
        return "10×↔2.5×"

    if (
        "40x" in s
        and
        "2p5x" in s
    ):
        return "40×↔2.5×"

    return str(x)


def find_column(
    df,
    candidates,
):

    lookup = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]

    return None


def find_retrieval_file():

    candidates = sorted(
        RESULT_ROOT.glob(
            "07_cross_scale_retrieval_similarity_8pfm*.csv"
        )
    )

    if not candidates:

        candidates = sorted(
            RESULT_ROOT.glob(
                "*retrieval*8pfm*.csv"
            )
        )

    valid = []

    for path in candidates:

        try:

            d = pd.read_csv(
                path,
                nrows=5,
            )

        except Exception:

            continue

        cols = {
            c.lower()
            for c in d.columns
        }

        if (
            (
                "model" in cols
                or
                "model_display" in cols
            )
            and
            "cancer" in cols
            and
            "retrieval_similarity" in cols
            and
            (
                "scale_pair" in cols
                or
                (
                    "scale_a" in cols
                    and
                    "scale_b" in cols
                )
            )
        ):

            valid.append(
                path
            )


    # Prefer per-case / values tables
    valid = sorted(
        valid,
        key=lambda p: (
            "summary" in p.name.lower(),
            "overall" in p.name.lower(),
            "cancer" in p.name.lower(),
            len(p.name),
        ),
    )


    if not valid:

        raise RuntimeError(
            "Could not identify case-level retrieval CSV.\n\n"
            "Run:\n"
            "ls -lh "
            "native_fov/reviewer_analysis/results/"
            "07_cross_scale_retrieval_similarity_8pfm*.csv"
        )

    return valid[0]


# ============================================================
# LOAD
# ============================================================

INPUT = find_retrieval_file()

print(
    f"[RETRIEVAL INPUT]\n{INPUT}"
)

raw = pd.read_csv(
    INPUT
)


model_col = find_column(
    raw,
    [
        "model",
        "model_display",
    ],
)

cancer_col = find_column(
    raw,
    [
        "cancer",
        "cancer_type",
    ],
)

case_col = find_column(
    raw,
    [
        "case_id",
        "slide_id",
        "wsi_id",
        "case",
    ],
)

pair_col = find_column(
    raw,
    [
        "scale_pair",
        "pair",
        "scale_pair_label",
    ],
)

retrieval_col = find_column(
    raw,
    [
        "retrieval_similarity",
        "mean_retrieval_similarity",
    ],
)


if retrieval_col is None:

    raise RuntimeError(
        "retrieval_similarity column not found."
    )


df = pd.DataFrame({
    "model":
        raw[
            model_col
        ].map(
            normalize_model
        ),

    "cancer":
        raw[
            cancer_col
        ].astype(str).str.upper(),

    "retrieval_similarity":
        pd.to_numeric(
            raw[
                retrieval_col
            ],
            errors="coerce",
        ),
})


if case_col is not None:

    df[
        "case_id"
    ] = raw[
        case_col
    ].astype(str)

else:

    raise RuntimeError(
        "Case identifier required."
    )


if pair_col is not None:

    df[
        "scale_pair"
    ] = raw[
        pair_col
    ].map(
        normalize_pair
    )

else:

    sa = find_column(
        raw,
        [
            "scale_a",
        ],
    )

    sb = find_column(
        raw,
        [
            "scale_b",
        ],
    )

    if (
        sa is None
        or
        sb is None
    ):

        raise RuntimeError(
            "No scale-pair information."
        )

    df[
        "scale_pair"
    ] = (
        raw[
            sa
        ].astype(str)
        +
        "↔"
        +
        raw[
            sb
        ].astype(str)
    ).map(
        normalize_pair
    )


df = (
    df[
        df[
            "model"
        ].isin(
            MODELS
        )
        &
        df[
            "cancer"
        ].isin(
            CANCERS
        )
        &
        df[
            "scale_pair"
        ].isin(
            PAIR_ORDER
        )
    ]
    .dropna(
        subset=[
            "retrieval_similarity",
        ]
    )
    .copy()
)


# ============================================================
# HARD AUDIT
# ============================================================

expected_rows = (
    71
    * 8
    * 3
)

if len(
    df
) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"found {len(df)}"
    )


if df[
    "case_id"
].nunique() != 71:

    raise RuntimeError(
        "Expected 71 unique cases."
    )


if df[
    "model"
].nunique() != 8:

    raise RuntimeError(
        "Expected 8 PFMs."
    )


# ============================================================
# MODEL × PAIR SUMMARY
# ============================================================

pair_summary = (
    df.groupby(
        [
            "model",
            "scale_pair",
        ],
        as_index=False,
    )
    .agg(
        mean_retrieval=(
            "retrieval_similarity",
            "mean",
        ),

        sd_retrieval=(
            "retrieval_similarity",
            "std",
        ),

        n_cases=(
            "case_id",
            "nunique",
        ),
    )
)


pair_summary[
    "model_display"
] = pair_summary[
    "model"
].map(
    MODEL_DISPLAY
)


# ============================================================
# CASE-LEVEL OVERALL RETRIEVAL
# ============================================================

case_model = (
    df.groupby(
        [
            "cancer",
            "case_id",
            "model",
        ],
        as_index=False,
    )
    .agg(
        overall_retrieval=(
            "retrieval_similarity",
            "mean",
        )
    )
)


overall = (
    case_model.groupby(
        "model",
        as_index=False,
    )
    .agg(
        mean_retrieval=(
            "overall_retrieval",
            "mean",
        ),

        sd_retrieval=(
            "overall_retrieval",
            "std",
        ),

        median_retrieval=(
            "overall_retrieval",
            "median",
        ),
    )
)


overall[
    "model_display"
] = overall[
    "model"
].map(
    MODEL_DISPLAY
)


overall = overall.sort_values(
    "mean_retrieval",
    ascending=False,
).reset_index(
    drop=True
)


MODEL_ORDER = overall[
    "model"
].tolist()


# ============================================================
# CANCER-WISE MATRIX
# ============================================================

cancer_mean = (
    case_model.groupby(
        [
            "cancer",
            "model",
        ],
        as_index=False,
    )
    .agg(
        mean_retrieval=(
            "overall_retrieval",
            "mean",
        ),

        n_cases=(
            "case_id",
            "nunique",
        ),
    )
)


matrix = (
    cancer_mean.pivot(
        index="cancer",
        columns="model",
        values="mean_retrieval",
    )
    .reindex(
        index=CANCERS,
        columns=MODEL_ORDER,
    )
)


if matrix.isna().any().any():

    raise RuntimeError(
        "Cancer × PFM matrix contains NaN."
    )


# ============================================================
# AUDIT FROZEN VALUES
# ============================================================

audit_lines = []

audit_lines.append(
    f"Input: {INPUT}"
)

audit_lines.append(
    ""
)

audit_lines.append(
    "Overall retrieval audit:"
)


max_overall_delta = 0.0


for _, row in (
    overall.iterrows()
):

    model = row[
        "model"
    ]

    observed = float(
        row[
            "mean_retrieval"
        ]
    )

    expected = EXPECTED_OVERALL[
        model
    ]

    delta = (
        observed
        - expected
    )

    max_overall_delta = max(
        max_overall_delta,
        abs(
            delta
        ),
    )

    audit_lines.append(
        f"{MODEL_DISPLAY[model]:15s} "
        f"observed={observed:.6f} "
        f"expected={expected:.6f} "
        f"delta={delta:+.6f}"
    )


audit_lines.append(
    ""
)

audit_lines.append(
    "Pair-level audit:"
)


max_pair_delta = 0.0


for _, row in (
    pair_summary.iterrows()
):

    key = (
        row[
            "model"
        ],
        row[
            "scale_pair"
        ],
    )

    observed = float(
        row[
            "mean_retrieval"
        ]
    )

    expected = EXPECTED_PAIR[
        key
    ]

    delta = (
        observed
        - expected
    )

    max_pair_delta = max(
        max_pair_delta,
        abs(
            delta
        ),
    )


audit_lines.append(
    f"Max overall audit delta = "
    f"{max_overall_delta:.6g}"
)

audit_lines.append(
    f"Max pair audit delta    = "
    f"{max_pair_delta:.6g}"
)


# Virchow2 cancer-wise dominance
n_cancers_best_virchow2 = 0


for cancer in CANCERS:

    row = matrix.loc[
        cancer
    ]

    if (
        row.idxmax()
        == "virchow2"
    ):

        n_cancers_best_virchow2 += 1


audit_lines.append(
    ""
)

audit_lines.append(
    f"Virchow2 highest cancer-wise mean: "
    f"{n_cancers_best_virchow2}/8 cancers"
)


if max_overall_delta > 0.003:

    raise RuntimeError(
        "Overall retrieval audit failed."
    )


if max_pair_delta > 0.003:

    raise RuntimeError(
        "Pair retrieval audit failed."
    )


pair_summary.to_csv(
    OUT_MODEL_PAIR,
    index=False,
)

overall.to_csv(
    OUT_MODEL_OVERALL,
    index=False,
)


matrix_out = matrix.copy()

matrix_out.columns = [
    MODEL_DISPLAY[
        x
    ]
    for x in matrix_out.columns
]

matrix_out.to_csv(
    OUT_CANCER_MATRIX
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
        8.7,

    "ytick.labelsize":
        8.7,

    "legend.fontsize":
        8.2,

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
        13.4,
        7.7,
    ),
    facecolor="white",
)


outer = fig.add_gridspec(
    2,
    1,
    height_ratios=[
        0.90,
        1.10,
    ],
    hspace=0.38,
)


# ============================================================
# PANEL A
# Model-wise scale-pair profiles
# ============================================================

ax_a = fig.add_subplot(
    outer[
        0
    ]
)


y = np.arange(
    len(
        MODEL_ORDER
    )
)


offset = {
    "40×↔10×": -0.18,
    "10×↔2.5×": 0.00,
    "40×↔2.5×": 0.18,
}


for yi, model in enumerate(
    MODEL_ORDER
):

    sub = pair_summary[
        pair_summary[
            "model"
        ]
        == model
    ]


    values = sub[
        "mean_retrieval"
    ].to_numpy(
        dtype=float
    )


    # light connector indicates model's full pair range
    ax_a.hlines(
        yi,
        values.min(),
        values.max(),

        color="0.82",

        linewidth=1.2,

        zorder=1,
    )


    for pair in PAIR_ORDER:

        row = sub[
            sub[
                "scale_pair"
            ]
            == pair
        ]

        value = float(
            row[
                "mean_retrieval"
            ].iloc[0]
        )


        ax_a.scatter(
            value,
            yi + offset[
                pair
            ],

            s=55,

            marker=PAIR_MARKER[
                pair
            ],

            color=PAIR_COLOR[
                pair
            ],

            edgecolor="white",

            linewidth=0.55,

            zorder=3,

            label=(
                pair
                if yi == 0
                else None
            ),
        )


    overall_value = float(
        overall.loc[
            overall[
                "model"
            ]
            == model,
            "mean_retrieval",
        ].iloc[0]
    )


    ax_a.scatter(
        overall_value,
        yi,

        marker="D",

        s=66,

        facecolor="black",

        edgecolor="white",

        linewidth=0.7,

        zorder=4,

        label=(
            "Overall mean"
            if yi == 0
            else None
        ),
    )


    ax_a.text(
        overall_value + 0.009,
        yi,

        f"{overall_value:.3f}",

        ha="left",

        va="center",

        fontsize=7.9,

        fontweight="bold",
    )


ax_a.set_yticks(
    y
)

ax_a.set_yticklabels(
    [
        MODEL_DISPLAY[
            x
        ]
        for x in MODEL_ORDER
    ]
)


ax_a.invert_yaxis()


ax_a.set_xlim(
    0.15,
    0.76,
)


ax_a.set_xlabel(
    "Mean bidirectional Top-5 cosine retrieval similarity"
)


ax_a.set_title(
    "A   Cross-scale retrieval similarity across PFMs",
    loc="left",
    fontweight="bold",
    pad=8,
)


ax_a.grid(
    axis="x",
    alpha=0.10,
    linewidth=0.7,
)


ax_a.spines[
    "top"
].set_visible(False)

ax_a.spines[
    "right"
].set_visible(False)

ax_a.spines[
    "left"
].set_visible(False)

ax_a.tick_params(
    axis="y",
    length=0,
)


ax_a.legend(
    frameon=False,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    columnspacing=1.35,
    handletextpad=0.45,
    borderaxespad=0,
    fontsize=8.3,
)


# ============================================================
# LOWER PANELS
# ============================================================

lower = outer[
    1
].subgridspec(
    1,
    2,
    width_ratios=[
        1.55,
        1.0,
    ],
    wspace=0.28,
)


# ============================================================
# PANEL B
# Cancer-wise heatmap
# ============================================================

ax_b = fig.add_subplot(
    lower[
        0,
        0,
    ]
)


arr = matrix.to_numpy(
    dtype=float
)


vmin = min(
    0.25,
    np.floor(
        arr.min()
        * 20
    )
    / 20,
)

vmax = max(
    0.70,
    np.ceil(
        arr.max()
        * 20
    )
    / 20,
)


im = ax_b.imshow(
    arr,

    cmap="Greens",

    vmin=vmin,
    vmax=vmax,

    aspect="auto",

    interpolation="nearest",
)


ax_b.set_xticks(
    np.arange(
        len(
            MODEL_ORDER
        )
    )
)

ax_b.set_xticklabels(
    [
        MODEL_DISPLAY[
            x
        ]
        for x in MODEL_ORDER
    ],

    rotation=25,

    ha="right",
)


ax_b.set_yticks(
    np.arange(
        len(
            CANCERS
        )
    )
)

ax_b.set_yticklabels(
    CANCERS
)


midpoint = (
    vmin
    + vmax
) / 2


for r in range(
    arr.shape[
        0
    ]
):

    for c in range(
        arr.shape[
            1
        ]
    ):

        value = arr[
            r,
            c
        ]

        ax_b.text(
            c,
            r,

            f"{value:.2f}",

            ha="center",

            va="center",

            fontsize=7.7,

            color=(
                "white"
                if value > midpoint
                else "black"
            ),

            fontweight=(
                "bold"
                if (
                    MODEL_ORDER[c]
                    == "virchow2"
                )
                else "normal"
            ),
        )


# white cell separators
ax_b.set_xticks(
    np.arange(
        -0.5,
        arr.shape[
            1
        ],
        1,
    ),
    minor=True,
)

ax_b.set_yticks(
    np.arange(
        -0.5,
        arr.shape[
            0
        ],
        1,
    ),
    minor=True,
)

ax_b.grid(
    which="minor",

    color="white",

    linewidth=0.75,

    alpha=0.85,
)


ax_b.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

ax_b.tick_params(
    which="major",
    length=0,
)


for spine in (
    ax_b.spines.values()
):

    spine.set_visible(
        False
    )


ax_b.set_title(
    "B   Cancer-wise retrieval profiles",
    loc="left",
    fontweight="bold",
    pad=8,
)


cbar = fig.colorbar(
    im,
    ax=ax_b,

    fraction=0.028,

    pad=0.025,
)

cbar.set_label(
    "Mean retrieval similarity",
    fontsize=8.8,
)

cbar.ax.tick_params(
    labelsize=8,
)




# Panel-B interpretation moved to figure caption.


# ============================================================
# PANEL C
# Formal GEE summary — compact publication style
# ============================================================

ax_c = fig.add_subplot(
    lower[
        0,
        1,
    ]
)

ax_c.axis(
    "off"
)


ax_c.text(
    0.00,
    1.00,

    "C   GEE model × scale-pair interaction analysis",

    transform=ax_c.transAxes,

    fontsize=11,

    fontweight="bold",

    va="top",
)


# ------------------------------------------------------------
# Compact statistical rows
# Interaction row deliberately emphasized.
# ------------------------------------------------------------

row_y = [
    0.76,
    0.58,
    0.40,
]


for i, (row, yrow) in enumerate(
    zip(
        GEE_RESULTS,
        row_y,
    )
):

    is_interaction = (
        row[
            "effect"
        ]
        == "PFM × scale pair"
    )


    if is_interaction:

        face = "#DDEFE3"
        accent = "#0B6B35"

    else:

        face = "#F3F8F4"
        accent = "#2E6F4E"


    rect = plt.Rectangle(
        (
            0.00,
            yrow - 0.065,
        ),

        0.98,
        0.13,

        transform=ax_c.transAxes,

        facecolor=face,

        edgecolor=(
            "#B9D9C3"
            if is_interaction
            else "none"
        ),

        linewidth=(
            0.9
            if is_interaction
            else 0.0
        ),

        zorder=0,
    )

    ax_c.add_patch(
        rect
    )


    ax_c.text(
        0.025,
        yrow + 0.024,

        row[
            "effect"
        ],

        transform=ax_c.transAxes,

        fontsize=(
            9.7
            if is_interaction
            else 9.3
        ),

        fontweight="bold",

        color=accent,

        va="center",
    )


    stats = (
        f"Robust Wald χ² = {row['wald']:.2f}"
        f"    df = {row['df']}"
        f"    p {row['p']}"
    )


    ax_c.text(
        0.025,
        yrow - 0.027,

        stats,

        transform=ax_c.transAxes,

        fontsize=8.6,

        color="0.25",

        va="center",

        fontweight=(
            "bold"
            if is_interaction
            else "normal"
        ),
    )


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

ax_c.plot(
    [
        0.00,
        0.98,
    ],
    [
        0.27,
        0.27,
    ],

    transform=ax_c.transAxes,

    color="0.84",

    linewidth=0.8,
)


ax_c.text(
    0.02,
    0.19,

    "Scale-pair dependence is strongly model-specific",

    transform=ax_c.transAxes,

    fontsize=10.1,

    fontweight="bold",

    color="#0B6B35",

    va="center",
)


ax_c.text(
    0.02,
    0.105,

    (
        "40×↔2.5× showed the lowest retrieval\n"
        "similarity in all 8 PFMs."
    ),

    transform=ax_c.transAxes,

    fontsize=9.0,

    color="0.28",

    va="top",

    linespacing=1.35,
)


# ============================================================
# GLOBAL LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.07,
    right=0.96,
    top=0.96,
    bottom=0.10,
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
    "\n"
    + "=" * 105
)

print(
    "FIGURE 4 — RETRIEVAL + GEE + CANCER-WISE"
)

print(
    "=" * 105
)


print(
    "\nOverall retrieval similarity:"
)

print(
    overall[
        [
            "model_display",
            "mean_retrieval",
            "median_retrieval",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nModel × scale-pair means:"
)

print(
    pair_summary[
        [
            "model_display",
            "scale_pair",
            "mean_retrieval",
        ]
    ]
    .sort_values(
        [
            "model_display",
            "scale_pair",
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nGEE robust Wald:"
)

for row in GEE_RESULTS:

    print(
        f"{row['effect']:20s} | "
        f"chi2={row['wald']:.4f} | "
        f"df={row['df']:2d} | "
        f"p{row['p']}"
    )


print(
    f"\nVirchow2 highest cancer-wise: "
    f"{n_cancers_best_virchow2}/8"
)


print(
    "\nOutputs:"
)

for path in [
    OUT_MODEL_PAIR,
    OUT_MODEL_OVERALL,
    OUT_CANCER_MATRIX,
    OUT_AUDIT,
    OUT_PNG,
    OUT_PDF,
    OUT_TIFF,
]:

    print(
        path
    )
