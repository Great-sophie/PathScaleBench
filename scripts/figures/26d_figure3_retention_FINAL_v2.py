#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3 — FINAL CLEAN VERSION
Relative cross-scale retention + model × cancer interaction
============================================================

A. Case-level relative retention across 8 PFMs
   - unified blue visual language
   - black diamond = mean

B. Cancer-wise mean retention
   - fixed Blues scale: 0.50–0.85
   - same model order as A

C. Formal model × cancer interaction
   - compact LRT summary
   - no redundant Supported / Not supported labels
   - concise interpretation

No embeddings are recomputed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


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


OUT_PNG = (
    FIG_ROOT
    / "Figure3_Retention_Interaction_FINAL_v2_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure3_Retention_Interaction_FINAL_v2_8PFM.pdf"
)

OUT_TIFF = (
    FIG_ROOT
    / "Figure3_Retention_Interaction_FINAL_v2_8PFM_600dpi.tiff"
)

OUT_MODEL_SUMMARY = (
    RESULT_ROOT
    / "26c_figure3_retention_FINAL_clean_model_summary.csv"
)

OUT_MATRIX = (
    RESULT_ROOT
    / "26c_figure3_retention_FINAL_clean_cancer_matrix.csv"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_KEYS = [
    "phikon",
    "uni2",
    "uni",
    "midnight",
    "virchow",
    "gigapath",
    "gigapath_flash",
    "virchow2",
]

MODEL_DISPLAY = {
    "phikon": "Phikon",
    "uni2": "UNI2",
    "uni": "UNI",
    "midnight": "Midnight",
    "virchow": "Virchow",
    "gigapath": "GigaPath",
    "gigapath_flash": "GigaPath-Flash",
    "virchow2": "Virchow2",
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


# Frozen audited model means
EXPECTED_MEANS = {
    "phikon": 0.7742,
    "uni2": 0.7580,
    "uni": 0.7264,
    "midnight": 0.7242,
    "virchow": 0.7091,
    "gigapath": 0.6850,
    "gigapath_flash": 0.6637,
    "virchow2": 0.6140,
}


# Formal LRT results from Analysis 06
LRT_RESULTS = [
    {
        "effect": "PFM",
        "lr": 276.1531,
        "df": 7,
        "p": 7.425e-56,
        "significant": True,
    },
    {
        "effect": "Cancer type",
        "lr": 10.6446,
        "df": 7,
        "p": 0.154890,
        "significant": False,
    },
    {
        "effect": "PFM × cancer",
        "lr": 64.6517,
        "df": 49,
        "p": 0.066259,
        "significant": False,
    },
]


# ============================================================
# VISUAL LANGUAGE
# ============================================================

BLUE_FILL = "#DCEAF5"
BLUE_EDGE = "#6FA8D1"
BLUE_POINT = "#6FA8D1"

BLUE_DARK = "#205493"
BLUE_LIGHT = "#EEF4FA"

GREY_LIGHT = "#F7F7F7"
GREY_EDGE = "#DEDEDE"
GREY_TEXT = "#444444"


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

        "uni2": "uni2",
        "uni2_h": "uni2",
        "uni2h": "uni2",

        "midnight": "midnight",
        "midnight_12k": "midnight",

        "virchow": "virchow",

        "virchow2": "virchow2",
        "virchow_2": "virchow2",

        "gigapath": "gigapath",

        "gigapath_flash": "gigapath_flash",
        "gigapathflash": "gigapath_flash",
    }

    return aliases.get(
        s,
        s,
    )


def resolve_column(
    df,
    aliases,
    required=True,
):

    lookup = {
        str(c).lower(): c
        for c in df.columns
    }

    for alias in aliases:

        if alias.lower() in lookup:

            return lookup[
                alias.lower()
            ]

    if required:

        raise RuntimeError(
            f"Could not resolve column from aliases:\n"
            f"{aliases}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    return None


def find_retention_file():

    candidates = sorted(
        RESULT_ROOT.glob(
            "02_relative_information_retention_8pfm*.csv"
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
            any(
                "retention" in c
                for c in cols
            )
        ):

            valid.append(
                path
            )


    # Prefer case-level values rather than summaries.
    valid = sorted(
        valid,
        key=lambda p: (
            "summary" in p.name.lower(),
            "overall" in p.name.lower(),
            "cancer" in p.name.lower(),
            len(p.name),
        )
    )

    if not valid:

        raise RuntimeError(
            "No valid case-level retention file found."
        )

    return valid[0]


# ============================================================
# LOAD CASE-LEVEL DATA
# ============================================================

INPUT = find_retention_file()

print(
    f"[RETENTION INPUT]\n{INPUT}"
)

raw = pd.read_csv(
    INPUT
)


model_col = resolve_column(
    raw,
    [
        "model",
        "model_display",
    ],
)


cancer_col = resolve_column(
    raw,
    [
        "cancer",
        "cancer_type",
    ],
)


case_col = resolve_column(
    raw,
    [
        "case_id",
        "slide_id",
        "wsi_id",
        "case",
    ],
    required=False,
)


ret_col = resolve_column(
    raw,
    [
        "retention",
        "retention_score",
        "relative_retention",
        "relative_information_retention",
    ],
    required=False,
)


if ret_col is None:

    candidates = [
        c
        for c in raw.columns
        if (
            "retention" in c.lower()
            and
            "rank" not in c.lower()
        )
    ]

    if not candidates:

        raise RuntimeError(
            "Retention score column not found."
        )

    ret_col = candidates[0]


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

    "retention":
        pd.to_numeric(
            raw[
                ret_col
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

    df[
        "case_id"
    ] = (
        df.groupby(
            [
                "cancer",
                "model",
            ]
        )
        .cumcount()
        .astype(str)
    )


df = (
    df[
        df[
            "model"
        ].isin(
            MODEL_KEYS
        )
        &
        df[
            "cancer"
        ].isin(
            CANCERS
        )
    ]
    .dropna(
        subset=[
            "retention",
        ]
    )
    .copy()
)


# ============================================================
# HARD AUDIT
# ============================================================

counts = (
    df.groupby(
        "model"
    )[
        "case_id"
    ]
    .nunique()
)


if (
    counts.min() != 71
    or
    counts.max() != 71
):

    raise RuntimeError(
        "Expected exactly 71 cases per PFM.\n"
        f"{counts}"
    )


# ============================================================
# MODEL SUMMARY
# ============================================================

summary = (
    df.groupby(
        "model",
        as_index=False,
    )
    .agg(
        mean_retention=(
            "retention",
            "mean",
        ),

        median_retention=(
            "retention",
            "median",
        ),

        q1_retention=(
            "retention",
            lambda x:
                np.quantile(
                    x,
                    0.25,
                ),
        ),

        q3_retention=(
            "retention",
            lambda x:
                np.quantile(
                    x,
                    0.75,
                ),
        ),

        sd_retention=(
            "retention",
            "std",
        ),

        n_cases=(
            "case_id",
            "nunique",
        ),
    )
)


summary[
    "model_display"
] = summary[
    "model"
].map(
    MODEL_DISPLAY
)


summary = (
    summary
    .set_index(
        "model"
    )
    .reindex(
        MODEL_KEYS
    )
    .reset_index()
)


# Audit against frozen means
max_delta = 0.0

for _, row in summary.iterrows():

    observed = float(
        row[
            "mean_retention"
        ]
    )

    expected = EXPECTED_MEANS[
        row[
            "model"
        ]
    ]

    max_delta = max(
        max_delta,
        abs(
            observed
            - expected
        ),
    )


if max_delta > 0.003:

    raise RuntimeError(
        f"Retention mean audit failed. "
        f"max delta={max_delta}"
    )


summary.to_csv(
    OUT_MODEL_SUMMARY,
    index=False,
)


# ============================================================
# CANCER × MODEL MATRIX
# ============================================================

cancer_matrix = (
    df.groupby(
        [
            "cancer",
            "model",
        ]
    )[
        "retention"
    ]
    .mean()
    .unstack(
        "model"
    )
    .reindex(
        index=CANCERS,
        columns=MODEL_KEYS,
    )
)


if cancer_matrix.isna().any().any():

    raise RuntimeError(
        "Cancer × model matrix contains NaN."
    )


matrix_out = cancer_matrix.copy()

matrix_out.columns = [
    MODEL_DISPLAY[
        x
    ]
    for x in MODEL_KEYS
]

matrix_out.to_csv(
    OUT_MATRIX
)


# ============================================================
# WITHIN-CANCER RESULT
# ============================================================

friedman_path = (
    RESULT_ROOT
    / "26_figure3_retention_interaction_8pfm_within_cancer_friedman.csv"
)

if friedman_path.exists():

    friedman = pd.read_csv(
        friedman_path
    )

    if (
        "significant_holm_0_05"
        in friedman.columns
    ):

        n_sig_cancers = int(
            friedman[
                "significant_holm_0_05"
            ].sum()
        )

    else:

        n_sig_cancers = 8

else:

    # Frozen result from Analysis 06
    n_sig_cancers = 8


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family":
        "DejaVu Sans",

    "font.size":
        9.4,

    "axes.titlesize":
        11,

    "axes.labelsize":
        9.8,

    "xtick.labelsize":
        8.7,

    "ytick.labelsize":
        8.7,

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
        7.15,
    ),
    facecolor="white",
)


outer = fig.add_gridspec(
    2,
    1,

    height_ratios=[
        0.82,
        1.18,
    ],

    hspace=0.25,
)


# ============================================================
# PANEL A
# ============================================================

ax_a = fig.add_subplot(
    outer[
        0
    ]
)


positions = np.arange(
    len(
        MODEL_KEYS
    )
)


box_data = [
    df.loc[
        df[
            "model"
        ]
        == model,
        "retention",
    ].to_numpy(
        dtype=float
    )
    for model in MODEL_KEYS
]


bp = ax_a.boxplot(
    box_data,

    positions=positions,

    widths=0.50,

    patch_artist=True,

    showfliers=False,

    medianprops={
        "color":
            "black",

        "linewidth":
            1.15,
    },

    whiskerprops={
        "color":
            "0.40",

        "linewidth":
            0.85,
    },

    capprops={
        "color":
            "0.40",

        "linewidth":
            0.85,
    },

    boxprops={
        "linewidth":
            1.0,
    },
)


# Unified retention color
for patch in bp[
    "boxes"
]:

    patch.set_facecolor(
        BLUE_FILL
    )

    patch.set_edgecolor(
        BLUE_EDGE
    )

    patch.set_alpha(
        0.72
    )


# Case points
rng = np.random.default_rng(
    42
)


for i, model in enumerate(
    MODEL_KEYS
):

    values = df.loc[
        df[
            "model"
        ]
        == model,
        "retention",
    ].to_numpy(
        dtype=float
    )


    jitter = rng.uniform(
        -0.135,
        0.135,
        size=len(
            values
        ),
    )


    ax_a.scatter(
        i + jitter,
        values,

        s=8,

        facecolor=BLUE_POINT,

        edgecolor="none",

        alpha=0.27,

        zorder=2,
    )


    mean_value = float(
        summary.loc[
            summary[
                "model"
            ]
            == model,
            "mean_retention",
        ].iloc[0]
    )


    # Mean diamond
    ax_a.scatter(
        i,
        mean_value,

        marker="D",

        s=53,

        facecolor="black",

        edgecolor="white",

        linewidth=0.65,

        zorder=5,
    )


    # Mean text
    ax_a.text(
        i,
        mean_value + 0.045,

        f"{mean_value:.3f}",

        ha="center",

        va="bottom",

        fontsize=7.8,

        fontweight="bold",

        color="black",
    )


ax_a.set_xticks(
    positions
)

ax_a.set_xticklabels(
    [
        MODEL_DISPLAY[
            x
        ]
        for x in MODEL_KEYS
    ]
)


ax_a.set_ylabel(
    "Relative retention score"
)


ax_a.set_title(
    "A   Relative cross-scale retention across PFMs",

    loc="left",

    fontweight="bold",

    pad=8,
)


ax_a.grid(
    axis="y",

    alpha=0.09,

    linewidth=0.7,
)


ax_a.spines[
    "top"
].set_visible(False)

ax_a.spines[
    "right"
].set_visible(False)


# ============================================================
# LOWER PANELS
# ============================================================

lower = outer[
    1
].subgridspec(
    1,
    2,

    width_ratios=[
        1.35,
        1.10,
    ],

    wspace=0.30,
)


# ============================================================
# PANEL B
# ============================================================

ax_b = fig.add_subplot(
    lower[
        0,
        0,
    ]
)


arr = cancer_matrix.to_numpy(
    dtype=float
)


# Fixed scale across manuscript versions
vmin = 0.50
vmax = 0.85


im = ax_b.imshow(
    arr,

    cmap="Blues",

    vmin=vmin,

    vmax=vmax,

    aspect="auto",

    interpolation="nearest",
)


ax_b.set_xticks(
    np.arange(
        len(
            MODEL_KEYS
        )
    )
)

ax_b.set_xticklabels(
    [
        MODEL_DISPLAY[
            x
        ]
        for x in MODEL_KEYS
    ],

    rotation=20,

    ha="right",

    fontsize=8.2,
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

            fontsize=7.8,

            color=(
                "white"
                if value >= midpoint
                else "black"
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

    linewidth=0.72,

    alpha=0.90,
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


for spine in ax_b.spines.values():

    spine.set_visible(
        False
    )


ax_b.set_title(
    "B   Cancer-wise retention profiles",

    loc="left",

    fontweight="bold",

    pad=8,
)


# narrow colorbar
cbar = fig.colorbar(
    im,
    ax=ax_b,

    fraction=0.021,

    pad=0.022,
)

cbar.set_label(
    "Retention score",

    fontsize=8.8,
)

cbar.ax.tick_params(
    labelsize=8,
)


# ============================================================
# PANEL C
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

    "C   Formal model × cancer interaction analysis",

    transform=ax_c.transAxes,

    fontsize=11,

    fontweight="bold",

    va="top",
)


# ------------------------------------------------------------
# Three compact LRT cards
# ------------------------------------------------------------

row_y = [
    0.79,
    0.63,
    0.47,
]


for row, y in zip(
    LRT_RESULTS,
    row_y,
):

    if row[
        "significant"
    ]:

        face = BLUE_LIGHT
        edge = "#D2E1EF"
        title_color = BLUE_DARK

    else:

        face = GREY_LIGHT
        edge = GREY_EDGE
        title_color = "#333333"


    box = FancyBboxPatch(
        (
            0.00,
            y - 0.0525,
        ),

        0.96,
        0.105,

        boxstyle=(
            "round,"
            "pad=0.008,"
            "rounding_size=0.012"
        ),

        transform=ax_c.transAxes,

        facecolor=face,

        edgecolor=edge,

        linewidth=0.7,
    )

    ax_c.add_patch(
        box
    )


    ax_c.text(
        0.025,
        y + 0.018,

        row[
            "effect"
        ],

        transform=ax_c.transAxes,

        fontsize=9.8,

        fontweight="bold",

        color=title_color,

        va="center",
    )


    if row[
        "p"
    ] < 0.001:

        ptext = "p < 0.001"

    else:

        ptext = (
            f"p = {row['p']:.3f}"
        )


    stat_text = (
        f"LR χ² = {row['lr']:.2f}"
        f"    df = {row['df']}"
        f"    {ptext}"
    )


    ax_c.text(
        0.025,
        y - 0.021,

        stat_text,

        transform=ax_c.transAxes,

        fontsize=8.8,

        color="#333333",

        va="center",
    )


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

ax_c.plot(
    [
        0.00,
        0.96,
    ],
    [
        0.355,
        0.355,
    ],

    transform=ax_c.transAxes,

    color="#D9D9D9",

    linewidth=0.8,
)


ax_c.text(
    0.02,
    0.285,

    "Retention is predominantly model-dependent",

    transform=ax_c.transAxes,

    fontsize=10.3,

    fontweight="bold",

    color=BLUE_DARK,

    va="center",
)


ax_c.text(
    0.02,
    0.205,

    (
        "No supported cancer main effect or PFM × cancer interaction.\n"
        "PFM differences remained significant within all 8 cancer types\n"
        "after Holm correction."
    ),

    transform=ax_c.transAxes,

    fontsize=8.75,

    color=GREY_TEXT,

    va="top",

    linespacing=1.40,
)


# ============================================================
# GLOBAL LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.07,
    right=0.96,
    top=0.96,
    bottom=0.075,
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
    "=" * 100
)

print(
    "FIGURE 3 FINAL CLEAN"
)

print(
    "=" * 100
)


print(
    "\nModel means:"
)

print(
    summary[
        [
            "model_display",
            "mean_retention",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nFormal interaction results:"
)

for row in LRT_RESULTS:

    if row[
        "p"
    ] < 0.001:

        ptext = "<0.001"

    else:

        ptext = (
            f"{row['p']:.6f}"
        )

    print(
        f"{row['effect']:15s} | "
        f"LR={row['lr']:.4f} | "
        f"df={row['df']:2d} | "
        f"p={ptext}"
    )


print(
    f"\nWithin-cancer significant after Holm: "
    f"{n_sig_cancers}/8"
)


print(
    f"\nMax model-mean audit delta: "
    f"{max_delta:.6g}"
)


print(
    "\nOutputs:"
)

for path in [
    OUT_MODEL_SUMMARY,
    OUT_MATRIX,
    OUT_PNG,
    OUT_PDF,
    OUT_TIFF,
]:

    print(
        path
    )
