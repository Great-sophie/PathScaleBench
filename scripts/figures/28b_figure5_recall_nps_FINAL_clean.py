#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5 — FINAL CLEAN VERSION
Biological identity preservation + neighborhood preservation
================================================================

A. Recall@1 / Recall@5 / Recall@10 across 8 PFMs
B. Cancer-wise Recall@1 heatmap
C. Cross-scale Neighborhood Preservation Score (NPS)
D. Formal inference summary

This is a plotting-only script.
No embeddings or benchmark metrics are recomputed.
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


RECALL_FILE = (
    RESULT_ROOT
    / "28_figure5_recall_nps_recall_summary.csv"
)

RECALL_CANCER_FILE = (
    RESULT_ROOT
    / "28_figure5_recall_nps_cancer_recall1_matrix.csv"
)

NPS_PAIR_FILE = (
    RESULT_ROOT
    / "28_figure5_recall_nps_nps_pair_summary.csv"
)

NPS_OVERALL_FILE = (
    RESULT_ROOT
    / "28_figure5_recall_nps_nps_overall_summary.csv"
)


OUT_PNG = (
    FIG_ROOT
    / "Figure5_Recall_NPS_FINAL_clean_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure5_Recall_NPS_FINAL_clean_8PFM.pdf"
)

OUT_TIFF = (
    FIG_ROOT
    / "Figure5_Recall_NPS_FINAL_clean_8PFM_600dpi.tiff"
)


# ============================================================
# MODEL CONFIG
# ============================================================

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


RECALL_ORDER = [
    "midnight",
    "uni2",
    "virchow2",
    "phikon",
    "gigapath",
    "gigapath_flash",
    "uni",
    "virchow",
]


NPS_ORDER = [
    "phikon",
    "uni",
    "uni2",
    "virchow",
    "gigapath",
    "midnight",
    "gigapath_flash",
    "virchow2",
]


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


# ============================================================
# VISUAL LANGUAGE
# ============================================================

# Biological identity = orange family
ORANGE_DARK = "#B84A00"
ORANGE_MID = "#D55E00"
ORANGE_LIGHT = "#E69F00"
ORANGE_PALE = "#F2C46D"
ORANGE_CARD = "#FFF4E8"


RECALL_COLOR = {
    1: ORANGE_MID,
    5: ORANGE_LIGHT,
    10: ORANGE_PALE,
}

RECALL_MARKER = {
    1: "o",
    5: "s",
    10: "^",
}


# NPS = blue family for overall,
# but scale-pair markers retain manuscript-wide encoding.
BLUE_DARK = "#205493"
BLUE_CARD = "#EEF4FA"

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


# ============================================================
# FORMAL STATISTICS
# ============================================================

RECALL_STATS = [
    {
        "label": "Recall@1",
        "q": 35.7050,
        "df": 7,
        "p_holm": 2.472e-05,
        "significant": True,
    },
    {
        "label": "Recall@5",
        "q": 14.4058,
        "df": 7,
        "p_holm": 0.088834,
        "significant": False,
    },
    {
        "label": "Recall@10",
        "q": 9.2000,
        "df": 7,
        "p_holm": 0.238614,
        "significant": False,
    },
]


NPS_STAT = {
    "chi2": 379.2300,
    "df": 7,
    "p": 6.762e-78,
    "kendall_w": 0.7630,
}


# ============================================================
# LOAD
# ============================================================

for path in [
    RECALL_FILE,
    RECALL_CANCER_FILE,
    NPS_PAIR_FILE,
    NPS_OVERALL_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing input:\n{path}\n\n"
            "Run 28_figure5_recall_nps_FINAL.py first."
        )


recall = pd.read_csv(
    RECALL_FILE
)

recall_cancer = pd.read_csv(
    RECALL_CANCER_FILE,
    index_col=0,
)

nps_pair = pd.read_csv(
    NPS_PAIR_FILE
)

nps_overall = pd.read_csv(
    NPS_OVERALL_FILE
)


# ============================================================
# AUDIT INPUT STRUCTURE
# ============================================================

required_recall = {
    "model",
    "k",
    "recall",
}

missing = (
    required_recall
    - set(
        recall.columns
    )
)

if missing:

    raise RuntimeError(
        f"Recall summary missing columns: {missing}"
    )


required_nps_pair = {
    "model",
    "scale_pair",
    "mean_nps",
}

missing = (
    required_nps_pair
    - set(
        nps_pair.columns
    )
)

if missing:

    raise RuntimeError(
        f"NPS pair summary missing columns: {missing}"
    )


required_nps_overall = {
    "model",
    "overall_nps",
}

missing = (
    required_nps_overall
    - set(
        nps_overall.columns
    )
)

if missing:

    raise RuntimeError(
        f"NPS overall summary missing columns: {missing}"
    )


if recall_cancer.shape != (
    8,
    8,
):

    raise RuntimeError(
        f"Expected 8×8 Recall@1 matrix, "
        f"found {recall_cancer.shape}"
    )


recall_cancer.index = (
    recall_cancer.index
    .astype(str)
    .str.upper()
)


# normalize Recall cancer-matrix column names
column_lookup = {}

for col in recall_cancer.columns:

    s = (
        str(col)
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
        "uni2": "uni2",
        "midnight": "midnight",
        "gigapath": "gigapath",
        "gigapath_flash": "gigapath_flash",
        "gigapathflash": "gigapath_flash",
    }

    key = aliases.get(
        s,
        s,
    )

    column_lookup[
        key
    ] = col


for model in RECALL_ORDER:

    if model not in column_lookup:

        raise RuntimeError(
            f"Recall@1 matrix missing model: "
            f"{model}"
        )


# ============================================================
# PREPARE PANEL A
# ============================================================

recall_lookup = {}

for model in RECALL_ORDER:

    recall_lookup[
        model
    ] = {}

    for k in [
        1,
        5,
        10,
    ]:

        row = recall[
            (
                recall[
                    "model"
                ]
                == model
            )
            &
            (
                recall[
                    "k"
                ]
                == k
            )
        ]

        if len(
            row
        ) != 1:

            raise RuntimeError(
                f"Expected one recall value for "
                f"{model}, K={k}; found {len(row)}"
            )

        recall_lookup[
            model
        ][
            k
        ] = float(
            row[
                "recall"
            ].iloc[0]
        )


# ============================================================
# PREPARE PANEL B
# ============================================================

recall_matrix = pd.DataFrame(
    index=CANCERS,
    columns=RECALL_ORDER,
    dtype=float,
)


for cancer in CANCERS:

    for model in RECALL_ORDER:

        recall_matrix.loc[
            cancer,
            model
        ] = float(
            recall_cancer.loc[
                cancer,
                column_lookup[
                    model
                ],
            ]
        )


if (
    recall_matrix.min().min()
    < 0
    or
    recall_matrix.max().max()
    > 1
):

    raise RuntimeError(
        "Recall@1 matrix outside [0,1]."
    )


# ============================================================
# PREPARE PANEL C
# ============================================================

nps_overall_lookup = {
    row[
        "model"
    ]:
        float(
            row[
                "overall_nps"
            ]
        )
    for _, row in nps_overall.iterrows()
}


for model in NPS_ORDER:

    if model not in nps_overall_lookup:

        raise RuntimeError(
            f"NPS overall missing {model}"
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
        9.5,

    "xtick.labelsize":
        8.5,

    "ytick.labelsize":
        8.5,

    "legend.fontsize":
        8.1,

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
        8.1,
    ),
    facecolor="white",
)


outer = fig.add_gridspec(
    2,
    2,

    height_ratios=[
        0.93,
        1.07,
    ],

    width_ratios=[
        1.48,
        1.0,
    ],

    hspace=0.33,

    wspace=0.30,
)


# ============================================================
# PANEL A — RECALL@K
# ============================================================

ax_a = fig.add_subplot(
    outer[
        0,
        0,
    ]
)


y = np.arange(
    len(
        RECALL_ORDER
    )
)


offset = {
    1: -0.17,
    5: 0.00,
    10: 0.17,
}


for yi, model in enumerate(
    RECALL_ORDER
):

    values = [
        recall_lookup[
            model
        ][
            k
        ]
        for k in [
            1,
            5,
            10,
        ]
    ]


    # subtle connector
    ax_a.hlines(
        yi,

        min(
            values
        ),

        max(
            values
        ),

        color="0.88",

        linewidth=0.9,

        zorder=1,
    )


    for k in [
        1,
        5,
        10,
    ]:

        value = recall_lookup[
            model
        ][
            k
        ]


        ax_a.scatter(
            value,

            yi
            + offset[
                k
            ],

            s=52,

            marker=RECALL_MARKER[
                k
            ],

            facecolor=RECALL_COLOR[
                k
            ],

            edgecolor="white",

            linewidth=0.55,

            zorder=3,

            label=(
                f"Recall@{k}"
                if yi == 0
                else None
            ),
        )


    # Label primary endpoint only
    r1 = recall_lookup[
        model
    ][1]


    ax_a.text(
        r1 - 0.012,

        yi - 0.17,

        f"{r1:.3f}",

        ha="right",

        va="center",

        fontsize=7.7,

        fontweight="bold",

        color=ORANGE_DARK,
    )


ax_a.set_yticks(
    y
)

ax_a.set_yticklabels(
    [
        MODEL_DISPLAY[
            model
        ]
        for model in RECALL_ORDER
    ]
)

ax_a.invert_yaxis()


ax_a.set_xlim(
    0.55,
    1.015,
)


ax_a.set_xlabel(
    "Cancer identity Recall@K"
)


ax_a.set_title(
    "A   Biological identity preservation across PFMs",

    loc="left",

    fontweight="bold",

    pad=8,
)


ax_a.grid(
    axis="x",

    alpha=0.09,

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


# legend outside plot region
ax_a.legend(
    frameon=False,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        -0.16,
    ),

    ncol=3,

    columnspacing=1.3,

    handletextpad=0.4,

    borderaxespad=0,
)


# ============================================================
# PANEL D — FORMAL INFERENCE
# ============================================================

ax_d = fig.add_subplot(
    outer[
        0,
        1,
    ]
)

ax_d.axis(
    "off"
)


ax_d.text(
    0.00,
    1.00,

    "D   Formal inference",

    transform=ax_d.transAxes,

    fontsize=11,

    fontweight="bold",

    va="top",
)


# ------------------------------------------------------------
# Biological identity card
# ------------------------------------------------------------

card1 = FancyBboxPatch(
    (
        0.00,
        0.52,
    ),

    0.97,
    0.36,

    boxstyle=(
        "round,"
        "pad=0.012,"
        "rounding_size=0.015"
    ),

    transform=ax_d.transAxes,

    facecolor=ORANGE_CARD,

    edgecolor="#F0D7BC",

    linewidth=0.8,
)

ax_d.add_patch(
    card1
)


ax_d.text(
    0.035,
    0.82,

    "Biological identity",

    transform=ax_d.transAxes,

    fontsize=9.8,

    fontweight="bold",

    color=ORANGE_DARK,

    va="center",
)


recall_y = [
    0.72,
    0.63,
    0.54,
]


for stat, yy in zip(
    RECALL_STATS,
    recall_y,
):

    if stat[
        "p_holm"
    ] < 0.001:

        ptext = "<0.001"

    else:

        ptext = (
            f"{stat['p_holm']:.3f}"
        )


    color = (
        ORANGE_DARK
        if stat[
            "significant"
        ]
        else "#555555"
    )


    fw = (
        "bold"
        if stat[
            "significant"
        ]
        else "normal"
    )


    ax_d.text(
        0.055,
        yy,

        (
            f"{stat['label']}: "
            f"Q = {stat['q']:.2f}, "
            f"pHolm = {ptext}"
        ),

        transform=ax_d.transAxes,

        fontsize=8.7,

        color=color,

        fontweight=fw,

        va="center",
    )


# ------------------------------------------------------------
# NPS card
# ------------------------------------------------------------

card2 = FancyBboxPatch(
    (
        0.00,
        0.09,
    ),

    0.97,
    0.32,

    boxstyle=(
        "round,"
        "pad=0.012,"
        "rounding_size=0.015"
    ),

    transform=ax_d.transAxes,

    facecolor=BLUE_CARD,

    edgecolor="#D3E1EE",

    linewidth=0.8,
)

ax_d.add_patch(
    card2
)


ax_d.text(
    0.035,
    0.35,

    "Neighborhood preservation",

    transform=ax_d.transAxes,

    fontsize=9.8,

    fontweight="bold",

    color=BLUE_DARK,

    va="center",
)


ax_d.text(
    0.055,
    0.245,

    (
        f"Friedman χ² = "
        f"{NPS_STAT['chi2']:.2f}, "
        f"df = {NPS_STAT['df']}, "
        f"p < 0.001"
    ),

    transform=ax_d.transAxes,

    fontsize=8.7,

    color="#333333",

    va="center",
)


ax_d.text(
    0.055,
    0.155,

    (
        f"Kendall W = "
        f"{NPS_STAT['kendall_w']:.3f}"
    ),

    transform=ax_d.transAxes,

    fontsize=9.0,

    fontweight="bold",

    color=BLUE_DARK,

    va="center",
)


# ============================================================
# PANEL B — CANCER-WISE RECALL@1
# ============================================================

ax_b = fig.add_subplot(
    outer[
        1,
        0,
    ]
)


arr = recall_matrix.to_numpy(
    dtype=float
)


im_b = ax_b.imshow(
    arr,

    cmap="Oranges",

    vmin=0.0,

    vmax=1.0,

    aspect="auto",

    interpolation="nearest",
)


ax_b.set_xticks(
    np.arange(
        len(
            RECALL_ORDER
        )
    )
)

ax_b.set_xticklabels(
    [
        MODEL_DISPLAY[
            model
        ]
        for model in RECALL_ORDER
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

            fontsize=7.6,

            color=(
                "white"
                if value >= 0.58
                else "black"
            ),
        )


# cell separators
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

    alpha=0.9,
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
    "B   Cancer-wise Recall@1 profiles",

    loc="left",

    fontweight="bold",

    pad=8,
)


cbar_b = fig.colorbar(
    im_b,

    ax=ax_b,

    fraction=0.021,

    pad=0.022,
)

cbar_b.set_label(
    "Recall@1",

    fontsize=8.7,
)

cbar_b.ax.tick_params(
    labelsize=8,
)


# ============================================================
# PANEL C — NPS
# ============================================================

ax_c = fig.add_subplot(
    outer[
        1,
        1,
    ]
)


y = np.arange(
    len(
        NPS_ORDER
    )
)


offset_nps = {
    "40×↔10×": -0.16,
    "10×↔2.5×": 0.00,
    "40×↔2.5×": 0.16,
}


for yi, model in enumerate(
    NPS_ORDER
):

    sub = nps_pair[
        nps_pair[
            "model"
        ]
        == model
    ]


    values = sub[
        "mean_nps"
    ].to_numpy(
        dtype=float
    )


    ax_c.hlines(
        yi,

        values.min(),

        values.max(),

        color="0.86",

        linewidth=0.9,

        zorder=1,
    )


    for pair in PAIR_ORDER:

        row = sub[
            sub[
                "scale_pair"
            ]
            == pair
        ]


        if len(
            row
        ) != 1:

            raise RuntimeError(
                f"NPS pair missing: "
                f"{model}/{pair}"
            )


        value = float(
            row[
                "mean_nps"
            ].iloc[0]
        )


        ax_c.scatter(
            value,

            yi
            + offset_nps[
                pair
            ],

            s=45,

            marker=PAIR_MARKER[
                pair
            ],

            facecolor=PAIR_COLOR[
                pair
            ],

            edgecolor="white",

            linewidth=0.5,

            zorder=3,

            label=(
                pair
                if yi == 0
                else None
            ),
        )


    overall_value = (
        nps_overall_lookup[
            model
        ]
    )


    ax_c.scatter(
        overall_value,

        yi,

        marker="D",

        s=57,

        facecolor="black",

        edgecolor="white",

        linewidth=0.65,

        zorder=4,

        label=(
            "Overall mean"
            if yi == 0
            else None
        ),
    )


    ax_c.text(
        overall_value + 0.005,

        yi,

        f"{overall_value:.3f}",

        ha="left",

        va="center",

        fontsize=7.4,

        fontweight="bold",
    )


ax_c.set_yticks(
    y
)

ax_c.set_yticklabels(
    [
        MODEL_DISPLAY[
            model
        ]
        for model in NPS_ORDER
    ]
)


ax_c.invert_yaxis()


ax_c.set_xlim(
    0.08,
    0.34,
)


ax_c.set_xlabel(
    "Neighborhood Preservation Score"
)


ax_c.set_title(
    "C   Cross-scale neighborhood preservation (NPS)",

    loc="left",

    fontweight="bold",

    pad=8,
)


ax_c.grid(
    axis="x",

    alpha=0.09,

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


# legend outside plotting area
ax_c.legend(
    frameon=False,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        -0.18,
    ),

    ncol=2,

    columnspacing=1.0,

    handletextpad=0.35,

    borderaxespad=0,

    fontsize=7.5,
)


# ============================================================
# GLOBAL LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.96,
    top=0.96,
    bottom=0.12,
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
    "FIGURE 5 FINAL CLEAN"
)

print(
    "=" * 100
)


print(
    "\nRecall@K:"
)

for model in RECALL_ORDER:

    print(
        f"{MODEL_DISPLAY[model]:15s} "
        f"R1={recall_lookup[model][1]:.4f} "
        f"R5={recall_lookup[model][5]:.4f} "
        f"R10={recall_lookup[model][10]:.4f}"
    )


print(
    "\nNPS:"
)

for model in NPS_ORDER:

    print(
        f"{MODEL_DISPLAY[model]:15s} "
        f"{nps_overall_lookup[model]:.4f}"
    )


print(
    "\nFormal inference:"
)

for stat in RECALL_STATS:

    if stat[
        "p_holm"
    ] < 0.001:

        ptext = "<0.001"

    else:

        ptext = (
            f"{stat['p_holm']:.6f}"
        )

    print(
        f"{stat['label']:10s} "
        f"Q={stat['q']:.4f} "
        f"Holm p={ptext}"
    )


print(
    f"NPS Friedman chi2="
    f"{NPS_STAT['chi2']:.4f}, "
    f"df={NPS_STAT['df']}, "
    f"p<0.001, "
    f"W={NPS_STAT['kendall_w']:.4f}"
)


print(
    "\nOutputs:"
)

for path in [
    OUT_PNG,
    OUT_PDF,
    OUT_TIFF,
]:

    print(
        path
    )
