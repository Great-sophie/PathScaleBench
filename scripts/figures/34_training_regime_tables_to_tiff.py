#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-ready TIFF supplementary tables
============================================

Outputs:
    Table_S1_PFM_training_regime_audit_600dpi.tiff
    Table_S2_Virchow_Virchow2_directional_MGG_600dpi.tiff
    Table_S3_Virchow_Virchow2_multidimensional_600dpi.tiff
    Table_S4_GigaPath_GigaPathFlash_distillation_600dpi.tiff

Resolution:
    600 dpi TIFF, LZW compression.

Source:
    Frozen Analysis 32 outputs.
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "results" / "reference"

PREFIX = "32_training_regime_multiscale_analysis_8pfm"

OUT = ROOT / "publication_tables_tiff"
OUT.mkdir(
    parents=True,
    exist_ok=True
)


TRAINING = (
    ROOT
    / f"{PREFIX}_training_regime_table.csv"
)

DIRECTION = (
    ROOT
    / f"{PREFIX}_virchow_direction_level.csv"
)

VIRCHOW = (
    ROOT
    / f"{PREFIX}_virchow_family_contrast.csv"
)

DISTILL = (
    ROOT
    / f"{PREFIX}_gigapath_distillation_contrast.csv"
)


# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.7,
})


def fmt4(x):
    """Format numeric value to four decimals."""
    try:
        x = float(x)

        if np.isnan(x):
            return ""

        return f"{x:.4f}"

    except Exception:
        return str(x)


def delta4(x):
    """Signed four-decimal formatting."""
    try:
        x = float(x)

        if np.isnan(x):
            return ""

        return f"{x:+.4f}"

    except Exception:
        return str(x)


def clean_bool(x):
    s = str(x).strip().lower()

    if s in ["true", "1", "yes"]:
        return "Yes"

    if s in ["false", "0", "no"]:
        return "No"

    if s in ["nan", "none", ""]:
        return "Not documented"

    return str(x)


def wrap_text(value, width):
    """
    Wrap long cell text while preserving concise numeric cells.
    """

    s = str(value)

    if len(s) <= width:
        return s

    return "\n".join(
        textwrap.wrap(
            s,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def save_600dpi_tiff(fig, outpath):
    """
    Save via temporary PNG and rewrite as LZW TIFF
    to ensure reliable 600-dpi metadata.
    """

    tmp = outpath.with_suffix(".tmp.png")

    fig.savefig(
        tmp,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close(fig)

    img = Image.open(tmp)

    if img.mode == "RGBA":
        bg = Image.new(
            "RGB",
            img.size,
            "white"
        )

        bg.paste(
            img,
            mask=img.getchannel("A")
        )

        img = bg

    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.save(
        outpath,
        format="TIFF",
        dpi=(600, 600),
        compression="tiff_lzw",
    )

    tmp.unlink()

    print(
        f"[DONE] {outpath.name} "
        f"| {img.size[0]}×{img.size[1]} px "
        f"| 600 dpi"
    )


def render_table(
    df,
    title,
    footnotes,
    outpath,
    col_widths=None,
    wrap_widths=None,
    figsize=(13, 5),
    font_size=8.2,
    first_col_bold=False,
    row_height_scale=1.45,
):
    """
    Render publication-style table.

    White background.
    Black text.
    Light horizontal rules.
    No decorative colors.
    """

    plot_df = df.copy()

    # Wrap selected columns
    if wrap_widths is not None:

        for col, width in wrap_widths.items():

            if col in plot_df.columns:

                plot_df[col] = (
                    plot_df[col]
                    .astype(str)
                    .map(
                        lambda x:
                            wrap_text(
                                x,
                                width
                            )
                    )
                )


    fig = plt.figure(
        figsize=figsize,
        facecolor="white"
    )

    ax = fig.add_axes([
        0.025,
        0.14,
        0.95,
        0.76,
    ])

    ax.axis("off")


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    fig.text(
        0.025,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    table = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[
            0,
            0,
            1,
            1,
        ],
    )


    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    table.scale(
        1.0,
        row_height_scale
    )


    nrows = len(plot_df)
    ncols = len(plot_df.columns)


    for (r, c), cell in table.get_celld().items():

        # Base styling
        cell.set_facecolor("white")
        cell.set_edgecolor("0.78")
        cell.set_linewidth(0.45)

        # Header
        if r == 0:

            cell.set_text_props(
                weight="bold",
                color="black",
                va="center",
            )

            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)

        else:

            cell.set_text_props(
                color="black",
                va="center",
            )


        # First column left aligned
        if c == 0:

            cell.get_text().set_ha("left")

            if first_col_bold and r > 0:

                cell.set_text_props(
                    weight="bold"
                )


    # Strong bottom border
    for c in range(ncols):

        cell = table[
            nrows,
            c
        ]

        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)


    # --------------------------------------------------------
    # FOOTNOTES
    # --------------------------------------------------------

    note_y = 0.105

    for i, note in enumerate(footnotes):

        fig.text(
            0.025,
            note_y - i * 0.024,
            note,
            ha="left",
            va="top",
            fontsize=7.0,
            color="0.20",
        )


    save_600dpi_tiff(
        fig,
        outpath
    )


# ============================================================
# TABLE S1 — TRAINING REGIME AUDIT
# ============================================================

df = pd.read_csv(
    TRAINING
)


df = df[[
    "model",
    "histological_scale_regime",
    "regime_class",
    "explicit_mixed_magnification",
    "distilled",
    "tcga_pretraining_overlap",
]].copy()


df.columns = [
    "PFM",
    "Histological pretraining\nscale regime",
    "Training-regime\nclassification",
    "Explicit mixed-\nmagnification",
    "Distilled",
    "TCGA in\npretraining",
]


df[
    "Training-regime\nclassification"
] = (
    df[
        "Training-regime\nclassification"
    ]
    .replace({
        "single_nominal_magnification":
            "Single nominal magnification",

        "explicit_mixed_magnification":
            "Explicit mixed magnification",

        "not_sufficiently_documented":
            "Not sufficiently documented",

        "distilled_family":
            "Distilled family",
    })
)


for c in [
    "Explicit mixed-\nmagnification",
    "Distilled",
    "TCGA in\npretraining",
]:

    df[c] = df[c].map(
        clean_bool
    )


render_table(
    df=df,

    title=(
        "Table S1. Pretraining-regime audit of the eight "
        "pathology foundation models evaluated in PathScaleBench."
    ),

    footnotes=[
        (
            "Histological multi-magnification exposure refers to "
            "explicit sampling at different physical resolutions "
            "or nominal magnifications during pretraining."
        ),
        (
            "Generic self-supervised multi-crop augmentation was "
            "not classified as histological multi-magnification training."
        ),
        (
            "Models without sufficiently explicit public documentation "
            "were conservatively labelled 'Not sufficiently documented'."
        ),
    ],

    outpath=(
        OUT
        / "Table_S1_PFM_training_regime_audit_600dpi.tiff"
    ),

    col_widths=[
        0.11,
        0.24,
        0.24,
        0.16,
        0.10,
        0.12,
    ],

    wrap_widths={
        "Histological pretraining\nscale regime":
            35,

        "Training-regime\nclassification":
            31,
    },

    figsize=(
        13.5,
        5.6
    ),

    font_size=8.0,

    first_col_bold=True,

    row_height_scale=1.65,
)


# ============================================================
# TABLE S2 — DIRECTIONAL MGG
# ============================================================

df = pd.read_csv(
    DIRECTION
)


df[
    "direction"
] = (
    df[
        "direction"
    ]
    .str.replace(
        "40x",
        "40×",
        regex=False
    )
    .str.replace(
        "10x",
        "10×",
        regex=False
    )
    .str.replace(
        "2.5x",
        "2.5×",
        regex=False
    )
    .str.replace(
        "->",
        "→",
        regex=False
    )
)


df[
    "Virchow_MGG"
] = df[
    "Virchow_MGG"
].map(fmt4)

df[
    "Virchow2_MGG"
] = df[
    "Virchow2_MGG"
].map(fmt4)

df[
    "Virchow2_minus_Virchow"
] = df[
    "Virchow2_minus_Virchow"
].map(delta4)


df[
    "Virchow2_lower_MGG"
] = df[
    "Virchow2_lower_MGG"
].map(
    clean_bool
)


df.columns = [
    "Source→target shift",
    "Virchow MGG",
    "Virchow2 MGG",
    "Δ MGG\n(Virchow2−Virchow)",
    "Lower MGG\nwith Virchow2",
]


# Add mean row
mean_row = pd.DataFrame([{
    "Source→target shift":
        "Mean across six shifts",

    "Virchow MGG":
        "0.4321",

    "Virchow2 MGG":
        "0.1812",

    "Δ MGG\n(Virchow2−Virchow)":
        "−0.2508",

    "Lower MGG\nwith Virchow2":
        "6/6 directions",
}])


df = pd.concat(
    [
        df,
        mean_row
    ],
    ignore_index=True
)


render_table(
    df=df,

    title=(
        "Table S2. Family-matched comparison of magnification "
        "generalization gaps between Virchow and Virchow2."
    ),

    footnotes=[
        (
            "MGG, magnification generalization gap; lower values "
            "indicate greater robustness to magnification shift."
        ),
        (
            "Virchow2 reduced mean MGG from 0.4321 to 0.1812 "
            "(58.1% relative reduction) and had lower MGG in all six directions."
        ),
        (
            "Two-sided exact paired sign-flip p=0.03125; "
            "exact paired Wilcoxon signed-rank p=0.03125."
        ),
    ],

    outpath=(
        OUT
        / "Table_S2_Virchow_Virchow2_directional_MGG_600dpi.tiff"
    ),

    col_widths=[
        0.24,
        0.17,
        0.17,
        0.23,
        0.19,
    ],

    figsize=(
        10.8,
        5.0
    ),

    font_size=8.6,

    first_col_bold=False,

    row_height_scale=1.55,
)


# ============================================================
# TABLE S3 — MULTIDIMENSIONAL VIRCHOW CONTRAST
# ============================================================

df = pd.read_csv(
    VIRCHOW
)


METRIC_NAME = {
    "CKA":
        "Global alignment (CKA)",

    "Retention":
        "Relative retention",

    "Retrieval":
        "Cross-scale retrieval similarity",

    "Recall@1":
        "Biological identity Recall@1",

    "NPS":
        "Neighborhood Preservation Score",

    "mean_MGG":
        "Mean magnification generalization gap",

    "shift_robustness":
        "Probabilistic shift robustness",

    "mean_within_scale_BA":
        "Mean within-scale balanced accuracy",

    "mean_cross_scale_BA":
        "Mean cross-scale balanced accuracy",
}


df = df[
    df[
        "metric"
    ].isin(
        METRIC_NAME
    )
].copy()


df[
    "metric"
] = df[
    "metric"
].map(
    METRIC_NAME
)


df[
    "Virchow"
] = df[
    "Virchow"
].map(fmt4)

df[
    "Virchow2"
] = df[
    "Virchow2"
].map(fmt4)

df[
    "Virchow2_minus_Virchow"
] = df[
    "Virchow2_minus_Virchow"
].map(delta4)


df = df[[
    "metric",
    "Virchow",
    "Virchow2",
    "Virchow2_minus_Virchow",
]]


df.columns = [
    "Metric",
    "Virchow",
    "Virchow2",
    "Δ (Virchow2−Virchow)",
]


render_table(
    df=df,

    title=(
        "Table S3. Multidimensional representation and robustness "
        "comparison between Virchow and Virchow2."
    ),

    footnotes=[
        (
            "Virchow was pretrained at a single nominal 20×/0.5-mpp "
            "resolution; Virchow2 used explicitly documented mixed-magnification exposure."
        ),
        (
            "Virchow2 improved cross-scale retrieval, Recall@1 and "
            "magnification-shift robustness but did not uniformly increase all representation metrics."
        ),
        (
            "These values represent a family-matched observational comparison "
            "and should not be interpreted as a causal estimate of the training regime."
        ),
    ],

    outpath=(
        OUT
        / "Table_S3_Virchow_Virchow2_multidimensional_600dpi.tiff"
    ),

    col_widths=[
        0.49,
        0.15,
        0.15,
        0.21,
    ],

    wrap_widths={
        "Metric":
            43,
    },

    figsize=(
        10.5,
        5.3
    ),

    font_size=8.6,

    first_col_bold=False,

    row_height_scale=1.55,
)


# ============================================================
# TABLE S4 — DISTILLATION CONTRAST
# ============================================================

df = pd.read_csv(
    DISTILL
)


df = df[
    df[
        "metric"
    ].isin(
        METRIC_NAME
    )
].copy()


df[
    "metric"
] = df[
    "metric"
].map(
    METRIC_NAME
)


df[
    "GigaPath"
] = df[
    "GigaPath"
].map(fmt4)

df[
    "GigaPath-Flash"
] = df[
    "GigaPath-Flash"
].map(fmt4)

df[
    "Flash_minus_GigaPath"
] = df[
    "Flash_minus_GigaPath"
].map(delta4)


df = df[[
    "metric",
    "GigaPath",
    "GigaPath-Flash",
    "Flash_minus_GigaPath",
]]


df.columns = [
    "Metric",
    "GigaPath",
    "GigaPath-Flash",
    "Δ (Flash−GigaPath)",
]


render_table(
    df=df,

    title=(
        "Table S4. Descriptive within-family comparison of "
        "GigaPath and the distilled GigaPath-Flash model."
    ),

    footnotes=[
        (
            "This teacher–student comparison is descriptive because only "
            "one directly comparable distilled model family was available."
        ),
        (
            "Lower MGG indicates greater robustness, whereas higher "
            "probabilistic shift robustness indicates greater robustness."
        ),
        (
            "Distillation altered rather than uniformly preserved "
            "cross-scale representation behavior."
        ),
    ],

    outpath=(
        OUT
        / "Table_S4_GigaPath_GigaPathFlash_distillation_600dpi.tiff"
    ),

    col_widths=[
        0.49,
        0.16,
        0.18,
        0.19,
    ],

    wrap_widths={
        "Metric":
            43,
    },

    figsize=(
        10.5,
        4.8
    ),

    font_size=8.6,

    first_col_bold=False,

    row_height_scale=1.55,
)


# ============================================================
# FINAL AUDIT
# ============================================================

print()
print("=" * 88)
print("600-DPI TIFF PUBLICATION TABLES COMPLETE")
print("=" * 88)

for p in sorted(
    OUT.glob(
        "*.tiff"
    )
):
    img = Image.open(p)

    print(
        f"{p.name:65s} "
        f"{img.size[0]:5d} × {img.size[1]:5d} px "
        f"| dpi={img.info.get('dpi')}"
    )

print()
print("Output directory:")
print(OUT)
