#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training-regime / mixed-magnification analysis
==============================================

Primary question:
Does explicit mixed-magnification pretraining correspond to
greater magnification-shift robustness?

Primary family-matched contrast:
    Virchow       = single 20x / 0.5 mpp
    Virchow2      = explicit mixed 5x / 10x / 20x / 40x

Secondary descriptive contrast:
    GigaPath      = teacher
    GigaPath-Flash= distilled student

Statistical unit for Virchow-family MGG test:
    six matched source->target shift directions

No case/cross-validation repeats are treated as independent units.
"""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


RAW_BENCHMARK = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_raw.csv"
)


PREFIX = (
    "32_training_regime_multiscale_analysis_8pfm"
)


OUT_METADATA = (
    RESULT_ROOT
    / f"{PREFIX}_training_regime_table.csv"
)

OUT_METRICS = (
    RESULT_ROOT
    / f"{PREFIX}_model_metrics.csv"
)

OUT_VIRCHOW = (
    RESULT_ROOT
    / f"{PREFIX}_virchow_family_contrast.csv"
)

OUT_VIRCHOW_DIRECTION = (
    RESULT_ROOT
    / f"{PREFIX}_virchow_direction_level.csv"
)

OUT_DISTILL = (
    RESULT_ROOT
    / f"{PREFIX}_gigapath_distillation_contrast.csv"
)

OUT_LOG = (
    RESULT_ROOT
    / f"{PREFIX}_summary.txt"
)


# ============================================================
# TRAINING-REGIME METADATA
#
# IMPORTANT:
# "mixed_magnification" refers to explicitly documented
# histological WSI magnification sampling, NOT DINO multi-crop.
# ============================================================

metadata = pd.DataFrame([
    {
        "model": "Phikon",
        "histological_scale_regime":
            "20x / 0.5 um-per-pixel",
        "regime_class":
            "single_nominal_magnification",
        "explicit_mixed_magnification":
            False,
        "distilled":
            False,
        "tcga_pretraining_overlap":
            True,
        "evidence_note":
            "Source pathology tiles extracted at 20x / 0.5 um-per-pixel."
    },

    {
        "model": "UNI",
        "histological_scale_regime":
            "20x source tiles; 256/512 px FOV variation",
        "regime_class":
            "single_nominal_magnification",
        "explicit_mixed_magnification":
            False,
        "distilled":
            False,
        "tcga_pretraining_overlap":
            False,
        "evidence_note":
            "Training tiles sampled at 20x; different tile sizes change FOV but not nominal WSI magnification."
    },

    {
        "model": "Virchow",
        "histological_scale_regime":
            "20x / 0.5 mpp",
        "regime_class":
            "single_nominal_magnification",
        "explicit_mixed_magnification":
            False,
        "distilled":
            False,
        "tcga_pretraining_overlap":
            False,
        "evidence_note":
            "Official model card states tiles sampled at 0.5 mpp / 20x."
    },

    {
        "model": "Virchow2",
        "histological_scale_regime":
            "5x / 10x / 20x / 40x",
        "regime_class":
            "explicit_mixed_magnification",
        "explicit_mixed_magnification":
            True,
        "distilled":
            False,
        "tcga_pretraining_overlap":
            False,
        "evidence_note":
            "Official model card explicitly states 2.0, 1.0, 0.5, 0.25 mpp sampling."
    },

    {
        "model": "UNI2",
        "histological_scale_regime":
            "not explicitly documented in public model card",
        "regime_class":
            "not_sufficiently_documented",
        "explicit_mixed_magnification":
            np.nan,
        "distilled":
            False,
        "tcga_pretraining_overlap":
            False,
        "evidence_note":
            "Public model card reports >200M tiles / >300k H&E and IHC slides but does not explicitly document mixed WSI magnification sampling."
    },

    {
        "model": "Midnight",
        "histological_scale_regime":
            "not explicitly documented for released Midnight-12k",
        "regime_class":
            "not_sufficiently_documented",
        "explicit_mixed_magnification":
            np.nan,
        "distilled":
            False,
        "tcga_pretraining_overlap":
            True,
        "evidence_note":
            "Released Midnight-12k is trained on TCGA-12k; high-resolution post-training is described for proprietary Midnight-92k/392."
    },

    {
        "model": "GigaPath",
        "histological_scale_regime":
            "20x / 0.5 mpp",
        "regime_class":
            "single_nominal_magnification",
        "explicit_mixed_magnification":
            False,
        "distilled":
            False,
        "tcga_pretraining_overlap":
            False,
        "evidence_note":
            "Original GigaPath tile pretraining uses 256x256 pathology tiles at 20x / 0.5 MPP."
    },

    {
        "model": "GigaPath-Flash",
        "histological_scale_regime":
            "GigaPath-family distilled tile encoder",
        "regime_class":
            "distilled_family",
        "explicit_mixed_magnification":
            np.nan,
        "distilled":
            True,
        "tcga_pretraining_overlap":
            False,
        "evidence_note":
            "ViT-S tile encoder distilled from the original GigaPath ViT-g teacher."
    },
])


metadata.to_csv(
    OUT_METADATA,
    index=False,
)


# ============================================================
# FROZEN ANALYSIS-20 ROBUSTNESS RESULTS
# ============================================================

MGG = {
    "Phikon":
        [0.6170, 0.2333, 0.2260, 0.4302, 0.0590, 0.6206],

    "UNI":
        [0.6215, 0.1876, 0.3104, 0.4099, 0.1378, 0.6083],

    "Virchow":
        [0.6415, 0.3403, 0.3642, 0.4615, 0.1885, 0.5965],

    "Virchow2":
        [0.1112, 0.1844, 0.1041, 0.4293, 0.0411, 0.2174],

    "UNI2":
        [0.0338, 0.2627, 0.1885, 0.5325, 0.1223, 0.2037],

    "Midnight":
        [0.0169, 0.0632, 0.0556, 0.1770, 0.0490, 0.1274],

    "GigaPath":
        [0.3944, 0.4287, 0.2660, 0.4396, 0.0972, 0.5122],

    "GigaPath-Flash":
        [0.5342, 0.4424, 0.3083, 0.4596, 0.1249, 0.4976],
}


DIRECTIONS = [
    "10x->2.5x",
    "10x->40x",
    "2.5x->10x",
    "2.5x->40x",
    "40x->10x",
    "40x->2.5x",
]


SHIFT_ROBUSTNESS = {
    "Phikon": -1.2119,
    "UNI": -1.2275,
    "Virchow": -2.9459,
    "Virchow2": -0.5797,
    "UNI2": -0.5929,
    "Midnight": -0.3188,
    "GigaPath": -1.2885,
    "GigaPath-Flash": -2.0235,
}


# ============================================================
# BALANCED ACCURACY MATRICES
# ============================================================

BA = {

    "Virchow": np.array([
        [0.7252, 0.5367, 0.1287],
        [0.4417, 0.7819, 0.1404],
        [0.1902, 0.2875, 0.6517],
    ]),

    "Virchow2": np.array([
        [0.7859, 0.7448, 0.5686],
        [0.6732, 0.8576, 0.7465],
        [0.4256, 0.7509, 0.8550],
    ]),
}


# ============================================================
# LOAD FIGURE-6 RAW METRICS
# ============================================================

if not RAW_BENCHMARK.exists():

    raise FileNotFoundError(
        RAW_BENCHMARK
    )


raw = pd.read_csv(
    RAW_BENCHMARK
)


raw = raw.rename(
    columns={
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
)


if "model_display" in raw.columns:

    model_col = "model_display"

else:

    model_col = "model"


metric_rows = []


for model in metadata["model"]:

    row = raw[
        raw[
            model_col
        ]
        == model
    ]


    if len(
        row
    ) != 1:

        raise RuntimeError(
            f"Could not resolve {model} in Figure-6 raw table."
        )


    metric_rows.append({
        "model":
            model,

        "CKA":
            float(
                row[
                    "CKA"
                ].iloc[0]
            ),

        "Retention":
            float(
                row[
                    "Retention"
                ].iloc[0]
            ),

        "Retrieval":
            float(
                row[
                    "Retrieval"
                ].iloc[0]
            ),

        "Recall@1":
            float(
                row[
                    "Recall@1"
                ].iloc[0]
            ),

        "NPS":
            float(
                row[
                    "NPS"
                ].iloc[0]
            ),

        "mean_MGG":
            float(
                np.mean(
                    MGG[
                        model
                    ]
                )
            ),

        "shift_robustness":
            SHIFT_ROBUSTNESS[
                model
            ],
    })


metrics = pd.DataFrame(
    metric_rows
)


metrics = metrics.merge(
    metadata,
    on="model",
    how="left",
)


metrics.to_csv(
    OUT_METRICS,
    index=False,
)


# ============================================================
# EXACT SIGN-FLIP TEST
# ============================================================

def exact_sign_flip_test(
    differences,
):

    """
    Two-sided exact paired randomization test.

    Null:
        sign of each paired difference is exchangeable.

    Statistic:
        absolute mean paired difference.
    """

    d = np.asarray(
        differences,
        dtype=float,
    )

    observed = abs(
        float(
            d.mean()
        )
    )


    null = []


    for signs in product(
        [-1.0, 1.0],
        repeat=len(
            d
        ),
    ):

        signs = np.asarray(
            signs
        )

        null.append(
            abs(
                float(
                    np.mean(
                        d
                        *
                        signs
                    )
                )
            )
        )


    null = np.asarray(
        null
    )


    p = float(
        np.mean(
            null
            >=
            observed
            -
            1e-12
        )
    )


    return p


# ============================================================
# PRIMARY FAMILY-MATCHED CONTRAST:
# VIRCHOW -> VIRCHOW2
# ============================================================

v1 = np.asarray(
    MGG[
        "Virchow"
    ]
)

v2 = np.asarray(
    MGG[
        "Virchow2"
    ]
)


diff = (
    v2
    -
    v1
)


signflip_p = exact_sign_flip_test(
    diff
)


wilcox = wilcoxon(
    v2,
    v1,

    alternative="two-sided",

    method="exact",
)


direction_df = pd.DataFrame({
    "direction":
        DIRECTIONS,

    "Virchow_MGG":
        v1,

    "Virchow2_MGG":
        v2,

    "Virchow2_minus_Virchow":
        diff,

    "Virchow2_lower_MGG":
        v2
        <
        v1,
})


direction_df.to_csv(
    OUT_VIRCHOW_DIRECTION,
    index=False,
)


virchow_metrics = metrics.set_index(
    "model"
)


contrast_rows = []


for metric in [
    "CKA",
    "Retention",
    "Retrieval",
    "Recall@1",
    "NPS",
    "mean_MGG",
    "shift_robustness",
]:

    old = float(
        virchow_metrics.loc[
            "Virchow",
            metric
        ]
    )

    new = float(
        virchow_metrics.loc[
            "Virchow2",
            metric
        ]
    )


    contrast_rows.append({
        "metric":
            metric,

        "Virchow":
            old,

        "Virchow2":
            new,

        "Virchow2_minus_Virchow":
            new
            -
            old,
    })


virchow_contrast = pd.DataFrame(
    contrast_rows
)


# Add family-level magnification-specific summaries
diag_v1 = float(
    np.diag(
        BA[
            "Virchow"
        ]
    ).mean()
)

diag_v2 = float(
    np.diag(
        BA[
            "Virchow2"
        ]
    ).mean()
)


mask = ~np.eye(
    3,
    dtype=bool,
)


cross_v1 = float(
    BA[
        "Virchow"
    ][
        mask
    ].mean()
)

cross_v2 = float(
    BA[
        "Virchow2"
    ][
        mask
    ].mean()
)


extra = pd.DataFrame([
    {
        "metric":
            "mean_within_scale_BA",

        "Virchow":
            diag_v1,

        "Virchow2":
            diag_v2,

        "Virchow2_minus_Virchow":
            diag_v2
            -
            diag_v1,
    },

    {
        "metric":
            "mean_cross_scale_BA",

        "Virchow":
            cross_v1,

        "Virchow2":
            cross_v2,

        "Virchow2_minus_Virchow":
            cross_v2
            -
            cross_v1,
    },
])


virchow_contrast = pd.concat(
    [
        virchow_contrast,
        extra,
    ],

    ignore_index=True,
)


virchow_contrast[
    "mgg_exact_signflip_p"
] = np.nan

virchow_contrast[
    "mgg_exact_wilcoxon_p"
] = np.nan


mask_mgg = (
    virchow_contrast[
        "metric"
    ]
    ==
    "mean_MGG"
)


virchow_contrast.loc[
    mask_mgg,
    "mgg_exact_signflip_p"
] = signflip_p


virchow_contrast.loc[
    mask_mgg,
    "mgg_exact_wilcoxon_p"
] = float(
    wilcox.pvalue
)


virchow_contrast.to_csv(
    OUT_VIRCHOW,
    index=False,
)


# ============================================================
# SECONDARY DISTILLATION CONTRAST
# ============================================================

gp = virchow_metrics.loc[
    "GigaPath"
]

gpf = virchow_metrics.loc[
    "GigaPath-Flash"
]


distill_rows = []


for metric in [
    "CKA",
    "Retention",
    "Retrieval",
    "Recall@1",
    "NPS",
    "mean_MGG",
    "shift_robustness",
]:

    a = float(
        gp[
            metric
        ]
    )

    b = float(
        gpf[
            metric
        ]
    )


    distill_rows.append({
        "metric":
            metric,

        "GigaPath":
            a,

        "GigaPath-Flash":
            b,

        "Flash_minus_GigaPath":
            b
            -
            a,
    })


distill = pd.DataFrame(
    distill_rows
)


gp_mgg = np.asarray(
    MGG[
        "GigaPath"
    ]
)

gpf_mgg = np.asarray(
    MGG[
        "GigaPath-Flash"
    ]
)


distill_signflip = exact_sign_flip_test(
    gpf_mgg
    -
    gp_mgg
)


distill_wilcox = wilcoxon(
    gpf_mgg,
    gp_mgg,

    alternative="two-sided",

    method="exact",
)


mask_mgg = (
    distill[
        "metric"
    ]
    ==
    "mean_MGG"
)


distill[
    "mgg_exact_signflip_p"
] = np.nan

distill[
    "mgg_exact_wilcoxon_p"
] = np.nan


distill.loc[
    mask_mgg,
    "mgg_exact_signflip_p"
] = distill_signflip


distill.loc[
    mask_mgg,
    "mgg_exact_wilcoxon_p"
] = float(
    distill_wilcox.pvalue
)


distill.to_csv(
    OUT_DISTILL,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

mgg_v1 = float(
    v1.mean()
)

mgg_v2 = float(
    v2.mean()
)


relative_reduction = (
    1.0
    -
    mgg_v2
    /
    mgg_v1
)


lines = []

lines.append(
    "=" * 100
)

lines.append(
    "TRAINING-REGIME / MIXED-MAGNIFICATION ANALYSIS"
)

lines.append(
    "=" * 100
)


lines.append(
    ""
)

lines.append(
    "PRIMARY FAMILY-MATCHED CONTRAST"
)

lines.append(
    "Virchow = explicit single 20x / 0.5 mpp"
)

lines.append(
    "Virchow2 = explicit mixed 5x / 10x / 20x / 40x"
)


lines.append(
    ""
)

lines.append(
    f"Virchow mean MGG       : "
    f"{mgg_v1:.4f}"
)

lines.append(
    f"Virchow2 mean MGG      : "
    f"{mgg_v2:.4f}"
)

lines.append(
    f"Absolute delta         : "
    f"{mgg_v2 - mgg_v1:+.4f}"
)

lines.append(
    f"Relative MGG reduction : "
    f"{100 * relative_reduction:.1f}%"
)

lines.append(
    f"Lower MGG directions   : "
    f"{int((v2 < v1).sum())}/6"
)

lines.append(
    f"Exact sign-flip p      : "
    f"{signflip_p:.5f}"
)

lines.append(
    f"Exact Wilcoxon p       : "
    f"{wilcox.pvalue:.5f}"
)


lines.append(
    ""
)

lines.append(
    f"Mean within-scale BA:"
)

lines.append(
    f"  Virchow  = {diag_v1:.4f}"
)

lines.append(
    f"  Virchow2 = {diag_v2:.4f}"
)

lines.append(
    f"  Delta    = {diag_v2-diag_v1:+.4f}"
)


lines.append(
    ""
)

lines.append(
    f"Mean cross-scale BA:"
)

lines.append(
    f"  Virchow  = {cross_v1:.4f}"
)

lines.append(
    f"  Virchow2 = {cross_v2:.4f}"
)

lines.append(
    f"  Delta    = {cross_v2-cross_v1:+.4f}"
)


lines.append(
    ""
)

lines.append(
    "REPRESENTATION METRIC CHANGES"
)


for _, row in virchow_contrast.iloc[:7].iterrows():

    lines.append(
        f"{row['metric']:18s} "
        f"{row['Virchow']:.4f} -> "
        f"{row['Virchow2']:.4f} "
        f"(delta={row['Virchow2_minus_Virchow']:+.4f})"
    )


lines.append(
    ""
)

lines.append(
    "SECONDARY DISTILLATION CONTRAST"
)

lines.append(
    f"GigaPath mean MGG       = "
    f"{gp_mgg.mean():.4f}"
)

lines.append(
    f"GigaPath-Flash mean MGG = "
    f"{gpf_mgg.mean():.4f}"
)

lines.append(
    f"Exact paired p           = "
    f"{distill_signflip:.5f}"
)


OUT_LOG.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8",
)


print(
    "\n".join(
        lines
    )
)


print(
    "\nOutputs:"
)

for path in [
    OUT_METADATA,
    OUT_METRICS,
    OUT_VIRCHOW_DIRECTION,
    OUT_VIRCHOW,
    OUT_DISTILL,
    OUT_LOG,
]:

    print(
        path
    )
