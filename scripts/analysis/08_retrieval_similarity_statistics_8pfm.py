#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical analysis for 8-PFM Cross-scale Retrieval Similarity
===============================================================

Input
-----
07_cross_scale_retrieval_similarity_8pfm_values.csv

Data structure
--------------
71 cases × 8 PFMs × 3 scale pairs = 1704 observations.

Primary mixed-effects model
---------------------------
retrieval_similarity ~ model * scale_pair + (1 | case)

where:
- model      = 8-level within-case factor
- scale_pair = 3-level within-case factor
- case       = random intercept

Likelihood-ratio tests (ML fits)
--------------------------------
Model:
    scale_pair
    vs
    model + scale_pair

Scale pair:
    model
    vs
    model + scale_pair

Model × Scale pair:
    model + scale_pair
    vs
    model * scale_pair

Nonparametric sensitivity
-------------------------
A. Model effect:
   - first average 3 scale-pair values within each case/model
   - Friedman test across 8 models
   - paired Wilcoxon signed-rank
   - Holm correction across 28 model pairs

B. Scale-pair effect:
   - first average 8 model values within each case/scale-pair
   - Friedman test across 3 scale pairs
   - paired Wilcoxon signed-rank
   - Holm correction across 3 pairwise comparisons
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    chi2,
    friedmanchisquare,
    wilcoxon,
)

import statsmodels.formula.api as smf


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

INPUT = (
    RESULT_ROOT
    / "07_cross_scale_retrieval_similarity_8pfm_values.csv"
)

PREFIX = "08_retrieval_similarity_statistics_8pfm"

OUT_MIXED_LRT = (
    RESULT_ROOT
    / f"{PREFIX}_mixedlm_lrt.csv"
)

OUT_MIXED_FITS = (
    RESULT_ROOT
    / f"{PREFIX}_mixedlm_fit_summary.csv"
)

OUT_MIXED_COEFS = (
    RESULT_ROOT
    / f"{PREFIX}_mixedlm_full_coefficients.csv"
)

OUT_MODEL_FRIEDMAN = (
    RESULT_ROOT
    / f"{PREFIX}_model_friedman.csv"
)

OUT_MODEL_POSTHOC = (
    RESULT_ROOT
    / f"{PREFIX}_model_wilcoxon_holm.csv"
)

OUT_PAIR_FRIEDMAN = (
    RESULT_ROOT
    / f"{PREFIX}_scalepair_friedman.csv"
)

OUT_PAIR_POSTHOC = (
    RESULT_ROOT
    / f"{PREFIX}_scalepair_wilcoxon_holm.csv"
)

OUT_PROTOCOL = (
    RESULT_ROOT
    / f"{PREFIX}_protocol.json"
)


# ============================================================
# CONSTANTS
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

PAIR_CODE_ORDER = [
    "40_10",
    "10_2p5",
    "40_2p5",
]

PAIR_DISPLAY = {
    "40_10": "40× ↔ 10×",
    "10_2p5": "10× ↔ 2.5×",
    "40_2p5": "40× ↔ 2.5×",
}

PAIR_FROM_SCALES = {
    ("40x", "10x"): "40_10",
    ("10x", "2p5x"): "10_2p5",
    ("40x", "2p5x"): "40_2p5",
}

EXPECTED_CASES = 71
EXPECTED_MODELS = 8
EXPECTED_PAIRS = 3
EXPECTED_ROWS = (
    EXPECTED_CASES
    * EXPECTED_MODELS
    * EXPECTED_PAIRS
)


# ============================================================
# HELPERS
# ============================================================

def holm_adjust(pvalues):
    p = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    m = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = np.empty(
        m,
        dtype=np.float64,
    )

    running_max = 0.0

    for i, raw_p in enumerate(ranked):
        adjusted = (
            (m - i)
            * raw_p
        )

        running_max = max(
            running_max,
            adjusted,
        )

        adjusted_ranked[i] = min(
            running_max,
            1.0,
        )

    out = np.empty(
        m,
        dtype=np.float64,
    )

    out[order] = adjusted_ranked

    return out


def p_to_stars(p):
    if not np.isfinite(p):
        return "NA"

    if p < 1e-4:
        return "****"

    if p < 1e-3:
        return "***"

    if p < 1e-2:
        return "**"

    if p < 0.05:
        return "*"

    return "ns"


def format_p(p):
    if not np.isfinite(p):
        return "NA"

    if p < 1e-4:
        return f"{p:.3e}"

    return f"{p:.6f}"


def matched_rank_biserial(x, y):
    """
    Matched-pairs rank-biserial correlation.

    Positive:
        x > y

    Negative:
        x < y
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    d = x - y

    d = d[
        np.isfinite(d)
        & (d != 0)
    ]

    if len(d) == 0:
        return 0.0

    ranks = (
        pd.Series(
            np.abs(d)
        )
        .rank(
            method="average"
        )
        .to_numpy()
    )

    w_pos = ranks[
        d > 0
    ].sum()

    w_neg = ranks[
        d < 0
    ].sum()

    denom = (
        w_pos
        + w_neg
    )

    if denom == 0:
        return 0.0

    return float(
        (w_pos - w_neg)
        / denom
    )


# ============================================================
# LOAD
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing input:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

required = {
    "cancer",
    "case_id",
    "model",
    "scale_a",
    "scale_b",
    "retrieval_similarity",
}

missing = (
    required
    - set(df.columns)
)

if missing:
    raise RuntimeError(
        f"Missing columns: "
        f"{sorted(missing)}"
    )

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"got {len(df)}"
    )

if not np.isfinite(
    df[
        "retrieval_similarity"
    ].to_numpy(
        dtype=np.float64
    )
).all():
    raise RuntimeError(
        "retrieval_similarity "
        "contains NaN/Inf."
    )


# ============================================================
# SUBJECT ID / SCALE-PAIR CODE
# ============================================================

df["subject_id"] = (
    df["cancer"].astype(str)
    + "::"
    + df["case_id"].astype(str)
)

df["pair_code"] = [
    PAIR_FROM_SCALES.get(
        (a, b),
        None,
    )
    for a, b in zip(
        df["scale_a"],
        df["scale_b"],
    )
]

if df[
    "pair_code"
].isna().any():
    bad = df.loc[
        df["pair_code"].isna(),
        [
            "scale_a",
            "scale_b",
        ],
    ].drop_duplicates()

    raise RuntimeError(
        "Unknown scale pairs:\n"
        + bad.to_string(
            index=False
        )
    )


# ============================================================
# AUDIT REPEATED-MEASURES STRUCTURE
# ============================================================

if (
    df["subject_id"].nunique()
    != EXPECTED_CASES
):
    raise RuntimeError(
        "Expected 71 unique cases."
    )

subject_audit = (
    df.groupby(
        "subject_id",
        observed=True,
    )
    .agg(
        n_models=(
            "model",
            "nunique",
        ),
        n_pairs=(
            "pair_code",
            "nunique",
        ),
        n_rows=(
            "retrieval_similarity",
            "size",
        ),
    )
)

bad = subject_audit[
    (
        subject_audit[
            "n_models"
        ]
        != EXPECTED_MODELS
    )
    |
    (
        subject_audit[
            "n_pairs"
        ]
        != EXPECTED_PAIRS
    )
    |
    (
        subject_audit[
            "n_rows"
        ]
        != (
            EXPECTED_MODELS
            * EXPECTED_PAIRS
        )
    )
]

if len(bad) > 0:
    raise RuntimeError(
        "Repeated-measures audit failed:\n"
        + bad.to_string()
    )


# ============================================================
# EXPLICIT CATEGORICAL ORDER
# ============================================================

df["model"] = pd.Categorical(
    df["model"],
    categories=MODELS,
    ordered=False,
)

df["pair_code"] = pd.Categorical(
    df["pair_code"],
    categories=PAIR_CODE_ORDER,
    ordered=False,
)


print("=" * 100)
print(
    "8-PFM CROSS-SCALE RETRIEVAL "
    "SIMILARITY — STATISTICS"
)
print("=" * 100)

print(
    f"Input        : {INPUT}"
)

print(
    f"Observations : {len(df)}"
)

print(
    f"Cases        : "
    f"{df['subject_id'].nunique()}"
)

print(
    f"Models       : "
    f"{df['model'].nunique()}"
)

print(
    f"Scale pairs  : "
    f"{df['pair_code'].nunique()}"
)

print(
    "Random unit  : case"
)

print("=" * 100)


# ============================================================
# MIXED MODEL
# ============================================================

FORMULAS = {
    "intercept_only":
        "retrieval_similarity ~ 1",

    "pair_only":
        (
            "retrieval_similarity ~ "
            "C(pair_code)"
        ),

    "model_only":
        (
            "retrieval_similarity ~ "
            "C(model)"
        ),

    "additive":
        (
            "retrieval_similarity ~ "
            "C(model) + C(pair_code)"
        ),

    "full_interaction":
        (
            "retrieval_similarity ~ "
            "C(model) * C(pair_code)"
        ),
}


def fit_mixedlm(
    name,
    formula,
):
    print(
        f"\n[FIT] {name}"
    )

    print(
        f"      {formula}"
    )

    model = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["subject_id"],
        re_formula="1",
    )

    methods = [
        "lbfgs",
        "bfgs",
        "cg",
    ]

    last_error = None

    for method in methods:
        try:
            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    reml=False,
                    method=method,
                    maxiter=3000,
                    disp=False,
                )

            converged = bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            )

            print(
                f"      optimizer="
                f"{method} | "
                f"converged="
                f"{converged} | "
                f"llf="
                f"{result.llf:.4f}"
            )

            if caught:
                unique_warnings = sorted(
                    set(
                        str(w.message)
                        for w in caught
                    )
                )

                for w in unique_warnings:
                    print(
                        f"      [warning] {w}"
                    )

            if converged:
                return (
                    result,
                    method,
                )

            last_error = RuntimeError(
                f"{name}: "
                f"{method} did not converge"
            )

        except Exception as e:
            print(
                f"      optimizer="
                f"{method} failed: "
                f"{repr(e)}"
            )

            last_error = e

    raise RuntimeError(
        f"All optimizers failed "
        f"for {name}"
    ) from last_error


fits = {}
optimizers = {}

for name, formula in FORMULAS.items():

    result, optimizer = (
        fit_mixedlm(
            name,
            formula,
        )
    )

    fits[name] = result
    optimizers[name] = optimizer


# ============================================================
# MIXED MODEL FIT SUMMARY
# ============================================================

fit_rows = []

for name in FORMULAS:

    r = fits[name]

    random_var = float(
        np.asarray(
            r.cov_re
        )[0, 0]
    )

    fit_rows.append({
        "model_name":
            name,

        "formula":
            FORMULAS[name],

        "optimizer":
            optimizers[name],

        "converged":
            bool(
                r.converged
            ),

        "n_obs":
            int(
                r.nobs
            ),

        "n_fixed_effects":
            int(
                len(
                    r.fe_params
                )
            ),

        "log_likelihood":
            float(
                r.llf
            ),

        "aic":
            float(
                r.aic
            ),

        "bic":
            float(
                r.bic
            ),

        "random_intercept_variance":
            random_var,

        "residual_variance":
            float(
                r.scale
            ),
    })


fit_df = pd.DataFrame(
    fit_rows
)

fit_df.to_csv(
    OUT_MIXED_FITS,
    index=False,
)


# ============================================================
# LIKELIHOOD-RATIO TEST
# ============================================================

def likelihood_ratio_test(
    label,
    reduced_name,
    full_name,
):

    reduced = fits[
        reduced_name
    ]

    full = fits[
        full_name
    ]

    lr = (
        2.0
        * (
            full.llf
            - reduced.llf
        )
    )

    df_diff = (
        len(
            full.fe_params
        )
        -
        len(
            reduced.fe_params
        )
    )

    if df_diff <= 0:
        raise RuntimeError(
            f"{label}: invalid "
            f"df difference="
            f"{df_diff}"
        )

    p = chi2.sf(
        max(
            lr,
            0.0,
        ),
        df_diff,
    )

    return {
        "effect":
            label,

        "reduced_model":
            reduced_name,

        "full_model":
            full_name,

        "lr_statistic":
            float(lr),

        "df":
            int(df_diff),

        "p_value":
            float(p),

        "significance":
            p_to_stars(p),
    }


lrt_rows = [

    likelihood_ratio_test(
        label="Model",
        reduced_name="pair_only",
        full_name="additive",
    ),

    likelihood_ratio_test(
        label="Scale pair",
        reduced_name="model_only",
        full_name="additive",
    ),

    likelihood_ratio_test(
        label="Model × Scale pair",
        reduced_name="additive",
        full_name="full_interaction",
    ),
]


lrt_df = pd.DataFrame(
    lrt_rows
)

lrt_df.to_csv(
    OUT_MIXED_LRT,
    index=False,
)


# ============================================================
# FULL MIXED MODEL COEFFICIENTS
# ============================================================

full = fits[
    "full_interaction"
]

conf = full.conf_int()

coef_rows = []

for term in full.fe_params.index:

    estimate = float(
        full.fe_params[
            term
        ]
    )

    se = float(
        full.bse_fe[
            term
        ]
    )

    z = (
        estimate / se
        if se > 0
        else np.nan
    )

    try:
        p = float(
            full.pvalues[
                term
            ]
        )
    except Exception:
        p = np.nan

    try:
        ci_low = float(
            conf.loc[
                term,
                0,
            ]
        )

        ci_high = float(
            conf.loc[
                term,
                1,
            ]
        )

    except Exception:
        ci_low = np.nan
        ci_high = np.nan

    coef_rows.append({
        "term":
            term,

        "estimate":
            estimate,

        "std_error":
            se,

        "z_value":
            z,

        "p_value":
            p,

        "ci95_low":
            ci_low,

        "ci95_high":
            ci_high,

        "significance":
            p_to_stars(p),
    })


coef_df = pd.DataFrame(
    coef_rows
)

coef_df.to_csv(
    OUT_MIXED_COEFS,
    index=False,
)


# ============================================================
# NONPARAMETRIC A:
# MODEL EFFECT
# ============================================================
#
# Average 3 scale pairs within each case/model.
# Then each case contributes one value per model.
# ============================================================

case_model = (
    df.groupby(
        [
            "subject_id",
            "model",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        retrieval_similarity=(
            "retrieval_similarity",
            "mean",
        )
    )
)


wide_model = (
    case_model
    .pivot(
        index="subject_id",
        columns="model",
        values="retrieval_similarity",
    )
    .reindex(
        columns=MODELS
    )
)


if wide_model.isna().any().any():
    raise RuntimeError(
        "Model Friedman matrix "
        "contains missing values."
    )


model_arrays = [
    wide_model[
        model
    ].to_numpy(
        dtype=np.float64
    )
    for model in MODELS
]


model_friedman = (
    friedmanchisquare(
        *model_arrays
    )
)

model_chi2 = float(
    model_friedman.statistic
)

model_p = float(
    model_friedman.pvalue
)

model_k = len(MODELS)
model_n = len(wide_model)

model_w = (
    model_chi2
    /
    (
        model_n
        * (
            model_k
            - 1
        )
    )
)


model_friedman_df = pd.DataFrame([
    {
        "test":
            "Friedman",

        "effect":
            "Model",

        "n_cases":
            model_n,

        "n_levels":
            model_k,

        "chi2":
            model_chi2,

        "df":
            model_k - 1,

        "p_value":
            model_p,

        "kendall_w":
            model_w,

        "significance":
            p_to_stars(
                model_p
            ),
    }
])


model_friedman_df.to_csv(
    OUT_MODEL_FRIEDMAN,
    index=False,
)


# ============================================================
# MODEL PAIRWISE WILCOXON
# ============================================================

model_posthoc_rows = []

for model_a, model_b in itertools.combinations(
    MODELS,
    2,
):

    x = wide_model[
        model_a
    ].to_numpy(
        dtype=np.float64
    )

    y = wide_model[
        model_b
    ].to_numpy(
        dtype=np.float64
    )

    diff = x - y

    if np.allclose(
        diff,
        0.0,
    ):
        stat = 0.0
        p = 1.0

    else:
        res = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        stat = float(
            res.statistic
        )

        p = float(
            res.pvalue
        )

    model_posthoc_rows.append({
        "model_a":
            model_a,

        "model_a_display":
            MODEL_DISPLAY[
                model_a
            ],

        "model_b":
            model_b,

        "model_b_display":
            MODEL_DISPLAY[
                model_b
            ],

        "n_pairs":
            len(x),

        "mean_a":
            float(
                np.mean(x)
            ),

        "mean_b":
            float(
                np.mean(y)
            ),

        "mean_difference_a_minus_b":
            float(
                np.mean(diff)
            ),

        "median_difference_a_minus_b":
            float(
                np.median(diff)
            ),

        "rank_biserial_a_minus_b":
            matched_rank_biserial(
                x,
                y,
            ),

        "wilcoxon_statistic":
            stat,

        "p_raw":
            p,
    })


model_posthoc_df = pd.DataFrame(
    model_posthoc_rows
)

model_posthoc_df[
    "p_holm"
] = holm_adjust(
    model_posthoc_df[
        "p_raw"
    ].to_numpy()
)

model_posthoc_df[
    "significance_holm"
] = (
    model_posthoc_df[
        "p_holm"
    ]
    .apply(
        p_to_stars
    )
)

model_posthoc_df[
    "significant_holm_0p05"
] = (
    model_posthoc_df[
        "p_holm"
    ] < 0.05
)

model_posthoc_df = (
    model_posthoc_df
    .sort_values(
        [
            "p_holm",
            "p_raw",
        ]
    )
    .reset_index(
        drop=True
    )
)

model_posthoc_df.to_csv(
    OUT_MODEL_POSTHOC,
    index=False,
)


# ============================================================
# NONPARAMETRIC B:
# SCALE-PAIR EFFECT
# ============================================================
#
# Average 8 model scores within each case/scale pair.
# Then each case contributes one value per scale pair.
# ============================================================

case_pair = (
    df.groupby(
        [
            "subject_id",
            "pair_code",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        retrieval_similarity=(
            "retrieval_similarity",
            "mean",
        )
    )
)


wide_pair = (
    case_pair
    .pivot(
        index="subject_id",
        columns="pair_code",
        values="retrieval_similarity",
    )
    .reindex(
        columns=PAIR_CODE_ORDER
    )
)


if wide_pair.isna().any().any():
    raise RuntimeError(
        "Scale-pair Friedman matrix "
        "contains missing values."
    )


pair_arrays = [
    wide_pair[
        pair
    ].to_numpy(
        dtype=np.float64
    )
    for pair in PAIR_CODE_ORDER
]


pair_friedman = (
    friedmanchisquare(
        *pair_arrays
    )
)

pair_chi2 = float(
    pair_friedman.statistic
)

pair_p = float(
    pair_friedman.pvalue
)

pair_k = len(
    PAIR_CODE_ORDER
)

pair_n = len(
    wide_pair
)

pair_w = (
    pair_chi2
    /
    (
        pair_n
        * (
            pair_k
            - 1
        )
    )
)


pair_friedman_df = pd.DataFrame([
    {
        "test":
            "Friedman",

        "effect":
            "Scale pair",

        "n_cases":
            pair_n,

        "n_levels":
            pair_k,

        "chi2":
            pair_chi2,

        "df":
            pair_k - 1,

        "p_value":
            pair_p,

        "kendall_w":
            pair_w,

        "significance":
            p_to_stars(
                pair_p
            ),
    }
])


pair_friedman_df.to_csv(
    OUT_PAIR_FRIEDMAN,
    index=False,
)


# ============================================================
# SCALE-PAIR PAIRWISE WILCOXON
# ============================================================

pair_posthoc_rows = []

for pair_a, pair_b in itertools.combinations(
    PAIR_CODE_ORDER,
    2,
):

    x = wide_pair[
        pair_a
    ].to_numpy(
        dtype=np.float64
    )

    y = wide_pair[
        pair_b
    ].to_numpy(
        dtype=np.float64
    )

    diff = x - y

    if np.allclose(
        diff,
        0.0,
    ):
        stat = 0.0
        p = 1.0

    else:
        res = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        stat = float(
            res.statistic
        )

        p = float(
            res.pvalue
        )

    pair_posthoc_rows.append({
        "pair_a":
            pair_a,

        "pair_a_display":
            PAIR_DISPLAY[
                pair_a
            ],

        "pair_b":
            pair_b,

        "pair_b_display":
            PAIR_DISPLAY[
                pair_b
            ],

        "n_pairs":
            len(x),

        "mean_a":
            float(
                np.mean(x)
            ),

        "mean_b":
            float(
                np.mean(y)
            ),

        "mean_difference_a_minus_b":
            float(
                np.mean(diff)
            ),

        "median_difference_a_minus_b":
            float(
                np.median(diff)
            ),

        "rank_biserial_a_minus_b":
            matched_rank_biserial(
                x,
                y,
            ),

        "wilcoxon_statistic":
            stat,

        "p_raw":
            p,
    })


pair_posthoc_df = pd.DataFrame(
    pair_posthoc_rows
)

pair_posthoc_df[
    "p_holm"
] = holm_adjust(
    pair_posthoc_df[
        "p_raw"
    ].to_numpy()
)

pair_posthoc_df[
    "significance_holm"
] = (
    pair_posthoc_df[
        "p_holm"
    ]
    .apply(
        p_to_stars
    )
)

pair_posthoc_df[
    "significant_holm_0p05"
] = (
    pair_posthoc_df[
        "p_holm"
    ] < 0.05
)

pair_posthoc_df = (
    pair_posthoc_df
    .sort_values(
        [
            "p_holm",
            "p_raw",
        ]
    )
    .reset_index(
        drop=True
    )
)

pair_posthoc_df.to_csv(
    OUT_PAIR_POSTHOC,
    index=False,
)


# ============================================================
# PROTOCOL JSON
# ============================================================

protocol = {
    "analysis":
        (
            "cross_scale_retrieval_"
            "similarity_statistics"
        ),

    "input":
        str(INPUT),

    "n_cases":
        EXPECTED_CASES,

    "n_models":
        EXPECTED_MODELS,

    "n_scale_pairs":
        EXPECTED_PAIRS,

    "n_observations":
        EXPECTED_ROWS,

    "primary_mixed_model":
        (
            "retrieval_similarity ~ "
            "C(model) * C(pair_code) "
            "+ random intercept(case)"
        ),

    "mixed_model_estimation":
        (
            "Maximum likelihood "
            "(REML=False)"
        ),

    "mixed_model_lrt": {
        "model_effect":
            (
                "pair-only vs additive"
            ),

        "scale_pair_effect":
            (
                "model-only vs additive"
            ),

        "interaction":
            (
                "additive vs "
                "model×scale-pair"
            ),
    },

    "model_nonparametric":
        (
            "Average 3 scale-pair "
            "scores within each "
            "case/model; Friedman "
            "across 8 models; "
            "paired Wilcoxon; "
            "Holm correction "
            "across 28 comparisons."
        ),

    "scale_pair_nonparametric":
        (
            "Average 8 model scores "
            "within each case/scale pair; "
            "Friedman across 3 scale pairs; "
            "paired Wilcoxon; "
            "Holm correction across "
            "3 comparisons."
        ),
}


with open(
    OUT_PROTOCOL,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        protocol,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# REPORT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "MIXED-EFFECTS OMNIBUS TESTS"
)

print(
    "=" * 100
)

print(
    lrt_df[
        [
            "effect",
            "lr_statistic",
            "df",
            "p_value",
            "significance",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "lr_statistic":
                lambda x:
                f"{x:.4f}",

            "p_value":
                format_p,
        },
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "MODEL EFFECT — FRIEDMAN"
)

print(
    "=" * 100
)

print(
    model_friedman_df
    .to_string(
        index=False,
        formatters={
            "chi2":
                lambda x:
                f"{x:.4f}",

            "p_value":
                format_p,

            "kendall_w":
                lambda x:
                f"{x:.4f}",
        },
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "MODEL PAIRWISE — "
    "WILCOXON + HOLM"
)

print(
    "=" * 100
)

print(
    model_posthoc_df[
        [
            "model_a_display",
            "model_b_display",
            "n_pairs",
            "mean_difference_a_minus_b",
            "rank_biserial_a_minus_b",
            "p_raw",
            "p_holm",
            "significance_holm",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "mean_difference_a_minus_b":
                lambda x:
                f"{x:.6f}",

            "rank_biserial_a_minus_b":
                lambda x:
                f"{x:.4f}",

            "p_raw":
                format_p,

            "p_holm":
                format_p,
        },
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "SCALE-PAIR EFFECT — FRIEDMAN"
)

print(
    "=" * 100
)

print(
    pair_friedman_df
    .to_string(
        index=False,
        formatters={
            "chi2":
                lambda x:
                f"{x:.4f}",

            "p_value":
                format_p,

            "kendall_w":
                lambda x:
                f"{x:.4f}",
        },
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "SCALE-PAIR PAIRWISE — "
    "WILCOXON + HOLM"
)

print(
    "=" * 100
)

print(
    pair_posthoc_df[
        [
            "pair_a_display",
            "pair_b_display",
            "n_pairs",
            "mean_difference_a_minus_b",
            "rank_biserial_a_minus_b",
            "p_raw",
            "p_holm",
            "significance_holm",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "mean_difference_a_minus_b":
                lambda x:
                f"{x:.6f}",

            "rank_biserial_a_minus_b":
                lambda x:
                f"{x:.4f}",

            "p_raw":
                format_p,

            "p_holm":
                format_p,
        },
    )
)


print(
    "\nOutputs:"
)

for p in [
    OUT_MIXED_LRT,
    OUT_MIXED_FITS,
    OUT_MIXED_COEFS,
    OUT_MODEL_FRIEDMAN,
    OUT_MODEL_POSTHOC,
    OUT_PAIR_FRIEDMAN,
    OUT_PAIR_POSTHOC,
    OUT_PROTOCOL,
]:
    print(p)
