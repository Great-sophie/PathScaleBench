#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model × Cancer Interaction Analysis of Relative Cross-scale Retention
=====================================================================

Outcome
-------
Relative cross-scale representation retention:

    R = CKA(40x, 2.5x) /
        mean[CKA(40x, 10x), CKA(10x, 2.5x)]

Data structure
--------------
71 audited native-FOV cases
8 pathology foundation models per case
568 observations total

Mixed-effects model
-------------------
    relative_retention ~ model * cancer + (1 | case)

- Model: within-case repeated factor
- Cancer: between-case factor
- Case: random intercept

Omnibus tests
-------------
Likelihood-ratio tests using ML fits:

1. Model contribution:
       model + cancer
       vs
       cancer

2. Cancer contribution:
       model + cancer
       vs
       model

3. Model × Cancer interaction:
       model * cancer
       vs
       model + cancer

Sensitivity analysis
--------------------
Within each cancer:
- Friedman repeated-measures test across 8 PFMs
- Kendall's W effect size
- Holm correction across the 8 cancer-specific Friedman tests

Outputs
-------
- mixed-model likelihood-ratio tests
- full-model fixed-effect coefficients
- cancer × model means
- additive-expected means
- interaction deviation matrix
- cancer-specific Friedman tests
- protocol JSON
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import chi2, friedmanchisquare

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

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT = (
    RESULT_ROOT
    / "02_relative_information_retention_8pfm_values.csv"
)

PREFIX = "06_model_cancer_interaction_retention"

OUT_LRT = (
    RESULT_ROOT
    / f"{PREFIX}_mixedlm_lrt.csv"
)

OUT_COEFS = (
    RESULT_ROOT
    / f"{PREFIX}_mixedlm_full_coefficients.csv"
)

OUT_FITS = (
    RESULT_ROOT
    / f"{PREFIX}_mixedlm_fit_summary.csv"
)

OUT_CELL = (
    RESULT_ROOT
    / f"{PREFIX}_cancer_model_summary.csv"
)

OUT_INTERACTION_LONG = (
    RESULT_ROOT
    / f"{PREFIX}_interaction_deviation_long.csv"
)

OUT_INTERACTION_MATRIX = (
    RESULT_ROOT
    / f"{PREFIX}_interaction_deviation_matrix.csv"
)

OUT_FRIEDMAN = (
    RESULT_ROOT
    / f"{PREFIX}_cancerwise_friedman_holm.csv"
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

MODEL_ORDER_DISPLAY = [
    "Phikon",
    "UNI2",
    "UNI",
    "Midnight",
    "Virchow",
    "GigaPath",
    "GigaPath-Flash",
    "Virchow2",
]

CANCER_ORDER = [
    "BLCA",
    "BRCA",
    "COAD",
    "HNSC",
    "KIRC",
    "LUAD",
    "STAD",
    "UCEC",
]

EXPECTED_COUNTS = {
    "BLCA": 10,
    "BRCA": 10,
    "COAD": 7,
    "HNSC": 8,
    "KIRC": 9,
    "LUAD": 9,
    "STAD": 9,
    "UCEC": 9,
}

EXPECTED_CASES = 71
EXPECTED_MODELS = 8
EXPECTED_ROWS = EXPECTED_CASES * EXPECTED_MODELS


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
        adj = (m - i) * raw_p

        running_max = max(
            running_max,
            adj,
        )

        adjusted_ranked[i] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    adjusted[order] = adjusted_ranked

    return adjusted


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


# ============================================================
# LOAD AND AUDIT
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
    "relative_retention",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


df = df[
    df["cancer"].isin(CANCER_ORDER)
    & df["model"].isin(MODELS)
].copy()


if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"got {len(df)}"
    )


if not np.isfinite(
    df["relative_retention"]
    .to_numpy(dtype=np.float64)
).all():
    raise RuntimeError(
        "relative_retention contains NaN/Inf."
    )


# Make globally unique case identifier.
df["subject_id"] = (
    df["cancer"].astype(str)
    + "::"
    + df["case_id"].astype(str)
)


# Exactly 8 model observations per case.
per_subject = (
    df.groupby(
        "subject_id",
        observed=True,
    )
    .agg(
        n_models=("model", "nunique"),
        n_rows=("model", "size"),
        n_cancers=("cancer", "nunique"),
    )
)

if len(per_subject) != EXPECTED_CASES:
    raise RuntimeError(
        f"Expected {EXPECTED_CASES} cases, "
        f"got {len(per_subject)}"
    )

bad = per_subject[
    (per_subject["n_models"] != EXPECTED_MODELS)
    | (per_subject["n_rows"] != EXPECTED_MODELS)
    | (per_subject["n_cancers"] != 1)
]

if len(bad) > 0:
    raise RuntimeError(
        "Repeated-measures structure failed:\n"
        + bad.to_string()
    )


# Cancer counts.
case_counts = (
    df[
        [
            "cancer",
            "subject_id",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "cancer",
        observed=True,
    )
    .size()
    .to_dict()
)

for cancer in CANCER_ORDER:
    expected = EXPECTED_COUNTS[cancer]
    observed = case_counts.get(
        cancer,
        0,
    )

    if observed != expected:
        raise RuntimeError(
            f"{cancer}: "
            f"{observed}/{expected} cases"
        )


# Explicit categorical references.
#
# References are not scientifically privileged;
# they only determine coefficient parameterization.
df["model"] = pd.Categorical(
    df["model"],
    categories=MODELS,
    ordered=False,
)

df["cancer"] = pd.Categorical(
    df["cancer"],
    categories=CANCER_ORDER,
    ordered=False,
)


print("=" * 96)
print("MODEL × CANCER INTERACTION — RELATIVE RETENTION")
print("=" * 96)

print(
    f"Input       : {INPUT}"
)

print(
    f"Observations: {len(df)}"
)

print(
    f"Cases       : {df['subject_id'].nunique()}"
)

print(
    f"Models      : {df['model'].nunique()}"
)

print(
    f"Cancers     : {df['cancer'].nunique()}"
)

print(
    "Random unit : case / subject_id"
)

print("=" * 96)


# ============================================================
# MIXED MODEL FITTING
# ============================================================

def fit_mixedlm(
    name,
    formula,
):
    """
    ML fit with identical random-intercept structure
    across all candidate models.

    Try several optimizers for robustness.
    """

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
                f"      optimizer={method} | "
                f"converged={converged} | "
                f"llf={result.llf:.4f}"
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
                return result, method

            last_error = RuntimeError(
                f"{name}: "
                f"{method} did not converge"
            )

        except Exception as e:
            print(
                f"      optimizer={method} failed: "
                f"{repr(e)}"
            )

            last_error = e

    raise RuntimeError(
        f"All optimizers failed for {name}"
    ) from last_error


# Hierarchical model set.
FORMULAS = {
    "intercept_only":
        "relative_retention ~ 1",

    "cancer_only":
        "relative_retention ~ C(cancer)",

    "model_only":
        "relative_retention ~ C(model)",

    "additive":
        (
            "relative_retention ~ "
            "C(model) + C(cancer)"
        ),

    "full_interaction":
        (
            "relative_retention ~ "
            "C(model) * C(cancer)"
        ),
}


fits = {}
optimizers = {}

for name, formula in FORMULAS.items():
    result, optimizer = fit_mixedlm(
        name,
        formula,
    )

    fits[name] = result
    optimizers[name] = optimizer


# ============================================================
# FIT SUMMARY
# ============================================================

fit_rows = []

for name in FORMULAS:
    r = fits[name]

    fit_rows.append({
        "model_name": name,
        "formula": FORMULAS[name],
        "optimizer": optimizers[name],
        "converged":
            bool(r.converged),
        "n_obs":
            int(r.nobs),
        "n_fixed_effects":
            int(len(r.fe_params)),
        "log_likelihood":
            float(r.llf),
        "aic":
            float(r.aic),
        "bic":
            float(r.bic),
        "random_intercept_variance":
            float(
                np.asarray(
                    r.cov_re
                )[0, 0]
            ),
        "residual_scale":
            float(r.scale),
    })


fit_df = pd.DataFrame(
    fit_rows
)

fit_df.to_csv(
    OUT_FITS,
    index=False,
)


# ============================================================
# LIKELIHOOD-RATIO TESTS
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

    lr = 2.0 * (
        full.llf
        - reduced.llf
    )

    df_diff = (
        len(full.fe_params)
        - len(reduced.fe_params)
    )

    if df_diff <= 0:
        raise RuntimeError(
            f"{label}: invalid df difference "
            f"{df_diff}"
        )

    p = chi2.sf(
        max(lr, 0.0),
        df_diff,
    )

    return {
        "effect": label,
        "reduced_model":
            reduced_name,
        "full_model":
            full_name,
        "ll_reduced":
            float(reduced.llf),
        "ll_full":
            float(full.llf),
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
        reduced_name="cancer_only",
        full_name="additive",
    ),

    likelihood_ratio_test(
        label="Cancer",
        reduced_name="model_only",
        full_name="additive",
    ),

    likelihood_ratio_test(
        label="Model × Cancer",
        reduced_name="additive",
        full_name="full_interaction",
    ),
]


lrt_df = pd.DataFrame(
    lrt_rows
)

lrt_df.to_csv(
    OUT_LRT,
    index=False,
)


# ============================================================
# FULL MODEL COEFFICIENTS
# ============================================================

full = fits[
    "full_interaction"
]

conf = full.conf_int()

coef_rows = []

for term in full.fe_params.index:
    estimate = float(
        full.fe_params[term]
    )

    se = float(
        full.bse_fe[term]
    )

    z = float(
        estimate / se
    ) if se > 0 else np.nan

    # MixedLM result p-values include fixed + variance
    # parameters; index by coefficient name.
    try:
        p = float(
            full.pvalues[term]
        )
    except Exception:
        p = np.nan

    try:
        ci_low = float(
            conf.loc[term, 0]
        )

        ci_high = float(
            conf.loc[term, 1]
        )

    except Exception:
        ci_low = np.nan
        ci_high = np.nan

    coef_rows.append({
        "term": term,
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
    OUT_COEFS,
    index=False,
)


# ============================================================
# CANCER × MODEL CELL SUMMARY
# ============================================================

cell = (
    df.groupby(
        [
            "cancer",
            "model",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        n_cases=(
            "subject_id",
            "nunique",
        ),
        mean_retention=(
            "relative_retention",
            "mean",
        ),
        std_retention=(
            "relative_retention",
            "std",
        ),
        median_retention=(
            "relative_retention",
            "median",
        ),
    )
)


cell["model_display"] = (
    cell["model"]
    .astype(str)
    .map(MODEL_DISPLAY)
)


cell.to_csv(
    OUT_CELL,
    index=False,
)


# ============================================================
# INTERACTION DEVIATION
# ============================================================
#
# Descriptive interaction residual:
#
# observed cell mean
# -
# additive expected mean
#
# additive expected:
#
# model marginal mean
# + cancer marginal mean
# - grand mean
#
# Positive:
# model retains more than expected in that cancer.
#
# Negative:
# model retains less than expected in that cancer.
# ============================================================

grand_mean = float(
    df[
        "relative_retention"
    ].mean()
)


model_means = (
    df.groupby(
        "model",
        observed=True,
    )[
        "relative_retention"
    ]
    .mean()
    .to_dict()
)


cancer_means = (
    df.groupby(
        "cancer",
        observed=True,
    )[
        "relative_retention"
    ]
    .mean()
    .to_dict()
)


interaction_rows = []

for row in cell.itertuples(
    index=False
):
    model = str(row.model)
    cancer = str(row.cancer)

    expected_additive = (
        model_means[model]
        + cancer_means[cancer]
        - grand_mean
    )

    deviation = (
        float(row.mean_retention)
        - expected_additive
    )

    interaction_rows.append({
        "cancer":
            cancer,

        "model":
            model,

        "model_display":
            MODEL_DISPLAY[model],

        "n_cases":
            int(row.n_cases),

        "observed_mean_retention":
            float(
                row.mean_retention
            ),

        "model_marginal_mean":
            float(
                model_means[model]
            ),

        "cancer_marginal_mean":
            float(
                cancer_means[cancer]
            ),

        "grand_mean":
            grand_mean,

        "additive_expected_retention":
            float(
                expected_additive
            ),

        "interaction_deviation":
            float(
                deviation
            ),

        "abs_interaction_deviation":
            float(
                abs(deviation)
            ),
    })


interaction_df = pd.DataFrame(
    interaction_rows
)

interaction_df.to_csv(
    OUT_INTERACTION_LONG,
    index=False,
)


interaction_matrix = (
    interaction_df
    .pivot(
        index="cancer",
        columns="model_display",
        values="interaction_deviation",
    )
    .reindex(
        index=CANCER_ORDER,
        columns=MODEL_ORDER_DISPLAY,
    )
)


if interaction_matrix.isna().any().any():
    raise RuntimeError(
        "Interaction matrix contains NaN."
    )


interaction_matrix.to_csv(
    OUT_INTERACTION_MATRIX
)


# ============================================================
# CANCER-SPECIFIC FRIEDMAN TESTS
# ============================================================

friedman_rows = []

for cancer in CANCER_ORDER:

    sub = df[
        df["cancer"] == cancer
    ].copy()

    wide = (
        sub.pivot(
            index="subject_id",
            columns="model",
            values="relative_retention",
        )
        .reindex(
            columns=MODELS
        )
    )

    if wide.isna().any().any():
        raise RuntimeError(
            f"{cancer}: "
            "incomplete repeated-measures matrix"
        )

    n = len(wide)
    k = len(MODELS)

    arrays = [
        wide[model]
        .to_numpy(dtype=np.float64)
        for model in MODELS
    ]

    result = friedmanchisquare(
        *arrays
    )

    chi2_stat = float(
        result.statistic
    )

    p = float(
        result.pvalue
    )

    # Kendall's W for Friedman:
    #
    # W = chi-square / [n * (k - 1)]
    #
    kendall_w = (
        chi2_stat
        / (n * (k - 1))
    )

    friedman_rows.append({
        "cancer":
            cancer,

        "n_cases":
            n,

        "n_models":
            k,

        "friedman_chi2":
            chi2_stat,

        "df":
            k - 1,

        "p_raw":
            p,

        "kendall_w":
            float(kendall_w),
    })


friedman_df = pd.DataFrame(
    friedman_rows
)


friedman_df["p_holm"] = holm_adjust(
    friedman_df[
        "p_raw"
    ].to_numpy()
)


friedman_df[
    "significance_holm"
] = (
    friedman_df[
        "p_holm"
    ]
    .apply(
        p_to_stars
    )
)


friedman_df[
    "significant_holm_0p05"
] = (
    friedman_df[
        "p_holm"
    ] < 0.05
)


friedman_df.to_csv(
    OUT_FRIEDMAN,
    index=False,
)


# ============================================================
# PROTOCOL JSON
# ============================================================

protocol = {
    "analysis":
        "model_cancer_interaction_relative_retention",

    "source":
        str(INPUT),

    "outcome":
        "relative_retention",

    "n_observations":
        int(len(df)),

    "n_cases":
        int(
            df[
                "subject_id"
            ].nunique()
        ),

    "n_models":
        EXPECTED_MODELS,

    "models":
        MODELS,

    "cancers":
        CANCER_ORDER,

    "cancer_case_counts":
        EXPECTED_COUNTS,

    "mixed_model": (
        "relative_retention ~ "
        "C(model) * C(cancer) "
        "+ random intercept(case)"
    ),

    "estimation":
        (
            "Maximum likelihood (REML=False) "
            "for nested fixed-effect "
            "likelihood-ratio comparisons."
        ),

    "lrt_model_effect":
        (
            "additive model "
            "vs cancer-only model"
        ),

    "lrt_cancer_effect":
        (
            "additive model "
            "vs model-only model"
        ),

    "lrt_interaction":
        (
            "full model×cancer model "
            "vs additive model"
        ),

    "nonparametric_sensitivity":
        (
            "Cancer-specific Friedman tests "
            "across 8 repeated model measurements; "
            "Holm correction across 8 cancers."
        ),

    "interaction_deviation": (
        "observed cancer×model mean - "
        "[model marginal mean + "
        "cancer marginal mean - grand mean]"
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
    + "=" * 96
)

print(
    "MIXED-EFFECTS OMNIBUS TESTS"
)

print(
    "=" * 96
)


display_lrt = (
    lrt_df[
        [
            "effect",
            "lr_statistic",
            "df",
            "p_value",
            "significance",
        ]
    ]
    .copy()
)


print(
    display_lrt
    .to_string(
        index=False,
        formatters={
            "lr_statistic":
                lambda x: f"{x:.4f}",

            "p_value":
                format_p,
        },
    )
)


print(
    "\n"
    + "=" * 96
)

print(
    "CANCER-SPECIFIC FRIEDMAN TESTS"
)

print(
    "=" * 96
)


print(
    friedman_df[
        [
            "cancer",
            "n_cases",
            "friedman_chi2",
            "kendall_w",
            "p_raw",
            "p_holm",
            "significance_holm",
        ]
    ]
    .sort_values(
        "p_holm"
    )
    .to_string(
        index=False,
        formatters={
            "friedman_chi2":
                lambda x: f"{x:.4f}",

            "kendall_w":
                lambda x: f"{x:.4f}",

            "p_raw":
                format_p,

            "p_holm":
                format_p,
        },
    )
)


print(
    "\n"
    + "=" * 96
)

print(
    "LARGEST MODEL × CANCER DEVIATIONS"
)

print(
    "=" * 96
)


print(
    interaction_df[
        [
            "cancer",
            "model_display",
            "observed_mean_retention",
            "additive_expected_retention",
            "interaction_deviation",
        ]
    ]
    .sort_values(
        "interaction_deviation",
        key=lambda x: np.abs(x),
        ascending=False,
    )
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


print("\nOutputs:")

for p in [
    OUT_LRT,
    OUT_COEFS,
    OUT_FITS,
    OUT_CELL,
    OUT_INTERACTION_LONG,
    OUT_INTERACTION_MATRIX,
    OUT_FRIEDMAN,
    OUT_PROTOCOL,
]:
    print(p)
