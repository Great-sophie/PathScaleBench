#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Model × Scale-pair interaction test for
8-PFM cross-scale retrieval similarity.

Why GEE?
--------
The previous random-intercept MixedLM collapsed to the boundary
(random-intercept variance = 0) for the additive and interaction
models, producing infinite log-likelihoods and invalid LRTs.

Here we use a marginal repeated-measures model:

    retrieval_similarity ~ C(model) * C(pair_code)

with:
    cluster = case
    family = Gaussian(identity)
    working correlation = Exchangeable
    covariance = robust sandwich

Primary target:
    omnibus Wald test of all Model × Scale-pair interaction terms.

The previously valid Friedman / paired-Wilcoxon tests remain the
primary nonparametric evidence for model and scale-pair main effects.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

import statsmodels.api as sm
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

PREFIX = "08b_retrieval_similarity_gee_interaction"

OUT_SUMMARY = (
    RESULT_ROOT
    / f"{PREFIX}_summary.txt"
)

OUT_COEFS = (
    RESULT_ROOT
    / f"{PREFIX}_coefficients.csv"
)

OUT_WALD = (
    RESULT_ROOT
    / f"{PREFIX}_wald_tests.csv"
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

PAIR_CODE_ORDER = [
    "40_10",
    "10_2p5",
    "40_2p5",
]

PAIR_FROM_SCALES = {
    ("40x", "10x"): "40_10",
    ("10x", "2p5x"): "10_2p5",
    ("40x", "2p5x"): "40_2p5",
}

EXPECTED_CASES = 71
EXPECTED_ROWS = 71 * 8 * 3


# ============================================================
# HELPERS
# ============================================================

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


def joint_wald(result, terms, label):
    """
    Joint Wald chi-square test H0: selected coefficients = 0.
    Uses the robust GEE covariance matrix.
    """

    names = list(result.params.index)

    idx = [
        i
        for i, name in enumerate(names)
        if terms(name)
    ]

    if len(idx) == 0:
        raise RuntimeError(
            f"No coefficients found for test: {label}"
        )

    beta = (
        result.params
        .to_numpy(dtype=float)[idx]
    )

    cov = (
        result.cov_params()
        .to_numpy(dtype=float)
    )

    cov_sub = cov[
        np.ix_(idx, idx)
    ]

    inv_cov = np.linalg.pinv(
        cov_sub
    )

    stat = float(
        beta.T
        @ inv_cov
        @ beta
    )

    df_test = len(idx)

    p = float(
        sm.stats.stattools.stats.chi2.sf(
            stat,
            df_test,
        )
    )

    return {
        "effect": label,
        "wald_chi2": stat,
        "df": df_test,
        "p_value": p,
        "significance": p_to_stars(p),
        "n_coefficients_tested": len(idx),
        "terms": " | ".join(
            names[i]
            for i in idx
        ),
    }


# scipy chi-square is safer / explicit
from scipy.stats import chi2


def joint_wald(result, selector, label):
    names = list(
        result.params.index
    )

    idx = [
        i
        for i, name in enumerate(names)
        if selector(name)
    ]

    if not idx:
        raise RuntimeError(
            f"No coefficients selected for {label}"
        )

    beta = (
        result.params
        .to_numpy(dtype=float)[idx]
    )

    cov = (
        result.cov_params()
        .to_numpy(dtype=float)
    )

    cov_sub = cov[
        np.ix_(idx, idx)
    ]

    stat = float(
        beta.T
        @ np.linalg.pinv(cov_sub)
        @ beta
    )

    df_test = len(idx)

    p = float(
        chi2.sf(
            stat,
            df_test,
        )
    )

    return {
        "effect": label,
        "wald_chi2": stat,
        "df": df_test,
        "p_value": p,
        "significance": p_to_stars(p),
        "n_coefficients_tested": df_test,
        "terms": " | ".join(
            names[i]
            for i in idx
        ),
    }


# ============================================================
# LOAD / AUDIT
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

missing = required - set(
    df.columns
)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"got {len(df)}"
    )

df["subject_id"] = (
    df["cancer"].astype(str)
    + "::"
    + df["case_id"].astype(str)
)

df["pair_code"] = [
    PAIR_FROM_SCALES.get(
        (a, b)
    )
    for a, b in zip(
        df["scale_a"],
        df["scale_b"],
    )
]

if df["pair_code"].isna().any():
    raise RuntimeError(
        "Unknown scale pair detected."
    )

if (
    df["subject_id"].nunique()
    != EXPECTED_CASES
):
    raise RuntimeError(
        "Expected 71 cases."
    )

audit = (
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

bad = audit[
    (audit["n_models"] != 8)
    | (audit["n_pairs"] != 3)
    | (audit["n_rows"] != 24)
]

if len(bad):
    raise RuntimeError(
        "Repeated-measures audit failed:\n"
        + bad.to_string()
    )


# Explicit references
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


# ============================================================
# FULL GEE
# ============================================================

formula_full = (
    "retrieval_similarity ~ "
    "C(model) * C(pair_code)"
)

print("=" * 96)
print("GEE MODEL × SCALE-PAIR INTERACTION")
print("=" * 96)
print("Formula :", formula_full)
print("Rows    :", len(df))
print("Cases   :", df["subject_id"].nunique())
print(
    "Working correlation: Exchangeable"
)
print(
    "Covariance          : robust sandwich"
)
print("=" * 96)


model_full = smf.gee(
    formula=formula_full,
    groups="subject_id",
    data=df,
    family=sm.families.Gaussian(),
    cov_struct=sm.cov_struct.Exchangeable(),
)

result_full = model_full.fit(
    cov_type="robust",
    maxiter=200,
)


# ============================================================
# ADDITIVE GEE
# ============================================================

# Main-effect Wald tests are cleaner in the additive model,
# because treatment-coded "main effects" in an interaction
# model refer only to the reference level of the other factor.

formula_add = (
    "retrieval_similarity ~ "
    "C(model) + C(pair_code)"
)

model_add = smf.gee(
    formula=formula_add,
    groups="subject_id",
    data=df,
    family=sm.families.Gaussian(),
    cov_struct=sm.cov_struct.Exchangeable(),
)

result_add = model_add.fit(
    cov_type="robust",
    maxiter=200,
)


# ============================================================
# ROBUST WALD TESTS
# ============================================================

wald_rows = []

# Model main effect from additive model
wald_rows.append(
    joint_wald(
        result_add,
        selector=lambda name: (
            name.startswith("C(model)[T.")
        ),
        label="Model",
    )
)

# Scale-pair main effect from additive model
wald_rows.append(
    joint_wald(
        result_add,
        selector=lambda name: (
            name.startswith(
                "C(pair_code)[T."
            )
        ),
        label="Scale pair",
    )
)

# Interaction from full model:
# all terms containing both factors
wald_rows.append(
    joint_wald(
        result_full,
        selector=lambda name: (
            "C(model)[T." in name
            and "C(pair_code)[T." in name
            and ":" in name
        ),
        label="Model × Scale pair",
    )
)

wald_df = pd.DataFrame(
    wald_rows
)

wald_df.to_csv(
    OUT_WALD,
    index=False,
)


# ============================================================
# COEFFICIENT TABLE — FULL MODEL
# ============================================================

ci = result_full.conf_int()

coef_df = pd.DataFrame({
    "term":
        result_full.params.index,

    "estimate":
        result_full.params.values,

    "robust_se":
        result_full.bse.values,

    "z_value":
        result_full.tvalues.values,

    "p_value":
        result_full.pvalues.values,

    "ci95_low":
        ci.iloc[:, 0].values,

    "ci95_high":
        ci.iloc[:, 1].values,
})

coef_df["significance"] = (
    coef_df["p_value"]
    .apply(p_to_stars)
)

coef_df.to_csv(
    OUT_COEFS,
    index=False,
)


# ============================================================
# SUMMARY TEXT
# ============================================================

with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        result_full.summary().as_text()
    )


# ============================================================
# PROTOCOL
# ============================================================

protocol = {
    "analysis":
        "retrieval_similarity_gee_interaction",

    "reason_for_gee":
        (
            "MixedLM random-intercept variance "
            "collapsed to zero for multiple nested "
            "models, yielding infinite log-likelihood "
            "and invalid likelihood-ratio tests."
        ),

    "formula_full":
        formula_full,

    "formula_additive":
        formula_add,

    "cluster":
        "case / subject_id",

    "n_clusters":
        EXPECTED_CASES,

    "observations_per_cluster":
        24,

    "family":
        "Gaussian",

    "link":
        "identity",

    "working_correlation":
        "Exchangeable",

    "covariance":
        "robust sandwich",

    "interaction_test":
        (
            "Joint robust Wald chi-square test "
            "of all 14 Model × Scale-pair "
            "interaction coefficients."
        ),

    "main_effect_note":
        (
            "Model and scale-pair main effects "
            "are tested from an additive GEE; "
            "primary nonparametric confirmation "
            "comes from Friedman tests."
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
    "ROBUST GEE WALD TESTS"
)
print(
    "=" * 96
)

print(
    wald_df[
        [
            "effect",
            "wald_chi2",
            "df",
            "p_value",
            "significance",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "wald_chi2":
                lambda x: f"{x:.4f}",

            "p_value":
                lambda x: (
                    f"{x:.3e}"
                    if x < 1e-4
                    else f"{x:.6f}"
                ),
        },
    )
)

print(
    "\nWorking correlation parameter:"
)

try:
    print(
        result_full.cov_struct.dep_params
    )
except Exception:
    print("NA")

print("\nOutputs:")
print(OUT_WALD)
print(OUT_COEFS)
print(OUT_SUMMARY)
print(OUT_PROTOCOL)
