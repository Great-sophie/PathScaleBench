#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8-PFM Relative Cross-scale Representation Retention
====================================================

Definition inherited from the original submitted Figure 3A:

    short_range_alignment
        = mean(
            CKA(40x, 10x),
            CKA(10x, 2.5x)
          )

    long_range_alignment
        = CKA(40x, 2.5x)

    relative_retention
        = long_range_alignment / short_range_alignment

Revision protocol:
------------------
- 71 audited native-FOV same-center cases
- 8 pathology foundation models
- Full-patch per-case CKA
- No re-computation of embeddings or CKA
- Paired Wilcoxon signed-rank tests across models
- Holm correction across all 28 model-pair comparisons
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
NATIVE_ROOT = SCRIPT_PATH.parents[1]

RESULT_ROOT = (
    NATIVE_ROOT
    / "reviewer_analysis"
    / "results"
)

INPUT_CKA = (
    RESULT_ROOT
    / "01b_cross_scale_cka_8pfm_fullpatch_per_case.csv"
)

PREFIX = "02_relative_information_retention_8pfm"

OUT_VALUES = RESULT_ROOT / f"{PREFIX}_values.csv"
OUT_SUMMARY = RESULT_ROOT / f"{PREFIX}_summary.csv"
OUT_BY_CANCER = RESULT_ROOT / f"{PREFIX}_by_cancer.csv"
OUT_STATS = RESULT_ROOT / f"{PREFIX}_paired_wilcoxon_holm.csv"
OUT_PROTOCOL = RESULT_ROOT / f"{PREFIX}_protocol.json"


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

EXPECTED_CASES = 71
SEED = 2026

PAIR_40_10 = ("40x", "10x")
PAIR_40_25 = ("40x", "2p5x")
PAIR_10_25 = ("10x", "2p5x")


# ============================================================
# UTILITIES
# ============================================================

def bootstrap_ci(values, n_boot=5000, seed=SEED):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    means = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            len(x),
            size=len(x),
        )
        means[i] = x[idx].mean()

    lo, hi = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(lo), float(hi)


def holm_adjust(pvalues):
    """
    Holm step-down family-wise error correction.

    Returns adjusted p-values in the original order.
    """

    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = np.empty(m, dtype=np.float64)

    running_max = 0.0

    for i, raw_p in enumerate(ranked):
        adj = (m - i) * raw_p
        running_max = max(running_max, adj)
        adjusted_ranked[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=np.float64)
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


def matched_rank_biserial(x, y):
    """
    Matched-pairs rank-biserial correlation.

    Positive value means x > y overall.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    d = x - y

    d = d[
        np.isfinite(d)
        & (d != 0)
    ]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)

    ranks = pd.Series(abs_d).rank(
        method="average"
    ).to_numpy()

    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()

    denom = w_pos + w_neg

    if denom == 0:
        return 0.0

    return float(
        (w_pos - w_neg) / denom
    )


# ============================================================
# LOAD AND VALIDATE CKA
# ============================================================

def load_cka():
    if not INPUT_CKA.exists():
        raise FileNotFoundError(
            f"Missing input:\n{INPUT_CKA}"
        )

    df = pd.read_csv(INPUT_CKA)

    required = {
        "cancer",
        "case_id",
        "model",
        "scale_a",
        "scale_b",
        "cka",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[
        df["model"].isin(MODELS)
    ].copy()

    if len(df) != EXPECTED_CASES * len(MODELS) * 3:
        raise RuntimeError(
            "Unexpected CKA row count: "
            f"{len(df)} != "
            f"{EXPECTED_CASES * len(MODELS) * 3}"
        )

    if not np.isfinite(
        df["cka"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Input CKA contains NaN/Inf."
        )

    return df


# ============================================================
# COMPUTE RETENTION
# ============================================================

def compute_retention(cka_df):
    rows = []

    grouped = cka_df.groupby(
        [
            "cancer",
            "case_id",
            "model",
        ],
        observed=True,
        sort=False,
    )

    for (
        cancer,
        case_id,
        model,
    ), sub in grouped:

        lookup = {}

        for r in sub.itertuples(
            index=False
        ):
            lookup[
                (r.scale_a, r.scale_b)
            ] = float(r.cka)

        required = [
            PAIR_40_10,
            PAIR_40_25,
            PAIR_10_25,
        ]

        missing = [
            pair
            for pair in required
            if pair not in lookup
        ]

        if missing:
            raise RuntimeError(
                f"Missing scale pairs for "
                f"{cancer}/{case_id}/{model}: "
                f"{missing}"
            )

        c40_10 = lookup[PAIR_40_10]
        c40_25 = lookup[PAIR_40_25]
        c10_25 = lookup[PAIR_10_25]

        short_range = np.mean(
            [
                c40_10,
                c10_25,
            ]
        )

        long_range = c40_25

        if (
            not np.isfinite(short_range)
            or short_range <= 0
        ):
            raise RuntimeError(
                "Invalid short-range alignment: "
                f"{cancer}/{case_id}/{model} "
                f"short={short_range}"
            )

        retention = (
            long_range / short_range
        )

        if not np.isfinite(retention):
            raise RuntimeError(
                f"Non-finite retention: "
                f"{cancer}/{case_id}/{model}"
            )

        rows.append({
            "cancer": cancer,
            "case_id": case_id,
            "model": model,
            "model_display":
                MODEL_DISPLAY[model],

            "cka_40x_10x":
                c40_10,

            "cka_10x_2p5x":
                c10_25,

            "cka_40x_2p5x":
                c40_25,

            "short_range_alignment":
                short_range,

            "long_range_alignment":
                long_range,

            "relative_retention":
                retention,
        })

    out = pd.DataFrame(rows)

    expected = (
        EXPECTED_CASES
        * len(MODELS)
    )

    if len(out) != expected:
        raise RuntimeError(
            f"Expected {expected} "
            f"case-model retention records, "
            f"got {len(out)}"
        )

    counts = (
        out.groupby(
            "model",
            observed=True,
        )
        .size()
        .to_dict()
    )

    for model in MODELS:
        n = counts.get(model, 0)

        if n != EXPECTED_CASES:
            raise RuntimeError(
                f"{model}: "
                f"{n}/{EXPECTED_CASES} cases"
            )

    return out.sort_values(
        [
            "model",
            "cancer",
            "case_id",
        ]
    ).reset_index(drop=True)


# ============================================================
# SUMMARY
# ============================================================

def summarize(values, n_boot):
    rows = []

    for model in MODELS:
        sub = values[
            values["model"] == model
        ]

        x = sub[
            "relative_retention"
        ].to_numpy(dtype=np.float64)

        lo, hi = bootstrap_ci(
            x,
            n_boot=n_boot,
            seed=SEED,
        )

        q1, median, q3 = np.quantile(
            x,
            [0.25, 0.50, 0.75],
        )

        rows.append({
            "model": model,
            "model_display":
                MODEL_DISPLAY[model],

            "n_cases":
                len(x),

            "mean_retention":
                float(np.mean(x)),

            "std_retention":
                float(
                    np.std(
                        x,
                        ddof=1,
                    )
                ),

            "median_retention":
                float(median),

            "q1":
                float(q1),

            "q3":
                float(q3),

            "iqr":
                float(q3 - q1),

            "ci95_low":
                lo,

            "ci95_high":
                hi,

            "mean_short_range":
                float(
                    sub[
                        "short_range_alignment"
                    ].mean()
                ),

            "mean_long_range":
                float(
                    sub[
                        "long_range_alignment"
                    ].mean()
                ),
        })

    return pd.DataFrame(rows)


def summarize_by_cancer(values):
    rows = []

    grouped = values.groupby(
        [
            "cancer",
            "model",
        ],
        observed=True,
    )

    for (
        cancer,
        model,
    ), sub in grouped:

        x = sub[
            "relative_retention"
        ].to_numpy(dtype=np.float64)

        rows.append({
            "cancer": cancer,
            "model": model,
            "model_display":
                MODEL_DISPLAY[model],

            "n_cases":
                len(x),

            "mean_retention":
                float(np.mean(x)),

            "std_retention":
                float(
                    np.std(
                        x,
                        ddof=1,
                    )
                ) if len(x) > 1
                else np.nan,

            "median_retention":
                float(np.median(x)),

            "mean_short_range":
                float(
                    sub[
                        "short_range_alignment"
                    ].mean()
                ),

            "mean_long_range":
                float(
                    sub[
                        "long_range_alignment"
                    ].mean()
                ),
        })

    return pd.DataFrame(rows)


# ============================================================
# PAIRED MODEL COMPARISONS
# ============================================================

def paired_statistics(values):
    rows = []

    for model_a, model_b in itertools.combinations(
        MODELS,
        2,
    ):
        a = values[
            values["model"] == model_a
        ][
            [
                "cancer",
                "case_id",
                "relative_retention",
            ]
        ].rename(
            columns={
                "relative_retention":
                    "retention_a"
            }
        )

        b = values[
            values["model"] == model_b
        ][
            [
                "cancer",
                "case_id",
                "relative_retention",
            ]
        ].rename(
            columns={
                "relative_retention":
                    "retention_b"
            }
        )

        paired = a.merge(
            b,
            on=[
                "cancer",
                "case_id",
            ],
            how="inner",
            validate="one_to_one",
        )

        x = paired[
            "retention_a"
        ].to_numpy(dtype=np.float64)

        y = paired[
            "retention_b"
        ].to_numpy(dtype=np.float64)

        if len(x) != EXPECTED_CASES:
            raise RuntimeError(
                f"{model_a} vs {model_b}: "
                f"only {len(x)} paired cases"
            )

        diff = x - y

        if np.allclose(
            diff,
            0.0,
        ):
            statistic = 0.0
            p_raw = 1.0

        else:
            res = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(
                res.statistic
            )
            p_raw = float(
                res.pvalue
            )

        rows.append({
            "model_a":
                model_a,

            "model_a_display":
                MODEL_DISPLAY[model_a],

            "model_b":
                model_b,

            "model_b_display":
                MODEL_DISPLAY[model_b],

            "n_pairs":
                len(x),

            "mean_a":
                float(np.mean(x)),

            "mean_b":
                float(np.mean(y)),

            "median_a":
                float(np.median(x)),

            "median_b":
                float(np.median(y)),

            "mean_paired_difference_a_minus_b":
                float(np.mean(diff)),

            "median_paired_difference_a_minus_b":
                float(np.median(diff)),

            "rank_biserial_a_minus_b":
                matched_rank_biserial(
                    x,
                    y,
                ),

            "wilcoxon_statistic":
                statistic,

            "p_raw":
                p_raw,
        })

    stats = pd.DataFrame(rows)

    stats["p_holm"] = holm_adjust(
        stats["p_raw"].to_numpy()
    )

    stats["significance_holm"] = (
        stats["p_holm"]
        .apply(p_to_stars)
    )

    stats["significant_holm_0p05"] = (
        stats["p_holm"] < 0.05
    )

    return stats.sort_values(
        [
            "p_holm",
            "p_raw",
        ]
    ).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help=(
            "Bootstrap replicates for "
            "95% CI of model mean retention."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "8-PFM RELATIVE CROSS-SCALE "
        "REPRESENTATION RETENTION"
    )
    print("=" * 88)

    print(
        f"Input CKA : {INPUT_CKA}"
    )

    print(
        "Definition: "
        "CKA(40x,2.5x) / "
        "mean[CKA(40x,10x), "
        "CKA(10x,2.5x)]"
    )

    print(
        f"Expected cases/model: "
        f"{EXPECTED_CASES}"
    )

    print(
        f"Models: "
        f"{', '.join(MODELS)}"
    )

    print("=" * 88)

    cka = load_cka()

    values = compute_retention(
        cka
    )

    values.to_csv(
        OUT_VALUES,
        index=False,
    )

    summary = summarize(
        values,
        n_boot=args.bootstrap,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    by_cancer = summarize_by_cancer(
        values
    )

    by_cancer.to_csv(
        OUT_BY_CANCER,
        index=False,
    )

    stats = paired_statistics(
        values
    )

    stats.to_csv(
        OUT_STATS,
        index=False,
    )

    protocol = {
        "analysis":
            "relative_cross_scale_representation_retention",

        "source":
            str(INPUT_CKA),

        "cohort":
            "71 audited native-FOV same-center cases",

        "models":
            MODELS,

        "n_models":
            len(MODELS),

        "n_cases_per_model":
            EXPECTED_CASES,

        "cka_protocol":
            "audited full-patch per-case linear CKA",

        "retention_formula": (
            "CKA(40x,2.5x) / "
            "mean(CKA(40x,10x), "
            "CKA(10x,2.5x))"
        ),

        "interpretation": (
            "Relative preservation of "
            "long-range scale alignment "
            "compared with adjacent-scale "
            "alignment."
        ),

        "statistics":
            (
                "paired Wilcoxon signed-rank "
                "tests across matched cases; "
                "Holm correction across all "
                "28 model-pair comparisons"
            ),

        "bootstrap_replicates":
            args.bootstrap,

        "seed":
            SEED,
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

    print("\n" + "=" * 88)
    print("RETENTION SUMMARY")
    print("=" * 88)

    cols = [
        "model_display",
        "n_cases",
        "mean_retention",
        "median_retention",
        "ci95_low",
        "ci95_high",
        "mean_short_range",
        "mean_long_range",
    ]

    print(
        summary[
            cols
        ]
        .sort_values(
            "mean_retention",
            ascending=False,
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 88)
    print(
        "PAIRED WILCOXON + HOLM"
    )
    print("=" * 88)

    stat_cols = [
        "model_a_display",
        "model_b_display",
        "n_pairs",
        "mean_paired_difference_a_minus_b",
        "rank_biserial_a_minus_b",
        "p_raw",
        "p_holm",
        "significance_holm",
    ]

    print(
        stats[
            stat_cols
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(OUT_VALUES)
    print(OUT_SUMMARY)
    print(OUT_BY_CANCER)
    print(OUT_STATS)
    print(OUT_PROTOCOL)


if __name__ == "__main__":
    main()
