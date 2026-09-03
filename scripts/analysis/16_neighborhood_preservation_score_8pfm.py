#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5D analysis — Neighborhood Preservation Score (NPS)
===========================================================

Preserves the ORIGINAL Figure 5D metric while using the audited
71-case native-FOV same-center embeddings and 8 PFMs.

For each slide/model:

1. Use row-aligned patch embeddings at 40x, 10x and 2.5x.
2. Randomly subsample at most 1000 common patch rows (seed=42).
3. L2-normalize patch embeddings.
4. Build cosine kNN neighborhoods independently at each scale.
5. For each patch i and scale pair A,B:

       NPS_i(A,B) =
           |N_k^A(i) ∩ N_k^B(i)| / k

   with k=10.

6. Mean over sampled patches.
7. Overall slide NPS = mean across:
       40x <-> 10x
       10x <-> 2.5x
       40x <-> 2.5x

Important revision improvement:
The three scales MUST contain exactly the same number of row-aligned
patches. No min(n) truncation is permitted.

Statistics:
- 71 paired cases × 8 models
- Friedman omnibus
- paired Wilcoxon signed-rank
- Holm correction across 28 model comparisons
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
)


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
NATIVE_ROOT = SCRIPT.parents[1]

EMBED_ROOT = (
    NATIVE_ROOT
    / "embeddings"
)

RESULT_ROOT = (
    NATIVE_ROOT
    / "reviewer_analysis"
    / "results"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PREFIX = "16_neighborhood_preservation_score_8pfm"

OUT_PAIR = (
    RESULT_ROOT
    / f"{PREFIX}_per_case_pair.csv"
)

OUT_CASE = (
    RESULT_ROOT
    / f"{PREFIX}_per_case.csv"
)

OUT_SUMMARY = (
    RESULT_ROOT
    / f"{PREFIX}_summary.csv"
)

OUT_PAIR_SUMMARY = (
    RESULT_ROOT
    / f"{PREFIX}_summary_by_pair.csv"
)

OUT_CANCER = (
    RESULT_ROOT
    / f"{PREFIX}_by_cancer.csv"
)

OUT_FRIEDMAN = (
    RESULT_ROOT
    / f"{PREFIX}_friedman.csv"
)

OUT_WILCOXON = (
    RESULT_ROOT
    / f"{PREFIX}_wilcoxon_holm.csv"
)

OUT_PROTOCOL = (
    RESULT_ROOT
    / f"{PREFIX}_protocol.json"
)


# ============================================================
# CONFIG
# ============================================================

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

SCALES = [
    "40x",
    "10x",
    "2p5x",
]

SCALE_PAIRS = [
    ("40x", "10x"),
    ("10x", "2p5x"),
    ("40x", "2p5x"),
]

PAIR_LABEL = {
    ("40x", "10x"):
        "40× ↔ 10×",

    ("10x", "2p5x"):
        "10× ↔ 2.5×",

    ("40x", "2p5x"):
        "40× ↔ 2.5×",
}

K_NEIGHBORS = 10
N_SUBSAMPLE = 1000
RANDOM_SEED = 42

EXPECTED_CASES = 71


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("NEIGHBORHOOD PRESERVATION SCORE — 8 PFMs")
print("=" * 100)
print(f"Device       : {DEVICE}")
print(f"k            : {K_NEIGHBORS}")
print(f"Max patches  : {N_SUBSAMPLE}")
print(f"Seed         : {RANDOM_SEED}")
print(f"Embedding root: {EMBED_ROOT}")
print("=" * 100)


# ============================================================
# HELPERS
# ============================================================

def get_prefixes(
    cancer,
    model,
):
    root = (
        EMBED_ROOT
        / cancer
    )

    suffix = (
        f"_{model}_40x.npy"
    )

    return sorted([
        p.name[:-len(suffix)]
        for p in root.iterdir()
        if (
            p.is_file()
            and p.name.endswith(suffix)
        )
    ])


def load_embedding(
    cancer,
    prefix,
    model,
    scale,
):
    path = (
        EMBED_ROOT
        / cancer
        / f"{prefix}_{model}_{scale}.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing:\n{path}"
        )

    X = np.load(
        path,
        mmap_mode="r",
    )

    if X.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embedding: "
            f"{path}, shape={X.shape}"
        )

    return X


def normalize_torch(X):
    norm = torch.linalg.vector_norm(
        X,
        dim=1,
        keepdim=True,
    )

    norm = torch.clamp(
        norm,
        min=1e-12,
    )

    return X / norm


@torch.inference_mode()
def knn_indices_gpu(
    X_np,
    k,
):
    """
    Exact cosine kNN within one sampled scale.

    Self neighbor excluded.
    """

    X = torch.as_tensor(
        np.asarray(
            X_np,
            dtype=np.float32,
        ),
        device=DEVICE,
    )

    X = normalize_torch(X)

    sim = (
        X
        @ X.T
    )

    sim.fill_diagonal_(
        -float("inf")
    )

    idx = torch.topk(
        sim,
        k=k,
        dim=1,
        largest=True,
        sorted=True,
    ).indices

    # sim no longer needed
    del sim, X

    return idx


@torch.inference_mode()
def overlap_score(
    nn_a,
    nn_b,
):
    """
    Mean |N_a(i) intersect N_b(i)| / k.

    nn_a, nn_b:
        [n_queries, k]
    """

    overlap = (
        nn_a.unsqueeze(2)
        ==
        nn_b.unsqueeze(1)
    )

    # for each neighbor in A:
    # is it present anywhere in B?
    overlap_count = (
        overlap
        .any(dim=2)
        .sum(dim=1)
        .float()
    )

    score = (
        overlap_count
        / nn_a.shape[1]
    )

    return float(
        score.mean().item()
    )


def holm_adjust(pvalues):
    p = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)

    ranked = p[
        order
    ]

    adj_ranked = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, raw in enumerate(
        ranked
    ):

        adjusted = (
            (m - i)
            * raw
        )

        running_max = max(
            running_max,
            adjusted,
        )

        adj_ranked[i] = min(
            running_max,
            1.0,
        )

    out = np.empty(
        m,
        dtype=float,
    )

    out[order] = adj_ranked

    return out


def matched_rank_biserial(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    d = x - y

    d = d[
        np.isfinite(d)
        &
        (d != 0)
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
        +
        w_neg
    )

    if denom == 0:
        return 0.0

    return float(
        (w_pos - w_neg)
        / denom
    )


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


# ============================================================
# COMPUTE
# ============================================================

pair_records = []

for model_i, model in enumerate(
    MODELS,
    start=1,
):

    model_display = (
        MODEL_DISPLAY[
            model
        ]
    )

    print(
        f"\n"
        f"[MODEL {model_i}/{len(MODELS)}] "
        f"{model_display}"
    )

    model_count = 0

    for cancer in CANCERS:

        prefixes = (
            get_prefixes(
                cancer,
                model,
            )
        )

        expected = (
            EXPECTED_COUNTS[
                cancer
            ]
        )

        if len(prefixes) != expected:
            raise RuntimeError(
                f"{cancer}/{model}: "
                f"{len(prefixes)}/{expected}"
            )

        print(
            f"  {cancer}: "
            f"{len(prefixes)} cases"
        )

        for case_i, prefix in enumerate(
            prefixes,
            start=1,
        ):

            arrays = {}

            row_counts = {}

            for scale in SCALES:

                X = load_embedding(
                    cancer,
                    prefix,
                    model,
                    scale,
                )

                arrays[
                    scale
                ] = X

                row_counts[
                    scale
                ] = len(X)


            # ----------------------------------------------
            # HARD REQUIRE ROW ALIGNMENT
            # ----------------------------------------------

            unique_rows = set(
                row_counts.values()
            )

            if len(unique_rows) != 1:

                raise RuntimeError(
                    f"ROW MISMATCH: "
                    f"{cancer}/{prefix}/{model}: "
                    f"{row_counts}"
                )

            n_total = next(
                iter(
                    unique_rows
                )
            )

            if (
                n_total
                <=
                K_NEIGHBORS + 5
            ):

                raise RuntimeError(
                    f"Too few patches: "
                    f"{cancer}/{prefix}/{model} "
                    f"n={n_total}"
                )


            # ----------------------------------------------
            # SAME SUBSAMPLE INDICES AT ALL 3 SCALES
            # exact old protocol
            # ----------------------------------------------

            rng = np.random.default_rng(
                RANDOM_SEED
            )

            if n_total > N_SUBSAMPLE:

                sample_idx = rng.choice(
                    np.arange(
                        n_total
                    ),
                    size=N_SUBSAMPLE,
                    replace=False,
                )

            else:

                sample_idx = np.arange(
                    n_total
                )


            # ----------------------------------------------
            # BUILD KNN ONCE PER SCALE
            # ----------------------------------------------

            knn = {}

            for scale in SCALES:

                sampled = np.asarray(
                    arrays[
                        scale
                    ][sample_idx],
                    dtype=np.float32,
                )

                knn[
                    scale
                ] = knn_indices_gpu(
                    sampled,
                    K_NEIGHBORS,
                )


            # ----------------------------------------------
            # THREE SCALE-PAIR NPS VALUES
            # ----------------------------------------------

            for s1, s2 in SCALE_PAIRS:

                score = overlap_score(
                    knn[
                        s1
                    ],
                    knn[
                        s2
                    ],
                )

                pair_records.append({
                    "cancer":
                        cancer,

                    "case_id":
                        prefix,

                    "model":
                        model,

                    "model_display":
                        model_display,

                    "scale_a":
                        s1,

                    "scale_b":
                        s2,

                    "scale_pair":
                        PAIR_LABEL[
                            (s1, s2)
                        ],

                    "n_total_patches":
                        n_total,

                    "n_sampled_patches":
                        len(
                            sample_idx
                        ),

                    "k_neighbors":
                        K_NEIGHBORS,

                    "nps":
                        score,
                })


            # release GPU tensors
            del knn

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            model_count += 1


    if model_count != EXPECTED_CASES:
        raise RuntimeError(
            f"{model}: "
            f"{model_count}/{EXPECTED_CASES}"
        )


# ============================================================
# PAIR-LEVEL TABLE
# ============================================================

pair_df = pd.DataFrame(
    pair_records
)

expected_pair_rows = (
    EXPECTED_CASES
    * len(MODELS)
    * len(SCALE_PAIRS)
)

if len(pair_df) != expected_pair_rows:

    raise RuntimeError(
        f"Expected {expected_pair_rows} "
        f"pair rows, got {len(pair_df)}"
    )


pair_df.to_csv(
    OUT_PAIR,
    index=False,
)


# ============================================================
# CASE-LEVEL OVERALL NPS
# ============================================================

case_df = (
    pair_df.groupby(
        [
            "cancer",
            "case_id",
            "model",
            "model_display",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        neighborhood_preservation_score=(
            "nps",
            "mean",
        ),

        n_scale_pairs=(
            "scale_pair",
            "nunique",
        ),

        n_sampled_patches=(
            "n_sampled_patches",
            "first",
        ),
    )
)

if (
    len(case_df)
    !=
    EXPECTED_CASES
    * len(MODELS)
):

    raise RuntimeError(
        "Bad case-level row count."
    )

if not (
    case_df[
        "n_scale_pairs"
    ] == 3
).all():

    raise RuntimeError(
        "Not all case/model records "
        "contain 3 scale pairs."
    )


case_df.to_csv(
    OUT_CASE,
    index=False,
)


# ============================================================
# SUMMARY BY MODEL
# ============================================================

summary = (
    case_df.groupby(
        [
            "model",
            "model_display",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        n_cases=(
            "case_id",
            "size",
        ),

        mean_nps=(
            "neighborhood_preservation_score",
            "mean",
        ),

        std_nps=(
            "neighborhood_preservation_score",
            "std",
        ),

        median_nps=(
            "neighborhood_preservation_score",
            "median",
        ),

        q1=(
            "neighborhood_preservation_score",
            lambda x:
                x.quantile(
                    0.25
                ),
        ),

        q3=(
            "neighborhood_preservation_score",
            lambda x:
                x.quantile(
                    0.75
                ),
        ),
    )
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


# ============================================================
# SUMMARY BY MODEL × SCALE PAIR
# ============================================================

pair_summary = (
    pair_df.groupby(
        [
            "model",
            "model_display",
            "scale_pair",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        n_cases=(
            "case_id",
            "size",
        ),

        mean_nps=(
            "nps",
            "mean",
        ),

        std_nps=(
            "nps",
            "std",
        ),

        median_nps=(
            "nps",
            "median",
        ),
    )
)

pair_summary.to_csv(
    OUT_PAIR_SUMMARY,
    index=False,
)


# ============================================================
# CANCER SUMMARY
# ============================================================

cancer_summary = (
    case_df.groupby(
        [
            "cancer",
            "model",
            "model_display",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        n_cases=(
            "case_id",
            "size",
        ),

        mean_nps=(
            "neighborhood_preservation_score",
            "mean",
        ),

        std_nps=(
            "neighborhood_preservation_score",
            "std",
        ),
    )
)

cancer_summary.to_csv(
    OUT_CANCER,
    index=False,
)


# ============================================================
# PAIRED MATRIX
# ============================================================

case_df[
    "subject_id"
] = (
    case_df[
        "cancer"
    ].astype(str)
    + "::"
    + case_df[
        "case_id"
    ].astype(str)
)

wide = (
    case_df.pivot(
        index="subject_id",
        columns="model",
        values="neighborhood_preservation_score",
    )
    .reindex(
        columns=MODELS
    )
)

if wide.shape != (
    EXPECTED_CASES,
    len(MODELS),
):

    raise RuntimeError(
        f"Bad paired matrix: "
        f"{wide.shape}"
    )

if wide.isna().any().any():

    raise RuntimeError(
        "Paired matrix contains missing values."
    )


# ============================================================
# FRIEDMAN
# ============================================================

arrays = [
    wide[
        model
    ].to_numpy(
        dtype=float
    )
    for model in MODELS
]

res = friedmanchisquare(
    *arrays
)

chi2_stat = float(
    res.statistic
)

p_friedman = float(
    res.pvalue
)

kendall_w = (
    chi2_stat
    /
    (
        EXPECTED_CASES
        * (
            len(MODELS) - 1
        )
    )
)

friedman_df = pd.DataFrame([
    {
        "test":
            "Friedman",

        "n_cases":
            EXPECTED_CASES,

        "n_models":
            len(MODELS),

        "chi2":
            chi2_stat,

        "df":
            len(MODELS) - 1,

        "p_value":
            p_friedman,

        "kendall_w":
            kendall_w,

        "significance":
            p_to_stars(
                p_friedman
            ),
    }
])

friedman_df.to_csv(
    OUT_FRIEDMAN,
    index=False,
)


# ============================================================
# PAIRED WILCOXON + HOLM
# ============================================================

rows = []

for a, b in itertools.combinations(
    MODELS,
    2,
):

    x = wide[
        a
    ].to_numpy(
        dtype=float
    )

    y = wide[
        b
    ].to_numpy(
        dtype=float
    )

    diff = x - y

    if np.allclose(
        diff,
        0.0,
    ):

        stat = 0.0
        p = 1.0

    else:

        w = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        stat = float(
            w.statistic
        )

        p = float(
            w.pvalue
        )

    rows.append({
        "model_a":
            a,

        "model_a_display":
            MODEL_DISPLAY[a],

        "model_b":
            b,

        "model_b_display":
            MODEL_DISPLAY[b],

        "n_pairs":
            EXPECTED_CASES,

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
                np.mean(
                    diff
                )
            ),

        "median_difference_a_minus_b":
            float(
                np.median(
                    diff
                )
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


wilcox_df = pd.DataFrame(
    rows
)

wilcox_df[
    "p_holm"
] = holm_adjust(
    wilcox_df[
        "p_raw"
    ].to_numpy()
)

wilcox_df[
    "significance_holm"
] = (
    wilcox_df[
        "p_holm"
    ].apply(
        p_to_stars
    )
)

wilcox_df[
    "significant_holm_0p05"
] = (
    wilcox_df[
        "p_holm"
    ]
    < 0.05
)

wilcox_df = (
    wilcox_df
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

wilcox_df.to_csv(
    OUT_WILCOXON,
    index=False,
)


# ============================================================
# PROTOCOL
# ============================================================

protocol = {
    "analysis":
        "Neighborhood Preservation Score",

    "unit":
        "patch-level neighborhoods summarized per case",

    "cohort":
        "71 audited native-FOV cases",

    "models":
        MODELS,

    "scale_pairs": [
        list(x)
        for x in SCALE_PAIRS
    ],

    "row_alignment":
        (
            "Hard-required equal row counts "
            "across 40x, 10x and 2.5x. "
            "No positional truncation."
        ),

    "subsampling":
        {
            "maximum_patches":
                N_SUBSAMPLE,

            "seed":
                RANDOM_SEED,

            "same_indices_across_scales":
                True,
        },

    "neighbors":
        {
            "k":
                K_NEIGHBORS,

            "similarity":
                "cosine",

            "self_excluded":
                True,
        },

    "nps_formula":
        (
            "mean_i "
            "|N_k^A(i) intersection N_k^B(i)| / k"
        ),

    "overall_case_nps":
        (
            "equal mean across 40x-10x, "
            "10x-2.5x and 40x-2.5x"
        ),

    "statistics":
        (
            "Friedman across 8 paired models; "
            "paired Wilcoxon signed-rank tests "
            "with Holm correction across 28 comparisons."
        ),

    "device":
        str(DEVICE),
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
    "NPS ANALYSIS COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nOVERALL NPS"
)

ranked = (
    summary.sort_values(
        "mean_nps",
        ascending=False,
    )
)

print(
    ranked[
        [
            "model_display",
            "n_cases",
            "mean_nps",
            "median_nps",
            "q1",
            "q3",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nNPS BY SCALE PAIR"
)

pair_matrix = (
    pair_summary.pivot(
        index="model_display",
        columns="scale_pair",
        values="mean_nps",
    )
)

print(
    pair_matrix
    .round(4)
    .to_string()
)


print(
    "\nFRIEDMAN"
)

print(
    friedman_df
    .to_string(
        index=False,
        formatters={
            "chi2":
                lambda x:
                f"{x:.4f}",

            "p_value":
                lambda x:
                (
                    f"{x:.3e}"
                    if x < 1e-4
                    else f"{x:.6f}"
                ),

            "kendall_w":
                lambda x:
                f"{x:.4f}",
        },
    )
)


print(
    "\nPAIRWISE WILCOXON + HOLM"
)

print(
    wilcox_df[
        [
            "model_a_display",
            "model_b_display",
            "mean_difference_a_minus_b",
            "rank_biserial_a_minus_b",
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

            "p_holm":
                lambda x:
                (
                    f"{x:.3e}"
                    if x < 1e-4
                    else f"{x:.6f}"
                ),
        },
    )
)


print(
    "\nOutputs:"
)

for p in [
    OUT_PAIR,
    OUT_CASE,
    OUT_SUMMARY,
    OUT_PAIR_SUMMARY,
    OUT_CANCER,
    OUT_FRIEDMAN,
    OUT_WILCOXON,
    OUT_PROTOCOL,
]:
    print(p)
