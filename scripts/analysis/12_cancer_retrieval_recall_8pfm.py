#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5A analysis
Biological identity preservation across 8 pathology foundation models
=====================================================================

Preserves the ORIGINAL Figure 5A retrieval definition:

1. Mean-pool patch embeddings separately at 40x, 10x, 2.5x.
2. L2-normalize each scale-level vector.
3. Equal-average the three normalized scale vectors.
4. L2-normalize the final multiscale slide embedding.
5. Within each PFM, perform leave-one-out cosine retrieval across slides.
6. Recall@K = 1 if at least one of the top-K retrieved slides has
   the same cancer label as the query slide; otherwise 0.

Revision:
- 71 audited native-FOV cases
- 8 PFMs
- exact preservation of original retrieval metric
- paired statistics:
    Cochran's Q omnibus
    pairwise exact McNemar
    Holm correction
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    binomtest,
    chi2,
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

PREFIX = "12_cancer_retrieval_recall_8pfm"

OUT_VALUES = (
    RESULT_ROOT
    / f"{PREFIX}_per_case.csv"
)

OUT_SUMMARY = (
    RESULT_ROOT
    / f"{PREFIX}_summary.csv"
)

OUT_CANCER = (
    RESULT_ROOT
    / f"{PREFIX}_by_cancer.csv"
)

OUT_METADATA = (
    RESULT_ROOT
    / f"{PREFIX}_slide_embedding_metadata.csv"
)

OUT_Q = (
    RESULT_ROOT
    / f"{PREFIX}_cochran_q.csv"
)

OUT_MCNEMAR = (
    RESULT_ROOT
    / f"{PREFIX}_mcnemar_holm.csv"
)

OUT_PROTOCOL = (
    RESULT_ROOT
    / f"{PREFIX}_protocol.json"
)


# ============================================================
# CONSTANTS
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

K_LIST = [
    1,
    5,
    10,
]

EXPECTED_CASES = 71


# ============================================================
# HELPERS
# ============================================================

def normalize_vec(x):
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    norm = np.linalg.norm(x)

    if norm == 0:
        return x

    return x / norm


def get_slide_prefixes(
    cancer,
    model,
):
    embed_dir = (
        EMBED_ROOT
        / cancer
    )

    if not embed_dir.exists():
        raise FileNotFoundError(
            f"Missing embedding directory:\n"
            f"{embed_dir}"
        )

    suffix = (
        f"_{model}_40x.npy"
    )

    prefixes = sorted([
        p.name[:-len(suffix)]
        for p in embed_dir.iterdir()
        if (
            p.is_file()
            and p.name.endswith(suffix)
        )
    ])

    return prefixes


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
            f"Missing embedding:\n{path}"
        )

    X = np.load(
        path,
        mmap_mode="r",
    )

    if X.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embedding: "
            f"{path} shape={X.shape}"
        )

    if len(X) < 10:
        raise RuntimeError(
            f"Too few patches: "
            f"{path} n={len(X)}"
        )

    return X


def build_slide_embedding(
    cancer,
    prefix,
    model,
):
    """
    EXACT original Figure 5A aggregation:

    scale patch embeddings
        -> mean pooling
        -> L2 normalization

    3 scale vectors
        -> equal mean
        -> L2 normalization
    """

    scale_vecs = []

    patch_counts = {}

    for scale in SCALES:

        X = load_embedding(
            cancer,
            prefix,
            model,
            scale,
        )

        patch_counts[
            scale
        ] = int(
            len(X)
        )

        # use float64 accumulation for numerical stability,
        # then convert normalized vector to float32
        v = np.asarray(
            X,
            dtype=np.float32,
        ).mean(
            axis=0,
            dtype=np.float64,
        )

        v = normalize_vec(v)

        scale_vecs.append(v)

    if len(scale_vecs) != 3:
        raise RuntimeError(
            f"{cancer}/{prefix}/{model}: "
            f"expected 3 scale vectors"
        )

    dims = {
        len(v)
        for v in scale_vecs
    }

    if len(dims) != 1:
        raise RuntimeError(
            f"{cancer}/{prefix}/{model}: "
            f"scale dimensions differ"
        )

    slide_vec = np.mean(
        np.stack(
            scale_vecs,
            axis=0,
        ),
        axis=0,
    )

    slide_vec = normalize_vec(
        slide_vec
    )

    return (
        slide_vec,
        patch_counts,
    )


def cosine_matrix(X):
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    norm = np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    norm[
        norm == 0
    ] = 1.0

    X = X / norm

    return (
        X
        @ X.T
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

        adj_ranked[
            i
        ] = min(
            running_max,
            1.0,
        )

    out = np.empty(
        m,
        dtype=float,
    )

    out[
        order
    ] = adj_ranked

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


def cochran_q_binary(X):
    """
    Cochran's Q test.

    X shape:
        n_subjects × k_methods

    Binary observations only.
    """

    X = np.asarray(
        X,
        dtype=int,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be 2D"
        )

    n, k = X.shape

    if not np.isin(
        X,
        [0, 1],
    ).all():
        raise ValueError(
            "Cochran Q requires binary data"
        )

    col_sum = X.sum(
        axis=0
    )

    row_sum = X.sum(
        axis=1
    )

    numerator = (
        (k - 1)
        * (
            k
            * np.sum(
                col_sum ** 2
            )
            -
            np.sum(
                col_sum
            ) ** 2
        )
    )

    denominator = (
        k
        * np.sum(
            row_sum
        )
        -
        np.sum(
            row_sum ** 2
        )
    )

    if denominator <= 0:
        return (
            np.nan,
            k - 1,
            np.nan,
        )

    Q = float(
        numerator
        / denominator
    )

    df_test = (
        k - 1
    )

    p = float(
        chi2.sf(
            Q,
            df_test,
        )
    )

    return (
        Q,
        df_test,
        p,
    )


def exact_mcnemar(
    x,
    y,
):
    """
    Two-sided exact McNemar test using discordant pairs.

    b = x=1, y=0
    c = x=0, y=1
    """

    x = np.asarray(
        x,
        dtype=int,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    b = int(
        np.sum(
            (x == 1)
            & (y == 0)
        )
    )

    c = int(
        np.sum(
            (x == 0)
            & (y == 1)
        )
    )

    discordant = (
        b + c
    )

    if discordant == 0:

        p = 1.0

    else:

        p = float(
            binomtest(
                min(b, c),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return (
        b,
        c,
        discordant,
        p,
    )


# ============================================================
# BUILD SLIDE EMBEDDINGS
# ============================================================

slide_records = []
metadata_records = []

print(
    "=" * 100
)

print(
    "BUILDING MULTISCALE "
    "SLIDE EMBEDDINGS"
)

print(
    "=" * 100
)


for model in MODELS:

    model_display = (
        MODEL_DISPLAY[
            model
        ]
    )

    print(
        f"\n[{model_display}]"
    )

    model_n = 0

    for cancer in CANCERS:

        prefixes = (
            get_slide_prefixes(
                cancer,
                model,
            )
        )

        expected = (
            EXPECTED_COUNTS[
                cancer
            ]
        )

        print(
            f"  {cancer}: "
            f"{len(prefixes)}/{expected}"
        )

        if (
            len(prefixes)
            != expected
        ):
            raise RuntimeError(
                f"{cancer}/{model}: "
                f"expected {expected}, "
                f"got {len(prefixes)}"
            )

        for prefix in prefixes:

            emb, counts = (
                build_slide_embedding(
                    cancer,
                    prefix,
                    model,
                )
            )

            slide_records.append({
                "cancer":
                    cancer,

                "case_id":
                    prefix,

                "model":
                    model,

                "model_display":
                    model_display,

                "embedding":
                    emb,
            })

            metadata_records.append({
                "cancer":
                    cancer,

                "case_id":
                    prefix,

                "model":
                    model,

                "model_display":
                    model_display,

                "embedding_dim":
                    len(emb),

                "n40x":
                    counts[
                        "40x"
                    ],

                "n10x":
                    counts[
                        "10x"
                    ],

                "n2p5x":
                    counts[
                        "2p5x"
                    ],
            })

            model_n += 1

    if model_n != EXPECTED_CASES:
        raise RuntimeError(
            f"{model}: "
            f"{model_n}/{EXPECTED_CASES}"
        )


metadata_df = pd.DataFrame(
    metadata_records
)

metadata_df.to_csv(
    OUT_METADATA,
    index=False,
)

print(
    f"\nTotal slide embeddings: "
    f"{len(slide_records)} "
    f"(expected "
    f"{EXPECTED_CASES * len(MODELS)})"
)


# ============================================================
# IMPORTANT CASE-ALIGNMENT AUDIT
# ============================================================

case_sets = {}

for model in MODELS:

    case_sets[
        model
    ] = set(
        (
            r["cancer"],
            r["case_id"],
        )
        for r in slide_records
        if r["model"] == model
    )

reference_model = (
    MODELS[0]
)

reference_cases = (
    case_sets[
        reference_model
    ]
)

for model in MODELS[1:]:

    if (
        case_sets[
            model
        ]
        != reference_cases
    ):

        missing = (
            reference_cases
            -
            case_sets[
                model
            ]
        )

        extra = (
            case_sets[
                model
            ]
            -
            reference_cases
        )

        raise RuntimeError(
            f"Case mismatch for {model}\n"
            f"missing={sorted(missing)}\n"
            f"extra={sorted(extra)}"
        )


# ============================================================
# LEAVE-ONE-OUT CANCER RETRIEVAL
# ============================================================

records = []

print(
    "\n"
    + "=" * 100
)

print(
    "LEAVE-ONE-OUT "
    "CANCER RETRIEVAL"
)

print(
    "=" * 100
)


for model in MODELS:

    model_display = (
        MODEL_DISPLAY[
            model
        ]
    )

    sub = [
        r
        for r in slide_records
        if r["model"] == model
    ]

    # deterministic common ordering
    sub = sorted(
        sub,
        key=lambda r: (
            r["cancer"],
            r["case_id"],
        ),
    )

    if (
        len(sub)
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            f"{model}: "
            f"expected 71 embeddings"
        )

    X = np.stack(
        [
            r["embedding"]
            for r in sub
        ],
        axis=0,
    )

    cancers = np.array(
        [
            r["cancer"]
            for r in sub
        ]
    )

    cases = np.array(
        [
            r["case_id"]
            for r in sub
        ]
    )

    sim = cosine_matrix(X)

    # leave-one-out:
    # query cannot retrieve itself
    np.fill_diagonal(
        sim,
        -np.inf,
    )

    for i in range(
        EXPECTED_CASES
    ):

        order = np.argsort(
            -sim[i],
            kind="stable",
        )

        for k in K_LIST:

            topk = order[
                :k
            ]

            same = (
                cancers[
                    topk
                ]
                ==
                cancers[
                    i
                ]
            )

            hit = int(
                np.any(
                    same
                )
            )

            # useful audit columns
            n_same = int(
                np.sum(
                    same
                )
            )

            records.append({
                "cancer":
                    cancers[i],

                "case_id":
                    cases[i],

                "model":
                    model,

                "model_display":
                    model_display,

                "k":
                    k,

                "metric":
                    f"Recall@{k}",

                "hit":
                    hit,

                "n_same_cancer_in_topk":
                    n_same,

                "top1_similarity":
                    float(
                        sim[
                            i,
                            order[0]
                        ]
                    ),

                "top1_cancer":
                    cancers[
                        order[0]
                    ],

                "top1_case":
                    cases[
                        order[0]
                    ],
            })


values_df = pd.DataFrame(
    records
)

expected_value_rows = (
    EXPECTED_CASES
    * len(MODELS)
    * len(K_LIST)
)

if (
    len(values_df)
    != expected_value_rows
):

    raise RuntimeError(
        f"Expected "
        f"{expected_value_rows} "
        f"retrieval records, "
        f"got {len(values_df)}"
    )


values_df.to_csv(
    OUT_VALUES,
    index=False,
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary = (
    values_df.groupby(
        [
            "model",
            "model_display",
            "k",
            "metric",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        n_cases=(
            "case_id",
            "size",
        ),

        n_hits=(
            "hit",
            "sum",
        ),

        recall=(
            "hit",
            "mean",
        ),
    )
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


# ============================================================
# CANCER-WISE SUMMARY
# ============================================================

cancer_summary = (
    values_df.groupby(
        [
            "cancer",
            "model",
            "model_display",
            "k",
            "metric",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        n_cases=(
            "case_id",
            "size",
        ),

        n_hits=(
            "hit",
            "sum",
        ),

        recall=(
            "hit",
            "mean",
        ),
    )
)

cancer_summary.to_csv(
    OUT_CANCER,
    index=False,
)


# ============================================================
# COMMON CASE KEY
# ============================================================

values_df[
    "subject_id"
] = (
    values_df[
        "cancer"
    ].astype(str)
    + "::"
    + values_df[
        "case_id"
    ].astype(str)
)


# ============================================================
# COCHRAN'S Q
# ============================================================

q_rows = []

for k in K_LIST:

    sub = values_df[
        values_df[
            "k"
        ] == k
    ]

    wide = (
        sub.pivot(
            index="subject_id",
            columns="model",
            values="hit",
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
            f"Recall@{k}: "
            f"bad paired matrix "
            f"{wide.shape}"
        )

    if wide.isna().any().any():
        raise RuntimeError(
            f"Recall@{k}: "
            f"missing paired values"
        )

    Q, df_test, p = (
        cochran_q_binary(
            wide.to_numpy(
                dtype=int
            )
        )
    )

    q_rows.append({
        "metric":
            f"Recall@{k}",

        "k":
            k,

        "n_cases":
            EXPECTED_CASES,

        "n_models":
            len(MODELS),

        "cochran_q":
            Q,

        "df":
            df_test,

        "p_value":
            p,

        "significance":
            p_to_stars(p),
    })


q_df = pd.DataFrame(
    q_rows
)

q_df.to_csv(
    OUT_Q,
    index=False,
)


# ============================================================
# PAIRWISE EXACT MCNEMAR + HOLM
# ============================================================

mcnemar_rows = []

for k in K_LIST:

    sub = values_df[
        values_df[
            "k"
        ] == k
    ]

    wide = (
        sub.pivot(
            index="subject_id",
            columns="model",
            values="hit",
        )
        .reindex(
            columns=MODELS
        )
    )

    local_rows = []

    for model_a, model_b in (
        itertools.combinations(
            MODELS,
            2,
        )
    ):

        x = wide[
            model_a
        ].to_numpy(
            dtype=int
        )

        y = wide[
            model_b
        ].to_numpy(
            dtype=int
        )

        b, c, discordant, p = (
            exact_mcnemar(
                x,
                y,
            )
        )

        local_rows.append({
            "metric":
                f"Recall@{k}",

            "k":
                k,

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

            "n_cases":
                EXPECTED_CASES,

            "recall_a":
                float(
                    np.mean(x)
                ),

            "recall_b":
                float(
                    np.mean(y)
                ),

            "difference_a_minus_b":
                float(
                    np.mean(x)
                    -
                    np.mean(y)
                ),

            "a_hit_b_miss":
                b,

            "a_miss_b_hit":
                c,

            "n_discordant":
                discordant,

            "p_raw":
                p,
        })

    local_df = pd.DataFrame(
        local_rows
    )

    local_df[
        "p_holm"
    ] = holm_adjust(
        local_df[
            "p_raw"
        ].to_numpy()
    )

    local_df[
        "significance_holm"
    ] = (
        local_df[
            "p_holm"
        ]
        .apply(
            p_to_stars
        )
    )

    mcnemar_rows.append(
        local_df
    )


mcnemar_df = pd.concat(
    mcnemar_rows,
    ignore_index=True,
)

mcnemar_df.to_csv(
    OUT_MCNEMAR,
    index=False,
)


# ============================================================
# PROTOCOL
# ============================================================

protocol = {
    "analysis":
        "biological_identity_preservation_cancer_retrieval",

    "cohort":
        "71 audited native-FOV cases across 8 TCGA cancer types",

    "models":
        MODELS,

    "scales":
        SCALES,

    "slide_embedding":
        [
            "Mean-pool patch embeddings independently at each scale.",
            "L2-normalize each scale-level mean vector.",
            "Equal-average the three normalized scale vectors.",
            "L2-normalize the final multiscale slide vector.",
        ],

    "retrieval":
        {
            "similarity":
                "cosine",

            "gallery":
                "all other slides represented by the same PFM",

            "self_match":
                "excluded",

            "mode":
                "leave-one-out slide-level retrieval",
        },

    "recall_definition":
        (
            "Recall@K = 1 for a query if at least one "
            "of its top-K retrieved slides has the same "
            "cancer label; otherwise 0."
        ),

    "k":
        K_LIST,

    "statistics":
        {
            "omnibus":
                (
                    "Cochran's Q across 8 paired models "
                    "separately for each K"
                ),

            "posthoc":
                (
                    "pairwise exact McNemar tests with "
                    "Holm correction across 28 model pairs "
                    "within each K"
                ),
        },
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
    "FIGURE 5A RETRIEVAL ANALYSIS COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nOVERALL RECALL"
)

display_summary = (
    summary.pivot(
        index="model_display",
        columns="metric",
        values="recall",
    )
)

# ranking by Recall@1
display_summary = (
    display_summary
    .sort_values(
        "Recall@1",
        ascending=False,
    )
)

print(
    display_summary
    .round(4)
    .to_string()
)


print(
    "\n"
    + "=" * 100
)

print(
    "COCHRAN'S Q"
)

print(
    "=" * 100
)

print(
    q_df[
        [
            "metric",
            "cochran_q",
            "df",
            "p_value",
            "significance",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "cochran_q":
                lambda x:
                f"{x:.4f}",

            "p_value":
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
    "\nTOP MODEL BY CANCER — Recall@1"
)

r1 = cancer_summary[
    cancer_summary[
        "k"
    ] == 1
]

for cancer in CANCERS:

    temp = (
        r1[
            r1[
                "cancer"
            ] == cancer
        ]
        .sort_values(
            "recall",
            ascending=False,
        )
    )

    best = temp.iloc[0]

    print(
        f"  {cancer}: "
        f"{best['model_display']} "
        f"({best['recall']:.4f})"
    )


print(
    "\nOutputs:"
)

for p in [
    OUT_VALUES,
    OUT_SUMMARY,
    OUT_CANCER,
    OUT_METADATA,
    OUT_Q,
    OUT_MCNEMAR,
    OUT_PROTOCOL,
]:
    print(p)
