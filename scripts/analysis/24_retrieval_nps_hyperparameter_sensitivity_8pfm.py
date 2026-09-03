#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis 24
Retrieval / NPS hyperparameter sensitivity
===========================================

Purpose
-------
Test whether PFM rankings depend strongly on reasonable
hyperparameter choices.

RETRIEVAL
---------
Primary:
    Q = 200 queries / direction
    Top-K = 5
    forward seed = 42
    reverse seed = 43

Sensitivity:
    Q in {100, 200}
    K in {1, 5, 10}

For each case/model/scale-pair:
    - exact bidirectional cosine retrieval
    - no ANN approximation
    - same original sampling logic

NPS
---
Primary:
    N <= 1000 sampled aligned rows
    k = 10
    seed = 42

Sensitivity:
    N in {500, 1000}
    k in {5, 10, 20}

For each case/model:
    - same sampled row indices at all three scales
    - exact cosine kNN
    - self excluded
    - three scale-pair NPS values

Primary-setting reproduction audit
----------------------------------
New:
    Retrieval Q=200,K=5
must reproduce:
    07_cross_scale_retrieval_similarity_8pfm_values.csv

New:
    NPS N=1000,k=10
must reproduce:
    16_neighborhood_preservation_score_8pfm_per_case_pair.csv

Ranking stability
-----------------
Analysis unit = PFM (n=8).

For every sensitivity setting:
    Spearman rho vs primary model means
    mean absolute rank displacement
    maximum absolute rank displacement
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import rankdata, spearmanr

import matplotlib.pyplot as plt


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

FIG_ROOT = (
    NATIVE_ROOT
    / "reviewer_analysis"
    / "figures"
)

FIG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RETENTION_FILE = (
    RESULT_ROOT
    / "02_relative_information_retention_8pfm_values.csv"
)

PRIMARY_RETRIEVAL_FILE = (
    RESULT_ROOT
    / "07_cross_scale_retrieval_similarity_8pfm_values.csv"
)

PRIMARY_NPS_FILE = (
    RESULT_ROOT
    / "16_neighborhood_preservation_score_8pfm_per_case_pair.csv"
)


PREFIX = "24_retrieval_nps_hyperparameter_sensitivity_8pfm"

OUT_RETRIEVAL_PAIR = (
    RESULT_ROOT
    / f"{PREFIX}_retrieval_per_case_pair.csv"
)

OUT_RETRIEVAL_SUMMARY = (
    RESULT_ROOT
    / f"{PREFIX}_retrieval_model_summary.csv"
)

OUT_RETRIEVAL_STABILITY = (
    RESULT_ROOT
    / f"{PREFIX}_retrieval_rank_stability.csv"
)

OUT_NPS_PAIR = (
    RESULT_ROOT
    / f"{PREFIX}_nps_per_case_pair.csv"
)

OUT_NPS_SUMMARY = (
    RESULT_ROOT
    / f"{PREFIX}_nps_model_summary.csv"
)

OUT_NPS_STABILITY = (
    RESULT_ROOT
    / f"{PREFIX}_nps_rank_stability.csv"
)

OUT_AUDIT = (
    RESULT_ROOT
    / f"{PREFIX}_primary_reproduction_audit.csv"
)

OUT_FIG_PNG = (
    FIG_ROOT
    / "FigureS_Hyperparameter_Sensitivity_8PFM.png"
)

OUT_FIG_PDF = (
    FIG_ROOT
    / "FigureS_Hyperparameter_Sensitivity_8PFM.pdf"
)


# ============================================================
# CONFIG
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
    ("40x", "10x"): "40× ↔ 10×",
    ("10x", "2p5x"): "10× ↔ 2.5×",
    ("40x", "2p5x"): "40× ↔ 2.5×",
}

EXPECTED_CASES = 71
EXPECTED_PAIR_ROWS = 71 * 8 * 3

RETRIEVAL_QUERIES = [
    100,
    200,
]

RETRIEVAL_K = [
    1,
    5,
    10,
]

PRIMARY_RETRIEVAL_Q = 200
PRIMARY_RETRIEVAL_K = 5

FORWARD_SEED = 42
REVERSE_SEED = 43


NPS_SAMPLE_BUDGETS = [
    500,
    1000,
]

NPS_K = [
    5,
    10,
    20,
]

PRIMARY_NPS_N = 1000
PRIMARY_NPS_K = 10

NPS_SEED = 42

PRIMARY_AUDIT_TOL = 2e-6


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--target-chunk",
        type=int,
        default=4096,
    )

    p.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
    )

    return p.parse_args()


# ============================================================
# DEVICE
# ============================================================

def get_device(name):

    if name == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(name)


# ============================================================
# COHORT
# ============================================================

def build_cohort():

    df = pd.read_csv(
        RETENTION_FILE
    )

    cohort = (
        df[
            [
                "cancer",
                "case_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "cancer",
                "case_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(cohort) != EXPECTED_CASES:

        raise RuntimeError(
            f"Expected 71 cases, "
            f"found {len(cohort)}"
        )

    return cohort


# ============================================================
# EMBEDDINGS
# ============================================================

def embedding_path(
    cancer,
    case_id,
    model,
    scale,
):

    return (
        EMBED_ROOT
        / cancer
        / f"{case_id}_{model}_{scale}.npy"
    )


def load_embedding(path):

    if not path.exists():
        raise FileNotFoundError(path)

    X = np.load(
        path,
        mmap_mode="r",
    )

    if X.ndim != 2:

        raise RuntimeError(
            f"Expected 2D embedding: "
            f"{path} shape={X.shape}"
        )

    if len(X) == 0:

        raise RuntimeError(
            f"Empty embedding: {path}"
        )

    return X


# ============================================================
# RETRIEVAL — EXACT ORIGINAL SAMPLING
# ============================================================

def sample_query_indices(
    n_rows,
    n_query,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    return rng.choice(
        np.arange(n_rows),
        size=min(
            n_query,
            n_rows,
        ),
        replace=False,
    )


def to_normalized_tensor(
    X,
    device,
):

    arr = np.asarray(
        X,
        dtype=np.float32,
    ).copy()

    t = torch.from_numpy(
        arr
    ).to(
        device=device,
        dtype=torch.float32,
    )

    norms = torch.linalg.vector_norm(
        t,
        ord=2,
        dim=1,
        keepdim=True,
    )

    norms = torch.where(
        norms == 0,
        torch.ones_like(norms),
        norms,
    )

    t.div_(
        norms
    )

    return t


@torch.inference_mode()
def exact_topmax_values(
    Xq_norm,
    Xt_norm,
    query_idx,
    max_k,
    target_chunk,
):

    n_t = int(
        Xt_norm.shape[0]
    )

    if n_t < max_k:

        raise RuntimeError(
            f"Target has {n_t} patches "
            f"but max_k={max_k}"
        )

    idx_t = torch.as_tensor(
        query_idx,
        dtype=torch.long,
        device=Xq_norm.device,
    )

    queries = Xq_norm[
        idx_t
    ]

    q = len(
        query_idx
    )

    best = torch.full(
        (
            q,
            max_k,
        ),
        -torch.inf,
        dtype=torch.float32,
        device=Xq_norm.device,
    )

    for start in range(
        0,
        n_t,
        target_chunk,
    ):

        end = min(
            start + target_chunk,
            n_t,
        )

        target = Xt_norm[
            start:end
        ]

        sims = (
            queries
            @ target.T
        )

        k_local = min(
            max_k,
            sims.shape[1],
        )

        local_best = torch.topk(
            sims,
            k=k_local,
            dim=1,
            largest=True,
            sorted=False,
        ).values

        candidates = torch.cat(
            [
                best,
                local_best,
            ],
            dim=1,
        )

        best = torch.topk(
            candidates,
            k=max_k,
            dim=1,
            largest=True,
            sorted=False,
        ).values

    return best


@torch.inference_mode()
def score_from_topmax(
    top_values,
    positions,
    k,
):

    pos_t = torch.as_tensor(
        positions,
        dtype=torch.long,
        device=top_values.device,
    )

    values = top_values[
        pos_t
    ]

    if k < values.shape[1]:

        values = torch.topk(
            values,
            k=k,
            dim=1,
            largest=True,
            sorted=False,
        ).values

    score = (
        values
        .mean(dim=1)
        .mean()
        .item()
    )

    return float(score)


def retrieval_direction_all_settings(
    Xq,
    Xt,
    seed,
    target_chunk,
):

    query_sets = {}

    for q_budget in RETRIEVAL_QUERIES:

        query_sets[
            q_budget
        ] = sample_query_indices(
            n_rows=len(Xq),
            n_query=q_budget,
            seed=seed,
        )

    # Q=100 and Q=200 are independently sampled
    # using the original function.
    # Use their union only to avoid duplicate similarity
    # computation; the individual query sets remain unchanged.
    union_idx = np.unique(
        np.concatenate(
            list(
                query_sets.values()
            )
        )
    )

    topmax = exact_topmax_values(
        Xq,
        Xt,
        query_idx=union_idx,
        max_k=max(
            RETRIEVAL_K
        ),
        target_chunk=target_chunk,
    )

    scores = {}

    for q_budget, idx in (
        query_sets.items()
    ):

        positions = np.searchsorted(
            union_idx,
            idx,
        )

        for k in RETRIEVAL_K:

            scores[
                (
                    q_budget,
                    k,
                )
            ] = score_from_topmax(
                topmax,
                positions,
                k,
            )

    return (
        scores,
        query_sets,
    )


# ============================================================
# RETRIEVAL SENSITIVITY
# ============================================================

def run_retrieval(
    cohort,
    device,
    target_chunk,
):

    records = []

    total = (
        EXPECTED_CASES
        * len(MODELS)
        * len(SCALE_PAIRS)
    )

    completed = 0

    start_all = time.time()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RETRIEVAL SENSITIVITY"
    )

    print(
        "=" * 100
    )

    for model in MODELS:

        print(
            f"\n[Retrieval] "
            f"{MODEL_DISPLAY[model]}"
        )

        for _, row in (
            cohort.iterrows()
        ):

            cancer = row[
                "cancer"
            ]

            case_id = row[
                "case_id"
            ]

            for s1, s2 in SCALE_PAIRS:

                pair_start = time.time()

                X1_np = load_embedding(
                    embedding_path(
                        cancer,
                        case_id,
                        model,
                        s1,
                    )
                )

                X2_np = load_embedding(
                    embedding_path(
                        cancer,
                        case_id,
                        model,
                        s2,
                    )
                )

                n1 = len(
                    X1_np
                )

                n2 = len(
                    X2_np
                )

                if (
                    n1 < max(
                        RETRIEVAL_K
                    )
                    or
                    n2 < max(
                        RETRIEVAL_K
                    )
                ):

                    raise RuntimeError(
                        f"Too few patches: "
                        f"{case_id}/{model}/"
                        f"{s1}-{s2}"
                    )

                X1 = to_normalized_tensor(
                    X1_np,
                    device,
                )

                X2 = to_normalized_tensor(
                    X2_np,
                    device,
                )

                del X1_np
                del X2_np

                forward, q_forward = (
                    retrieval_direction_all_settings(
                        X1,
                        X2,
                        seed=FORWARD_SEED,
                        target_chunk=target_chunk,
                    )
                )

                reverse, q_reverse = (
                    retrieval_direction_all_settings(
                        X2,
                        X1,
                        seed=REVERSE_SEED,
                        target_chunk=target_chunk,
                    )
                )

                for q_budget in (
                    RETRIEVAL_QUERIES
                ):

                    for k in RETRIEVAL_K:

                        score_forward = forward[
                            (
                                q_budget,
                                k,
                            )
                        ]

                        score_reverse = reverse[
                            (
                                q_budget,
                                k,
                            )
                        ]

                        score = (
                            score_forward
                            + score_reverse
                        ) / 2.0

                        records.append({
                            "cancer":
                                cancer,

                            "case_id":
                                case_id,

                            "model":
                                model,

                            "model_display":
                                MODEL_DISPLAY[
                                    model
                                ],

                            "scale_a":
                                s1,

                            "scale_b":
                                s2,

                            "scale_pair":
                                PAIR_LABEL[
                                    (
                                        s1,
                                        s2,
                                    )
                                ],

                            "query_budget":
                                q_budget,

                            "top_k":
                                k,

                            "setting":
                                f"Q{q_budget}_K{k}",

                            "n_patches_a":
                                n1,

                            "n_patches_b":
                                n2,

                            "n_query_a_to_b":
                                len(
                                    q_forward[
                                        q_budget
                                    ]
                                ),

                            "n_query_b_to_a":
                                len(
                                    q_reverse[
                                        q_budget
                                    ]
                                ),

                            "score_a_to_b":
                                score_forward,

                            "score_b_to_a":
                                score_reverse,

                            "retrieval_similarity":
                                score,
                        })

                del X1
                del X2

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                completed += 1

                if (
                    completed % 25 == 0
                    or
                    completed == total
                ):

                    elapsed = (
                        time.time()
                        - start_all
                    )

                    rate = (
                        completed
                        / elapsed
                    )

                    eta = (
                        (
                            total
                            - completed
                        )
                        / rate
                        / 60
                    )

                    print(
                        f"  {completed:4d}/"
                        f"{total} | "
                        f"ETA {eta:6.1f} min",
                        flush=True,
                    )

    result = pd.DataFrame(
        records
    )

    expected = (
        EXPECTED_PAIR_ROWS
        * len(
            RETRIEVAL_QUERIES
        )
        * len(
            RETRIEVAL_K
        )
    )

    if len(result) != expected:

        raise RuntimeError(
            f"Retrieval rows "
            f"{len(result)} != {expected}"
        )

    result.to_csv(
        OUT_RETRIEVAL_PAIR,
        index=False,
    )

    return result


# ============================================================
# NPS
# ============================================================

def normalize_torch_nps(X):

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
    device,
):

    X = torch.as_tensor(
        np.asarray(
            X_np,
            dtype=np.float32,
        ),
        device=device,
    )

    X = normalize_torch_nps(
        X
    )

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

    del sim
    del X

    return idx


@torch.inference_mode()
def overlap_score(
    nn_a,
    nn_b,
):

    overlap = (
        nn_a.unsqueeze(2)
        ==
        nn_b.unsqueeze(1)
    )

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


def sample_nps_indices(
    n_total,
    budget,
):

    rng = np.random.default_rng(
        NPS_SEED
    )

    if n_total > budget:

        return rng.choice(
            np.arange(
                n_total
            ),
            size=budget,
            replace=False,
        )

    return np.arange(
        n_total
    )


# ============================================================
# NPS SENSITIVITY
# ============================================================

def run_nps(
    cohort,
    device,
):

    records = []

    total = (
        EXPECTED_CASES
        * len(MODELS)
    )

    completed = 0

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NPS SENSITIVITY"
    )

    print(
        "=" * 100
    )

    for model in MODELS:

        print(
            f"\n[NPS] "
            f"{MODEL_DISPLAY[model]}"
        )

        for _, row in (
            cohort.iterrows()
        ):

            cancer = row[
                "cancer"
            ]

            case_id = row[
                "case_id"
            ]

            arrays = {}

            row_counts = {}

            for scale in SCALES:

                arr = load_embedding(
                    embedding_path(
                        cancer,
                        case_id,
                        model,
                        scale,
                    )
                )

                arrays[
                    scale
                ] = arr

                row_counts[
                    scale
                ] = len(
                    arr
                )

            if len(
                set(
                    row_counts.values()
                )
            ) != 1:

                raise RuntimeError(
                    f"ROW MISMATCH: "
                    f"{case_id}/{model}: "
                    f"{row_counts}"
                )

            n_total = next(
                iter(
                    row_counts.values()
                )
            )

            if (
                n_total
                <=
                max(
                    NPS_K
                )
            ):

                raise RuntimeError(
                    f"Too few patches: "
                    f"{case_id}/{model}: "
                    f"{n_total}"
                )

            for budget in (
                NPS_SAMPLE_BUDGETS
            ):

                sample_idx = (
                    sample_nps_indices(
                        n_total,
                        budget,
                    )
                )

                if (
                    len(sample_idx)
                    <=
                    max(
                        NPS_K
                    )
                ):

                    raise RuntimeError(
                        f"Sample too small: "
                        f"{case_id}/{model}/"
                        f"N={budget}"
                    )

                # IMPORTANT:
                # Compute each k independently using the exact
                # original Script-16 kNN routine.
                #
                # Do NOT compute Top-20 once and slice [:, :k].
                # torch.topk can select different members at
                # near-tied boundary values when called with
                # different k, which can prevent exact
                # reproduction of the primary k=10 setting.
                knn_by_k = {
                    k: {}
                    for k in NPS_K
                }

                for scale in SCALES:

                    sampled = np.asarray(
                        arrays[
                            scale
                        ][sample_idx],
                        dtype=np.float32,
                    )

                    for k in NPS_K:

                        knn_by_k[
                            k
                        ][
                            scale
                        ] = knn_indices_gpu(
                            sampled,
                            k=k,
                            device=device,
                        )

                for s1, s2 in (
                    SCALE_PAIRS
                ):

                    for k in NPS_K:

                        score = overlap_score(
                            knn_by_k[
                                k
                            ][
                                s1
                            ],
                            knn_by_k[
                                k
                            ][
                                s2
                            ],
                        )

                        records.append({
                            "cancer":
                                cancer,

                            "case_id":
                                case_id,

                            "model":
                                model,

                            "model_display":
                                MODEL_DISPLAY[
                                    model
                                ],

                            "scale_a":
                                s1,

                            "scale_b":
                                s2,

                            "scale_pair":
                                PAIR_LABEL[
                                    (
                                        s1,
                                        s2,
                                    )
                                ],

                            "sample_budget":
                                budget,

                            "n_sampled_patches":
                                len(
                                    sample_idx
                                ),

                            "k_neighbors":
                                k,

                            "setting":
                                f"N{budget}_k{k}",

                            "n_total_patches":
                                n_total,

                            "nps":
                                score,
                        })

                del knn_by_k

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del arrays

            gc.collect()

            completed += 1

            if (
                completed % 25 == 0
                or
                completed == total
            ):

                print(
                    f"  {completed:4d}/"
                    f"{total}",
                    flush=True,
                )

    result = pd.DataFrame(
        records
    )

    expected = (
        EXPECTED_PAIR_ROWS
        * len(
            NPS_SAMPLE_BUDGETS
        )
        * len(
            NPS_K
        )
    )

    if len(result) != expected:

        raise RuntimeError(
            f"NPS rows "
            f"{len(result)} != {expected}"
        )

    result.to_csv(
        OUT_NPS_PAIR,
        index=False,
    )

    return result


# ============================================================
# PRIMARY REPRODUCTION AUDIT
# ============================================================

def primary_audit(
    retrieval,
    nps,
):

    audit_rows = []


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    old = pd.read_csv(
        PRIMARY_RETRIEVAL_FILE
    )

    new = retrieval[
        (
            retrieval[
                "query_budget"
            ]
            == PRIMARY_RETRIEVAL_Q
        )
        &
        (
            retrieval[
                "top_k"
            ]
            == PRIMARY_RETRIEVAL_K
        )
    ].copy()

    keys = [
        "cancer",
        "case_id",
        "model",
        "scale_a",
        "scale_b",
    ]

    merged = old[
        keys
        + [
            "retrieval_similarity"
        ]
    ].merge(
        new[
            keys
            + [
                "retrieval_similarity"
            ]
        ],
        on=keys,
        how="inner",
        suffixes=(
            "_old",
            "_new",
        ),
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_PAIR_ROWS:

        raise RuntimeError(
            f"Retrieval audit rows "
            f"{len(merged)} != "
            f"{EXPECTED_PAIR_ROWS}"
        )

    delta = np.abs(
        merged[
            "retrieval_similarity_new"
        ]
        -
        merged[
            "retrieval_similarity_old"
        ]
    )

    audit_rows.append({
        "metric":
            "Retrieval",

        "primary_setting":
            "Q200_K5",

        "n_rows":
            len(merged),

        "mean_abs_delta":
            float(
                delta.mean()
            ),

        "max_abs_delta":
            float(
                delta.max()
            ),

        "tolerance":
            PRIMARY_AUDIT_TOL,

        "passed":
            bool(
                delta.max()
                <=
                PRIMARY_AUDIT_TOL
            ),
    })


    # --------------------------------------------------------
    # NPS
    # --------------------------------------------------------

    old = pd.read_csv(
        PRIMARY_NPS_FILE
    )

    new = nps[
        (
            nps[
                "sample_budget"
            ]
            == PRIMARY_NPS_N
        )
        &
        (
            nps[
                "k_neighbors"
            ]
            == PRIMARY_NPS_K
        )
    ].copy()

    merged = old[
        keys
        + [
            "nps"
        ]
    ].merge(
        new[
            keys
            + [
                "nps"
            ]
        ],
        on=keys,
        how="inner",
        suffixes=(
            "_old",
            "_new",
        ),
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_PAIR_ROWS:

        raise RuntimeError(
            f"NPS audit rows "
            f"{len(merged)} != "
            f"{EXPECTED_PAIR_ROWS}"
        )

    delta = np.abs(
        merged[
            "nps_new"
        ]
        -
        merged[
            "nps_old"
        ]
    )

    audit_rows.append({
        "metric":
            "NPS",

        "primary_setting":
            "N1000_k10",

        "n_rows":
            len(merged),

        "mean_abs_delta":
            float(
                delta.mean()
            ),

        "max_abs_delta":
            float(
                delta.max()
            ),

        "tolerance":
            PRIMARY_AUDIT_TOL,

        "passed":
            bool(
                delta.max()
                <=
                PRIMARY_AUDIT_TOL
            ),
    })


    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUT_AUDIT,
        index=False,
    )

    return audit


# ============================================================
# MODEL SUMMARY
# ============================================================

def summarize_retrieval(
    df,
):

    summary = (
        df.groupby(
            [
                "setting",
                "query_budget",
                "top_k",
                "model",
                "model_display",
            ],
            as_index=False,
        )
        .agg(
            mean_score=(
                "retrieval_similarity",
                "mean",
            ),

            sd_score=(
                "retrieval_similarity",
                "std",
            ),

            median_score=(
                "retrieval_similarity",
                "median",
            ),

            n_case_pairs=(
                "retrieval_similarity",
                "size",
            ),
        )
    )

    summary[
        "rank"
    ] = (
        summary.groupby(
            "setting"
        )[
            "mean_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    summary.to_csv(
        OUT_RETRIEVAL_SUMMARY,
        index=False,
    )

    return summary


def summarize_nps(
    df,
):

    summary = (
        df.groupby(
            [
                "setting",
                "sample_budget",
                "k_neighbors",
                "model",
                "model_display",
            ],
            as_index=False,
        )
        .agg(
            mean_score=(
                "nps",
                "mean",
            ),

            sd_score=(
                "nps",
                "std",
            ),

            median_score=(
                "nps",
                "median",
            ),

            n_case_pairs=(
                "nps",
                "size",
            ),
        )
    )

    summary[
        "rank"
    ] = (
        summary.groupby(
            "setting"
        )[
            "mean_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    summary.to_csv(
        OUT_NPS_SUMMARY,
        index=False,
    )

    return summary


# ============================================================
# RANK STABILITY
# ============================================================

def rank_stability(
    summary,
    primary_setting,
    output_path,
):

    primary = (
        summary[
            summary[
                "setting"
            ]
            == primary_setting
        ]
        .set_index(
            "model"
        )
        .reindex(
            MODELS
        )
    )

    if len(primary) != 8:

        raise RuntimeError(
            f"Primary setting "
            f"{primary_setting} "
            f"does not contain 8 models."
        )

    primary_scores = (
        primary[
            "mean_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    primary_ranks = (
        primary[
            "rank"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rows = []

    for setting in (
        summary[
            "setting"
        ]
        .drop_duplicates()
    ):

        current = (
            summary[
                summary[
                    "setting"
                ]
                == setting
            ]
            .set_index(
                "model"
            )
            .reindex(
                MODELS
            )
        )

        scores = (
            current[
                "mean_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        ranks = (
            current[
                "rank"
            ]
            .to_numpy(
                dtype=float
            )
        )

        rho = float(
            spearmanr(
                primary_scores,
                scores,
            ).statistic
        )

        rank_delta = np.abs(
            ranks
            -
            primary_ranks
        )

        rows.append({
            "setting":
                setting,

            "primary_setting":
                primary_setting,

            "spearman_vs_primary":
                rho,

            "mean_abs_rank_displacement":
                float(
                    rank_delta.mean()
                ),

            "max_abs_rank_displacement":
                float(
                    rank_delta.max()
                ),

            "n_models_changed_rank":
                int(
                    np.sum(
                        rank_delta > 0
                    )
                ),

            "same_top_model":
                bool(
                    np.argmin(
                        ranks
                    )
                    ==
                    np.argmin(
                        primary_ranks
                    )
                ),

            "mean_abs_model_score_difference":
                float(
                    np.mean(
                        np.abs(
                            scores
                            -
                            primary_scores
                        )
                    )
                ),
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


# ============================================================
# FIGURE
# ============================================================

def make_figure(
    retrieval_stability,
    nps_stability,
):

    plt.rcParams.update({
        "font.family":
            "DejaVu Sans",

        "font.size":
            9,

        "axes.titlesize":
            11,

        "axes.labelsize":
            10,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.2,
            4.7,
        )
    )


    for ax, data, title, primary in [
        (
            axes[0],
            retrieval_stability,
            "A   Retrieval sensitivity",
            "Q200_K5",
        ),
        (
            axes[1],
            nps_stability,
            "B   NPS sensitivity",
            "N1000_k10",
        ),
    ]:

        plot = (
            data[
                data[
                    "setting"
                ]
                != primary
            ]
            .sort_values(
                "spearman_vs_primary",
                ascending=True,
            )
        )

        y = np.arange(
            len(plot)
        )

        ax.barh(
            y,
            plot[
                "spearman_vs_primary"
            ],
        )

        ax.set_yticks(
            y
        )

        ax.set_yticklabels(
            plot[
                "setting"
            ]
        )

        ax.set_xlim(
            0,
            1.03,
        )

        ax.axvline(
            0.9,
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )

        for yi, rho in zip(
            y,
            plot[
                "spearman_vs_primary"
            ],
        ):

            ax.text(
                min(
                    rho + 0.015,
                    0.98,
                ),
                yi,
                f"{rho:.2f}",
                va="center",
                fontsize=8,
            )

        ax.set_xlabel(
            "Spearman rank correlation\n"
            "with primary 8-PFM ranking"
        )

        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.grid(
            axis="x",
            alpha=0.15,
        )

    fig.text(
        0.01,
        0.01,
        (
            "Dashed line: ρ = 0.90. "
            "Correlations are calculated across the eight PFM-level mean scores; "
            "sensitivity analyses assess ranking stability rather than constituting "
            "additional hypothesis tests."
        ),
        fontsize=7.8,
        ha="left",
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.19,
        top=0.90,
        wspace=0.45,
    )

    fig.savefig(
        OUT_FIG_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_FIG_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# REPORT MODEL RANKS
# ============================================================

def print_rank_table(
    summary,
    metric_name,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{metric_name.upper()} MODEL RANKINGS"
    )

    print(
        "=" * 100
    )

    table = (
        summary.pivot(
            index="model_display",
            columns="setting",
            values="rank",
        )
    )

    print(
        table.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    device = get_device(
        args.device
    )

    print(
        "=" * 110
    )

    print(
        "RETRIEVAL / NPS HYPERPARAMETER SENSITIVITY"
    )

    print(
        "=" * 110
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Target chunk : {args.target_chunk}"
    )

    print(
        "\nRetrieval settings:"
    )

    for q in RETRIEVAL_QUERIES:

        for k in RETRIEVAL_K:

            print(
                f"  Q={q}, K={k}"
            )

    print(
        "\nNPS settings:"
    )

    for n in NPS_SAMPLE_BUDGETS:

        for k in NPS_K:

            print(
                f"  N={n}, k={k}"
            )

    cohort = build_cohort()


    # ========================================================
    # CALCULATE
    # ========================================================

    retrieval = run_retrieval(
        cohort,
        device,
        args.target_chunk,
    )

    nps = run_nps(
        cohort,
        device,
    )


    # ========================================================
    # PRIMARY REPRODUCTION
    # ========================================================

    audit = primary_audit(
        retrieval,
        nps,
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "PRIMARY REPRODUCTION AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        audit
        .to_string(
            index=False
        )
    )

    if not audit[
        "passed"
    ].all():

        raise RuntimeError(
            "Primary-setting reproduction audit failed. "
            "Do not interpret sensitivity results."
        )


    # ========================================================
    # SUMMARIES
    # ========================================================

    retrieval_summary = (
        summarize_retrieval(
            retrieval
        )
    )

    nps_summary = (
        summarize_nps(
            nps
        )
    )


    # ========================================================
    # STABILITY
    # ========================================================

    retrieval_stability = (
        rank_stability(
            retrieval_summary,
            primary_setting="Q200_K5",
            output_path=
                OUT_RETRIEVAL_STABILITY,
        )
    )

    nps_stability = (
        rank_stability(
            nps_summary,
            primary_setting="N1000_k10",
            output_path=
                OUT_NPS_STABILITY,
        )
    )


    # ========================================================
    # FIGURE
    # ========================================================

    make_figure(
        retrieval_stability,
        nps_stability,
    )


    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "RETRIEVAL RANK STABILITY"
    )

    print(
        "=" * 110
    )

    print(
        retrieval_stability
        .sort_values(
            "setting"
        )
        .round(6)
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "NPS RANK STABILITY"
    )

    print(
        "=" * 110
    )

    print(
        nps_stability
        .sort_values(
            "setting"
        )
        .round(6)
        .to_string(
            index=False
        )
    )


    print_rank_table(
        retrieval_summary,
        "Retrieval"
    )

    print_rank_table(
        nps_summary,
        "NPS"
    )


    print(
        "\nOutputs:"
    )

    for path in [
        OUT_RETRIEVAL_PAIR,
        OUT_RETRIEVAL_SUMMARY,
        OUT_RETRIEVAL_STABILITY,
        OUT_NPS_PAIR,
        OUT_NPS_SUMMARY,
        OUT_NPS_STABILITY,
        OUT_AUDIT,
        OUT_FIG_PNG,
        OUT_FIG_PDF,
    ]:

        print(path)


if __name__ == "__main__":
    main()
