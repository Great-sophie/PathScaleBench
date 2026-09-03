#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8-PFM Cross-scale Retrieval Similarity
======================================

Faithful re-analysis of the original submitted Figure 4B metric.

For each slide/model/scale pair A <-> B:

1. Randomly sample up to 200 query patches from A (seed=42).
2. L2-normalize query and all B embeddings.
3. For each query patch, retrieve the Top-5 patches from B by cosine similarity.
4. Average the Top-5 similarities.
5. Average across query patches -> S(A -> B).
6. Repeat B -> A using seed=43.
7. Bidirectional retrieval similarity:

       S(A <-> B) = [S(A -> B) + S(B -> A)] / 2

Revision protocol:
- 71 audited native-FOV same-center cases
- 8 PFMs
- 3 scale pairs
- FP32
- exact Top-K retrieval
- same metric parameters as original Figure 4B

Important:
This metric does NOT require positional row correspondence across scales.
It is not Recall@K and is not Neighborhood Preservation Score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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

PREFIX = "07_cross_scale_retrieval_similarity_8pfm"

OUT_VALUES = (
    RESULT_ROOT
    / f"{PREFIX}_values.csv"
)

OUT_SUMMARY_PAIR = (
    RESULT_ROOT
    / f"{PREFIX}_summary_by_pair.csv"
)

OUT_SUMMARY_MODEL = (
    RESULT_ROOT
    / f"{PREFIX}_summary_by_model.csv"
)

OUT_CANCER = (
    RESULT_ROOT
    / f"{PREFIX}_cancer_summary.csv"
)

OUT_PROTOCOL = (
    RESULT_ROOT
    / f"{PREFIX}_protocol.json"
)


# ============================================================
# COHORT
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

EXPECTED_CASES = 71


# ============================================================
# MODELS
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


# ============================================================
# RETRIEVAL PROTOCOL — LOCKED TO ORIGINAL FIGURE 4B
# ============================================================

SCALE_PAIRS = [
    ("40x", "10x"),
    ("10x", "2p5x"),
    ("40x", "2p5x"),
]

PAIR_LABELS = {
    ("40x", "10x"):
        "40× ↔ 10×",

    ("10x", "2p5x"):
        "10× ↔ 2.5×",

    ("40x", "2p5x"):
        "40× ↔ 2.5×",
}

PAIR_ORDER = [
    "40× ↔ 10×",
    "10× ↔ 2.5×",
    "40× ↔ 2.5×",
]

TOP_K = 5
N_QUERY_PER_SLIDE = 200
FORWARD_SEED = 42
REVERSE_SEED = 43


# ============================================================
# HELPERS
# ============================================================

def get_case_prefixes(
    cancer_dir: Path,
    model: str,
):
    pattern = f"_{model}_40x.npy"

    if not cancer_dir.exists():
        return []

    prefixes = []

    for p in cancer_dir.glob(
        f"*{pattern}"
    ):
        name = p.name

        if name.endswith(pattern):
            prefixes.append(
                name[:-len(pattern)]
            )

    return sorted(prefixes)


def embedding_path(
    cancer_dir: Path,
    prefix: str,
    model: str,
    scale: str,
):
    return (
        cancer_dir
        / f"{prefix}_{model}_{scale}.npy"
    )


def load_embedding(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            str(path)
        )

    X = np.load(
        path,
        mmap_mode="r",
    )

    if X.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embedding: "
            f"{path} | shape={X.shape}"
        )

    if len(X) == 0:
        raise RuntimeError(
            f"Empty embedding: {path}"
        )

    return X


def sample_query_indices(
    n_rows: int,
    n_query: int,
    seed: int,
):
    """
    EXACT original sampling logic:

        rng = np.random.default_rng(seed)
        rng.choice(
            np.arange(n_q),
            size=min(n_query, n_q),
            replace=False
        )
    """

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
    """
    Convert embedding matrix to FP32 GPU/CPU tensor
    and perform row-wise L2 normalization.

    Equivalent to original NumPy normalize_features().
    """

    # Explicit copy avoids read-only mmap warnings.
    arr = np.asarray(
        X,
        dtype=np.float32,
    ).copy()

    t = torch.from_numpy(
        arr
    ).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )

    norms = torch.linalg.vector_norm(
        t,
        ord=2,
        dim=1,
        keepdim=True,
    )

    # Original code:
    # norm[norm == 0] = 1.0
    norms = torch.where(
        norms == 0,
        torch.ones_like(norms),
        norms,
    )

    t.div_(norms)

    return t


@torch.inference_mode()
def mean_topk_similarity_exact(
    Xq_norm: torch.Tensor,
    Xt_norm: torch.Tensor,
    query_idx: np.ndarray,
    top_k: int,
    target_chunk: int,
):
    """
    Exact Top-K cosine similarity.

    Uses target chunking only to reduce memory.

    The result is mathematically equivalent to:

        sims = Xt_norm @ query
        idx = np.argpartition(-sims, top_k-1)[:top_k]
        mean(topk similarities)

    No approximation / ANN is used.
    """

    n_t = Xt_norm.shape[0]

    if n_t < top_k:
        return None

    idx_t = torch.as_tensor(
        query_idx,
        dtype=torch.long,
        device=Xq_norm.device,
    )

    queries = Xq_norm[
        idx_t
    ]

    q = queries.shape[0]

    # Running exact Top-K for each query.
    best = torch.full(
        (q, top_k),
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

        # [Q, D] @ [D, T] -> [Q, T]
        sims = (
            queries
            @ target.T
        )

        k_local = min(
            top_k,
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
            k=top_k,
            dim=1,
            largest=True,
            sorted=False,
        ).values

    # First mean = Top-K per query.
    # Second mean = across queries.
    score = (
        best.mean(dim=1)
        .mean()
        .item()
    )

    return float(score)


def bootstrap_mean_ci(
    values,
    n_boot=5000,
    seed=2026,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            len(x),
            size=len(x),
        )

        means[i] = np.mean(
            x[idx]
        )

    lo, hi = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(lo), float(hi)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    p.add_argument(
        "--target-chunk",
        type=int,
        default=8192,
        help=(
            "Target rows processed per exact "
            "cosine-similarity chunk."
        ),
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
    )

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    print("=" * 96)
    print(
        "8-PFM CROSS-SCALE RETRIEVAL SIMILARITY"
    )
    print("=" * 96)

    print(
        f"Embedding root : {EMBED_ROOT}"
    )

    print(
        f"Device         : {device}"
    )

    if device.type == "cuda":
        print(
            "GPU            : "
            + torch.cuda.get_device_name(
                device
            )
        )

    print(
        f"Top-K          : {TOP_K}"
    )

    print(
        f"Queries/dir    : {N_QUERY_PER_SLIDE}"
    )

    print(
        f"Forward seed   : {FORWARD_SEED}"
    )

    print(
        f"Reverse seed   : {REVERSE_SEED}"
    )

    print(
        f"Target chunk   : {args.target_chunk}"
    )

    print(
        "Metric         : bidirectional "
        "mean Top-5 cosine similarity"
    )

    print("=" * 96)

    records = []

    total_expected = (
        EXPECTED_CASES
        * len(MODELS)
        * len(SCALE_PAIRS)
    )

    completed = 0

    global_start = time.time()

    # ========================================================
    # COMPUTE
    # ========================================================

    for cancer in CANCERS:

        cancer_dir = (
            EMBED_ROOT
            / cancer
        )

        print(
            f"\n===== {cancer} ====="
        )

        for model in MODELS:

            prefixes = get_case_prefixes(
                cancer_dir,
                model,
            )

            expected_n = (
                EXPECTED_COUNTS[cancer]
            )

            if len(prefixes) != expected_n:
                raise RuntimeError(
                    f"{cancer}/{model}: "
                    f"detected {len(prefixes)} cases, "
                    f"expected {expected_n}"
                )

            print(
                f"  {MODEL_DISPLAY[model]:15s} "
                f"cases={len(prefixes)}"
            )

            for prefix in prefixes:

                for s1, s2 in SCALE_PAIRS:

                    pair_start = time.time()

                    p1 = embedding_path(
                        cancer_dir,
                        prefix,
                        model,
                        s1,
                    )

                    p2 = embedding_path(
                        cancer_dir,
                        prefix,
                        model,
                        s2,
                    )

                    X1_np = load_embedding(
                        p1
                    )

                    X2_np = load_embedding(
                        p2
                    )

                    n1 = len(X1_np)
                    n2 = len(X2_np)

                    if (
                        n1 < 10
                        or n2 < TOP_K
                    ):
                        raise RuntimeError(
                            f"Too few patches: "
                            f"{cancer}/{prefix}/{model}/"
                            f"{s1}-{s2} | "
                            f"n1={n1}, n2={n2}"
                        )

                    # Transfer and normalize both full
                    # scale matrices once.
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

                    # ------------------------------
                    # Forward: s1 -> s2
                    # ------------------------------

                    qidx_12 = (
                        sample_query_indices(
                            n_rows=n1,
                            n_query=N_QUERY_PER_SLIDE,
                            seed=FORWARD_SEED,
                        )
                    )

                    score_12 = (
                        mean_topk_similarity_exact(
                            X1,
                            X2,
                            qidx_12,
                            top_k=TOP_K,
                            target_chunk=args.target_chunk,
                        )
                    )

                    # ------------------------------
                    # Reverse: s2 -> s1
                    # ------------------------------

                    qidx_21 = (
                        sample_query_indices(
                            n_rows=n2,
                            n_query=N_QUERY_PER_SLIDE,
                            seed=REVERSE_SEED,
                        )
                    )

                    score_21 = (
                        mean_topk_similarity_exact(
                            X2,
                            X1,
                            qidx_21,
                            top_k=TOP_K,
                            target_chunk=args.target_chunk,
                        )
                    )

                    if (
                        score_12 is None
                        or score_21 is None
                    ):
                        raise RuntimeError(
                            "Retrieval returned None."
                        )

                    score = (
                        score_12
                        + score_21
                    ) / 2.0

                    elapsed = (
                        time.time()
                        - pair_start
                    )

                    records.append({
                        "cancer":
                            cancer,

                        "case_id":
                            prefix,

                        "model":
                            model,

                        "model_display":
                            MODEL_DISPLAY[model],

                        "scale_a":
                            s1,

                        "scale_b":
                            s2,

                        "scale_pair":
                            PAIR_LABELS[
                                (s1, s2)
                            ],

                        "n_patches_a":
                            n1,

                        "n_patches_b":
                            n2,

                        "n_query_a_to_b":
                            len(qidx_12),

                        "n_query_b_to_a":
                            len(qidx_21),

                        "top_k":
                            TOP_K,

                        "score_a_to_b":
                            score_12,

                        "score_b_to_a":
                            score_21,

                        "retrieval_similarity":
                            score,

                        "elapsed_sec":
                            elapsed,
                    })

                    completed += 1

                    total_elapsed = (
                        time.time()
                        - global_start
                    )

                    rate = (
                        completed
                        / total_elapsed
                    )

                    remaining = (
                        total_expected
                        - completed
                    )

                    eta_min = (
                        remaining
                        / max(
                            rate,
                            1e-12,
                        )
                        / 60.0
                    )

                    print(
                        f"    "
                        f"{completed:4d}/"
                        f"{total_expected} | "
                        f"{prefix[:18]:18s} | "
                        f"{s1:4s}<->{s2:4s} | "
                        f"{score:.4f} | "
                        f"{elapsed:5.2f}s | "
                        f"ETA {eta_min:6.1f} min"
                    )

                    del X1
                    del X2

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    # ========================================================
    # VALIDATE
    # ========================================================

    values = pd.DataFrame(
        records
    )

    if len(values) != total_expected:
        raise RuntimeError(
            f"Expected {total_expected} records, "
            f"got {len(values)}"
        )

    if not np.isfinite(
        values[
            "retrieval_similarity"
        ].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError(
            "Non-finite retrieval scores."
        )

    # Every model must have exactly
    # 71 cases × 3 scale pairs.
    model_counts = (
        values.groupby(
            "model",
            observed=True,
        )
        .size()
    )

    for model in MODELS:

        expected = (
            EXPECTED_CASES
            * len(SCALE_PAIRS)
        )

        observed = int(
            model_counts.get(
                model,
                0,
            )
        )

        if observed != expected:
            raise RuntimeError(
                f"{model}: "
                f"{observed}/{expected}"
            )

    values.to_csv(
        OUT_VALUES,
        index=False,
    )

    # ========================================================
    # SUMMARY — MODEL × SCALE PAIR
    # ========================================================

    pair_rows = []

    for model in MODELS:

        for pair in PAIR_ORDER:

            sub = values[
                (
                    values["model"]
                    == model
                )
                &
                (
                    values["scale_pair"]
                    == pair
                )
            ]

            x = sub[
                "retrieval_similarity"
            ].to_numpy(
                dtype=np.float64
            )

            if len(x) != EXPECTED_CASES:
                raise RuntimeError(
                    f"{model}/{pair}: "
                    f"{len(x)}/"
                    f"{EXPECTED_CASES}"
                )

            lo, hi = (
                bootstrap_mean_ci(
                    x,
                    n_boot=args.bootstrap,
                    seed=2026,
                )
            )

            pair_rows.append({
                "model":
                    model,

                "model_display":
                    MODEL_DISPLAY[
                        model
                    ],

                "scale_pair":
                    pair,

                "n_cases":
                    len(x),

                "mean_retrieval_similarity":
                    float(
                        np.mean(x)
                    ),

                "std_retrieval_similarity":
                    float(
                        np.std(
                            x,
                            ddof=1,
                        )
                    ),

                "median_retrieval_similarity":
                    float(
                        np.median(x)
                    ),

                "q1":
                    float(
                        np.quantile(
                            x,
                            0.25,
                        )
                    ),

                "q3":
                    float(
                        np.quantile(
                            x,
                            0.75,
                        )
                    ),

                "ci95_low":
                    lo,

                "ci95_high":
                    hi,
            })

    summary_pair = pd.DataFrame(
        pair_rows
    )

    summary_pair.to_csv(
        OUT_SUMMARY_PAIR,
        index=False,
    )

    # ========================================================
    # SUMMARY — MODEL OVERALL
    # ========================================================
    #
    # First average the three scale pairs
    # WITHIN each case.
    #
    # This preserves each case as the unit
    # of analysis rather than treating
    # 213 pair observations as independent.
    # ========================================================

    case_model = (
        values.groupby(
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
            retrieval_similarity=(
                "retrieval_similarity",
                "mean",
            )
        )
    )

    model_rows = []

    for model in MODELS:

        sub = case_model[
            case_model["model"]
            == model
        ]

        x = sub[
            "retrieval_similarity"
        ].to_numpy(
            dtype=np.float64
        )

        if len(x) != EXPECTED_CASES:
            raise RuntimeError(
                f"{model}: "
                f"{len(x)}/"
                f"{EXPECTED_CASES} "
                "case-level overall scores"
            )

        lo, hi = bootstrap_mean_ci(
            x,
            n_boot=args.bootstrap,
            seed=2026,
        )

        model_rows.append({
            "model":
                model,

            "model_display":
                MODEL_DISPLAY[
                    model
                ],

            "n_cases":
                len(x),

            "mean_retrieval_similarity":
                float(
                    np.mean(x)
                ),

            "std_retrieval_similarity":
                float(
                    np.std(
                        x,
                        ddof=1,
                    )
                ),

            "median_retrieval_similarity":
                float(
                    np.median(x)
                ),

            "ci95_low":
                lo,

            "ci95_high":
                hi,
        })

    summary_model = pd.DataFrame(
        model_rows
    )

    summary_model.to_csv(
        OUT_SUMMARY_MODEL,
        index=False,
    )

    # ========================================================
    # CANCER × MODEL × PAIR
    # ========================================================

    cancer_summary = (
        values.groupby(
            [
                "cancer",
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
                "nunique",
            ),

            mean_retrieval_similarity=(
                "retrieval_similarity",
                "mean",
            ),

            std_retrieval_similarity=(
                "retrieval_similarity",
                "std",
            ),

            median_retrieval_similarity=(
                "retrieval_similarity",
                "median",
            ),
        )
    )

    cancer_summary.to_csv(
        OUT_CANCER,
        index=False,
    )

    # ========================================================
    # PROTOCOL
    # ========================================================

    protocol = {
        "analysis":
            "cross_scale_retrieval_similarity",

        "cohort":
            (
                "71 audited native-FOV "
                "same-center cases"
            ),

        "embedding_root":
            str(EMBED_ROOT),

        "models":
            MODELS,

        "n_models":
            len(MODELS),

        "n_cases":
            EXPECTED_CASES,

        "scale_pairs":
            [
                list(x)
                for x in SCALE_PAIRS
            ],

        "top_k":
            TOP_K,

        "n_query_per_direction":
            N_QUERY_PER_SLIDE,

        "forward_seed":
            FORWARD_SEED,

        "reverse_seed":
            REVERSE_SEED,

        "feature_normalization":
            "row-wise L2",

        "similarity":
            "cosine / normalized inner product",

        "gallery":
            (
                "all patches from the "
                "other scale within the "
                "same slide"
            ),

        "directional_score":
            (
                "mean across sampled queries "
                "of mean Top-5 cosine "
                "similarity"
            ),

        "bidirectional_score":
            (
                "(A_to_B + B_to_A) / 2"
            ),

        "implementation":
            (
                "exact FP32 GPU Top-K with "
                "target chunking; "
                "no approximate nearest-neighbor "
                "search"
            ),

        "row_correspondence_required":
            False,

        "original_metric":
            (
                "faithful to submitted "
                "Figure 4B"
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

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 96
    )

    print(
        "RETRIEVAL SIMILARITY — BY SCALE PAIR"
    )

    print(
        "=" * 96
    )

    print(
        summary_pair[
            [
                "model_display",
                "scale_pair",
                "n_cases",
                "mean_retrieval_similarity",
                "median_retrieval_similarity",
                "ci95_low",
                "ci95_high",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 96
    )

    print(
        "RETRIEVAL SIMILARITY — OVERALL"
    )

    print(
        "=" * 96
    )

    print(
        summary_model[
            [
                "model_display",
                "n_cases",
                "mean_retrieval_similarity",
                "median_retrieval_similarity",
                "ci95_low",
                "ci95_high",
            ]
        ]
        .sort_values(
            "mean_retrieval_similarity",
            ascending=False,
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        OUT_VALUES
    )

    print(
        OUT_SUMMARY_PAIR
    )

    print(
        OUT_SUMMARY_MODEL
    )

    print(
        OUT_CANCER
    )

    print(
        OUT_PROTOCOL
    )


if __name__ == "__main__":
    main()
