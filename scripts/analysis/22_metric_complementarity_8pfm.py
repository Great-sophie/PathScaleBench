#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer analysis:
Complementarity among five PathScaleBench dimensions
=====================================================

Analysis unit:
    pathology foundation model (n = 8)

Dimensions:
    1. Global alignment (CKA)
    2. Relative retention
    3. Cross-scale retrieval similarity
    4. Biological identity Recall@1
    5. Neighborhood preservation (NPS)

Questions
---------
A. Are the five dimensions redundant?

   Pairwise model-level Spearman correlations.
   For each of 10 metric pairs:
       - observed Spearman rho
       - exact two-sided permutation p-value
         using all 8! = 40,320 permutations
       - Holm correction across the 10 pairwise tests

B. Do the five dimensions give essentially the same PFM ranking?

   Kendall's coefficient of concordance W across
   five metric-specific rankings of eight PFMs.

   Significance is evaluated by Monte-Carlo permutation
   of model labels independently within metrics.

Interpretation
--------------
High W / uniformly high pairwise rho:
    dimensions largely redundant

Low-to-moderate W / heterogeneous rho:
    dimensions provide complementary model rankings

Important:
    complementarity does NOT imply exhaustiveness.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, spearmanr


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

FIG_ROOT = (
    NATIVE_ROOT
    / "reviewer_analysis"
    / "figures"
)

FIG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT = (
    RESULT_ROOT
    / "18_figure6_comprehensive_benchmark_raw.csv"
)

PREFIX = "22_metric_complementarity_8pfm"

OUT_RAW = (
    RESULT_ROOT
    / f"{PREFIX}_raw_matrix.csv"
)

OUT_RANKS = (
    RESULT_ROOT
    / f"{PREFIX}_model_ranks.csv"
)

OUT_RHO = (
    RESULT_ROOT
    / f"{PREFIX}_spearman_matrix.csv"
)

OUT_P = (
    RESULT_ROOT
    / f"{PREFIX}_exact_p_matrix.csv"
)

OUT_PHOLM = (
    RESULT_ROOT
    / f"{PREFIX}_holm_p_matrix.csv"
)

OUT_PAIRWISE = (
    RESULT_ROOT
    / f"{PREFIX}_pairwise_statistics.csv"
)

OUT_KENDALL = (
    RESULT_ROOT
    / f"{PREFIX}_kendall_w.csv"
)

OUT_PROTOCOL = (
    RESULT_ROOT
    / f"{PREFIX}_protocol.json"
)

OUT_PNG = (
    FIG_ROOT
    / "FigureS_Metric_Complementarity_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "FigureS_Metric_Complementarity_8PFM.pdf"
)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_MODELS = 8


# We use aliases because script 18 may have used either the
# compact Figure-6 column names or longer analysis names.
METRIC_ALIASES = {
    "Global alignment (CKA)": [
        "CKA",
        "cka",
        "global_alignment",
        "mean_cka",
    ],

    "Relative retention": [
        "Retention",
        "retention",
        "relative_retention",
    ],

    "Cross-scale retrieval": [
        "Retrieval",
        "retrieval",
        "cross_scale_retrieval",
        "retrieval_similarity",
    ],

    "Biological identity Recall@1": [
        "Recall@1",
        "recall@1",
        "Recall1",
        "recall1",
        "biological_identity_recall1",
        "biological_identity",
    ],

    "Neighborhood preservation": [
        "NPS",
        "nps",
        "neighborhood_preservation",
        "neighborhood_preservation_score",
    ],
}

SHORT_LABEL = {
    "Global alignment (CKA)": "CKA",
    "Relative retention": "Retention",
    "Cross-scale retrieval": "Retrieval",
    "Biological identity Recall@1": "Recall@1",
    "Neighborhood preservation": "NPS",
}


# ============================================================
# ARGS
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--kendall-permutations",
        type=int,
        default=100000,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


# ============================================================
# HELPERS
# ============================================================

def holm_adjust(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(pvalues)

    order = np.argsort(
        pvalues
    )

    sorted_p = pvalues[
        order
    ]

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, p in enumerate(
        sorted_p
    ):

        value = (
            (m - i)
            * p
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted


def resolve_column(
    df,
    aliases,
    metric_name,
):

    found = [
        col
        for col in aliases
        if col in df.columns
    ]

    if len(found) == 0:

        raise RuntimeError(
            f"\nCould not find column for:\n"
            f"  {metric_name}\n\n"
            f"Tried aliases:\n"
            f"  {aliases}\n\n"
            f"Available columns:\n"
            f"  {list(df.columns)}"
        )

    return found[0]


def exact_spearman_permutation(
    x,
    y,
):

    """
    Exact two-sided permutation p-value for n=8.

    Fix x and enumerate all 8! permutations of y.

    p = number(|rho_perm| >= |rho_obs|) / 8!

    Because the complete permutation space is enumerated,
    no +1 Monte-Carlo correction is required.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(x)

    if n != 8:
        raise RuntimeError(
            f"Exact permutation routine expects n=8, got {n}"
        )

    rho_obs = float(
        spearmanr(
            x,
            y,
        ).statistic
    )

    extreme = 0
    total = 0

    indices = range(n)

    for perm in itertools.permutations(
        indices
    ):

        yp = y[
            perm,
        ]

        rho_perm = spearmanr(
            x,
            yp,
        ).statistic

        total += 1

        if (
            abs(rho_perm)
            >=
            abs(rho_obs)
            - 1e-12
        ):
            extreme += 1

    p_exact = (
        extreme
        / total
    )

    return (
        rho_obs,
        float(p_exact),
        int(extreme),
        int(total),
    )


def kendall_w(
    rank_matrix,
):

    """
    Kendall's coefficient of concordance W.

    rank_matrix:
        shape (n_objects, m_metrics)

    Rows = PFMs
    Columns = metric-specific rankings

    Rank 1 = best model.

    Includes tie correction, although current PFM rankings
    are expected to be largely untied.
    """

    R = np.asarray(
        rank_matrix,
        dtype=float,
    )

    n, m = R.shape

    rank_sums = np.sum(
        R,
        axis=1,
    )

    mean_rank_sum = (
        m
        * (n + 1)
        / 2.0
    )

    S = np.sum(
        (
            rank_sums
            - mean_rank_sum
        ) ** 2
    )

    # Tie correction:
    # T_j = sum(t^3 - t) for tied groups in metric j
    tie_term = 0.0

    for j in range(m):

        _, counts = np.unique(
            R[:, j],
            return_counts=True,
        )

        tie_term += np.sum(
            counts ** 3
            - counts
        )

    denominator = (
        m ** 2
        * (
            n ** 3
            - n
        )
        -
        m
        * tie_term
    )

    if denominator <= 0:
        raise RuntimeError(
            "Invalid Kendall W denominator."
        )

    W = (
        12.0
        * S
        / denominator
    )

    return float(W)


def kendall_w_permutation(
    rank_matrix,
    n_perm,
    seed,
):

    """
    Monte-Carlo null distribution.

    Model labels are independently shuffled within
    each metric ranking.

    This destroys cross-metric concordance while
    preserving each metric's marginal ranks.
    """

    R = np.asarray(
        rank_matrix,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    W_obs = kendall_w(
        R
    )

    n, m = R.shape

    extreme = 0

    for _ in range(
        n_perm
    ):

        permuted = np.empty_like(
            R
        )

        # First metric may be fixed because the null
        # is invariant to a common relabeling.
        permuted[:, 0] = R[:, 0]

        for j in range(
            1,
            m
        ):

            permuted[:, j] = R[
                rng.permutation(n),
                j,
            ]

        W_perm = kendall_w(
            permuted
        )

        if (
            W_perm
            >=
            W_obs
            - 1e-12
        ):
            extreme += 1

    p = (
        extreme
        + 1
    ) / (
        n_perm
        + 1
    )

    return (
        W_obs,
        float(p),
        int(extreme),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print(
        "=" * 110
    )

    print(
        "PATHSCALEBENCH METRIC COMPLEMENTARITY ANALYSIS"
    )

    print(
        "=" * 110
    )

    if not INPUT.exists():

        raise FileNotFoundError(
            INPUT
        )

    df = pd.read_csv(
        INPUT
    )

    print(
        "\nInput:"
    )

    print(
        INPUT
    )

    print(
        "\nColumns:"
    )

    print(
        list(df.columns)
    )


    # ========================================================
    # IDENTIFY MODEL COLUMN
    # ========================================================

    if "model_display" in df.columns:

        model_col = "model_display"

    elif "model" in df.columns:

        model_col = "model"

    else:

        raise RuntimeError(
            "No model/model_display column."
        )


    if len(df) != EXPECTED_MODELS:

        raise RuntimeError(
            f"Expected 8 rows in Figure 6 raw table, "
            f"found {len(df)}"
        )


    if df[
        model_col
    ].nunique() != EXPECTED_MODELS:

        raise RuntimeError(
            "Expected eight unique PFMs."
        )


    # ========================================================
    # RESOLVE FIVE METRICS
    # ========================================================

    resolved = {}

    for metric_name, aliases in (
        METRIC_ALIASES.items()
    ):

        resolved[
            metric_name
        ] = resolve_column(
            df,
            aliases,
            metric_name,
        )


    print(
        "\nResolved metric columns:"
    )

    for metric, col in (
        resolved.items()
    ):

        print(
            f"  {metric:35s} <- {col}"
        )


    # ========================================================
    # BUILD MODEL × METRIC RAW MATRIX
    # ========================================================

    raw = pd.DataFrame({
        "model":
            df[
                model_col
            ].astype(str),
    })

    for metric_name, col in (
        resolved.items()
    ):

        raw[
            SHORT_LABEL[
                metric_name
            ]
        ] = pd.to_numeric(
            df[col],
            errors="raise",
        )


    if raw.isna().any().any():

        raise RuntimeError(
            "NaN detected in raw benchmark matrix."
        )


    raw.to_csv(
        OUT_RAW,
        index=False,
    )


    metric_cols = [
        "CKA",
        "Retention",
        "Retrieval",
        "Recall@1",
        "NPS",
    ]


    # ========================================================
    # MODEL RANKS
    #
    # All five metrics are oriented so higher = better.
    # ========================================================

    ranks = pd.DataFrame({
        "model":
            raw["model"],
    })

    for metric in metric_cols:

        ranks[
            metric
        ] = rankdata(
            -raw[
                metric
            ].to_numpy(
                dtype=float
            ),
            method="average",
        )


    ranks[
        "mean_rank"
    ] = ranks[
        metric_cols
    ].mean(
        axis=1
    )

    ranks[
        "rank_sd"
    ] = ranks[
        metric_cols
    ].std(
        axis=1,
        ddof=0,
    )

    ranks[
        "descriptive_overall_rank"
    ] = (
        ranks[
            "mean_rank"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )


    ranks.to_csv(
        OUT_RANKS,
        index=False,
    )


    # ========================================================
    # PAIRWISE EXACT SPEARMAN
    # ========================================================

    rho_matrix = pd.DataFrame(
        np.eye(
            len(metric_cols)
        ),
        index=metric_cols,
        columns=metric_cols,
    )

    p_matrix = pd.DataFrame(
        np.ones(
            (
                len(metric_cols),
                len(metric_cols),
            )
        ),
        index=metric_cols,
        columns=metric_cols,
    )

    pair_rows = []


    print(
        "\nRunning exact 8! Spearman permutations..."
    )

    for metric_a, metric_b in itertools.combinations(
        metric_cols,
        2,
    ):

        print(
            f"  {metric_a} vs {metric_b}",
            flush=True,
        )

        (
            rho,
            p_exact,
            extreme,
            total,
        ) = exact_spearman_permutation(
            raw[
                metric_a
            ].to_numpy(
                dtype=float
            ),
            raw[
                metric_b
            ].to_numpy(
                dtype=float
            ),
        )

        rho_matrix.loc[
            metric_a,
            metric_b,
        ] = rho

        rho_matrix.loc[
            metric_b,
            metric_a,
        ] = rho

        p_matrix.loc[
            metric_a,
            metric_b,
        ] = p_exact

        p_matrix.loc[
            metric_b,
            metric_a,
        ] = p_exact

        pair_rows.append({
            "metric_a":
                metric_a,

            "metric_b":
                metric_b,

            "spearman_rho":
                rho,

            "exact_two_sided_p":
                p_exact,

            "extreme_permutations":
                extreme,

            "total_permutations":
                total,
        })


    pairwise = pd.DataFrame(
        pair_rows
    )

    pairwise[
        "holm_p"
    ] = holm_adjust(
        pairwise[
            "exact_two_sided_p"
        ]
    )

    pairwise[
        "significant_holm_0_05"
    ] = (
        pairwise[
            "holm_p"
        ]
        < 0.05
    )


    # ========================================================
    # HOLM MATRIX
    # ========================================================

    holm_matrix = pd.DataFrame(
        np.ones(
            (
                len(metric_cols),
                len(metric_cols),
            )
        ),
        index=metric_cols,
        columns=metric_cols,
    )

    for _, row in (
        pairwise.iterrows()
    ):

        a = row[
            "metric_a"
        ]

        b = row[
            "metric_b"
        ]

        p = row[
            "holm_p"
        ]

        holm_matrix.loc[
            a,
            b,
        ] = p

        holm_matrix.loc[
            b,
            a,
        ] = p


    rho_matrix.to_csv(
        OUT_RHO
    )

    p_matrix.to_csv(
        OUT_P
    )

    holm_matrix.to_csv(
        OUT_PHOLM
    )

    pairwise.to_csv(
        OUT_PAIRWISE,
        index=False,
    )


    # ========================================================
    # KENDALL'S W
    # ========================================================

    rank_matrix = ranks[
        metric_cols
    ].to_numpy(
        dtype=float
    )

    print(
        "\nRunning Kendall W permutation test..."
    )

    (
        W,
        W_p,
        W_extreme,
    ) = kendall_w_permutation(
        rank_matrix,
        args.kendall_permutations,
        args.seed,
    )


    kendall_df = pd.DataFrame([
        {
            "n_models":
                EXPECTED_MODELS,

            "n_metrics":
                len(
                    metric_cols
                ),

            "kendall_W":
                W,

            "permutation_p":
                W_p,

            "n_permutations":
                args.kendall_permutations,

            "extreme_permutations":
                W_extreme,
        }
    ])

    kendall_df.to_csv(
        OUT_KENDALL,
        index=False,
    )


    # ========================================================
    # FIGURE — 5×5 SPEARMAN MATRIX
    # ========================================================

    plt.rcParams.update({
        "font.family":
            "DejaVu Sans",

        "font.size":
            10,

        "axes.titlesize":
            12,

        "xtick.labelsize":
            9,

        "ytick.labelsize":
            9,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    })


    fig, ax = plt.subplots(
        figsize=(
            7.4,
            6.5,
        )
    )


    arr = rho_matrix.to_numpy(
        dtype=float
    )

    im = ax.imshow(
        arr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        interpolation="nearest",
        aspect="equal",
    )


    # --------------------------------------------------------
    # Cell annotation:
    # rho + significance based on Holm-adjusted p
    # --------------------------------------------------------

    for i in range(
        len(metric_cols)
    ):

        for j in range(
            len(metric_cols)
        ):

            rho = arr[
                i,
                j
            ]

            if i == j:

                text = "1.00"

            else:

                p_holm = holm_matrix.iloc[
                    i,
                    j
                ]

                if p_holm < 0.001:
                    stars = "***"

                elif p_holm < 0.01:
                    stars = "**"

                elif p_holm < 0.05:
                    stars = "*"

                else:
                    stars = ""

                text = (
                    f"{rho:.2f}"
                    f"{stars}"
                )


            color = (
                "white"
                if abs(rho) >= 0.55
                else "black"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight=(
                    "bold"
                    if i != j
                    else "normal"
                ),
                color=color,
            )


    labels = [
        "CKA",
        "Retention",
        "Retrieval",
        "Recall@1",
        "NPS",
    ]


    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticklabels(
        labels
    )


    ax.set_title(
        "Complementarity among PathScaleBench dimensions",
        fontweight="bold",
        pad=12,
    )


    # Cell boundaries
    ax.set_xticks(
        np.arange(
            -0.5,
            len(labels),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(labels),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=1.0,
        color="white",
        alpha=0.8,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        pad=0.04,
    )

    cbar.set_label(
        "Model-level Spearman ρ"
    )


    fig.text(
        0.08,
        0.025,
        (
            f"Kendall's W = {W:.3f}; "
            f"permutation p = {W_p:.4g}. "
            "Pairwise p-values are exact over all 8! model permutations "
            "and Holm-adjusted across 10 metric pairs. "
            "* p<0.05, ** p<0.01, *** p<0.001 after Holm correction."
        ),
        ha="left",
        va="bottom",
        fontsize=8,
    )


    fig.subplots_adjust(
        left=0.16,
        right=0.91,
        top=0.90,
        bottom=0.18,
    )


    fig.savefig(
        OUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # ========================================================
    # PROTOCOL
    # ========================================================

    protocol = {
        "analysis":
            (
                "Complementarity among five "
                "PathScaleBench benchmark dimensions"
            ),

        "analysis_unit":
            "PFM",

        "n_models":
            EXPECTED_MODELS,

        "metrics":
            metric_cols,

        "pairwise_association":
            "Spearman rank correlation",

        "pairwise_inference":
            (
                "Exact two-sided permutation over "
                "all 8! = 40,320 permutations"
            ),

        "pairwise_multiplicity":
            (
                "Holm correction across all 10 "
                "metric-pair associations"
            ),

        "ranking_concordance":
            "Kendall coefficient of concordance W",

        "kendall_inference":
            (
                "Monte-Carlo independent model-label "
                "permutation within metric rankings"
            ),

        "kendall_permutations":
            args.kendall_permutations,

        "scope":
            (
                "Tests redundancy/complementarity among "
                "selected benchmark dimensions; does not "
                "test or imply exhaustiveness of PFM "
                "representation information."
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
        + "=" * 110
    )

    print(
        "RAW MODEL × METRIC MATRIX"
    )

    print(
        "=" * 110
    )

    print(
        raw
        .round(4)
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "MODEL RANKS — 1 = HIGHEST METRIC VALUE"
    )

    print(
        "=" * 110
    )

    print(
        ranks
        .sort_values(
            "mean_rank"
        )
        .round(3)
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "PAIRWISE MODEL-LEVEL SPEARMAN ASSOCIATIONS"
    )

    print(
        "=" * 110
    )

    print(
        pairwise[
            [
                "metric_a",
                "metric_b",
                "spearman_rho",
                "exact_two_sided_p",
                "holm_p",
                "significant_holm_0_05",
            ]
        ]
        .sort_values(
            "spearman_rho",
            ascending=False,
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
        "KENDALL RANK CONCORDANCE"
    )

    print(
        "=" * 110
    )

    print(
        f"Kendall W          = {W:.6f}"
    )

    print(
        f"Permutation p      = {W_p:.6g}"
    )

    print(
        f"Permutations       = "
        f"{args.kendall_permutations:,}"
    )


    print(
        "\nOutputs:"
    )

    for path in [
        OUT_RAW,
        OUT_RANKS,
        OUT_RHO,
        OUT_P,
        OUT_PHOLM,
        OUT_PAIRWISE,
        OUT_KENDALL,
        OUT_PROTOCOL,
        OUT_PNG,
        OUT_PDF,
    ]:

        print(path)


if __name__ == "__main__":
    main()
