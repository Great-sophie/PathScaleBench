#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Revised Figure 8 analysis
=========================

Magnification-shift downstream robustness across 8 PFMs.

Cohort
------
71 audited native-FOV same-center cases:
BLCA 10
BRCA 10
COAD 7
HNSC 8
KIRC 9
LUAD 9
STAD 9
UCEC 9

Models
------
phikon
uni
virchow
virchow2
uni2
midnight
gigapath
gigapath_flash

Scales
------
40x
10x
2p5x

Downstream protocol
-------------------
For each model and scale:

1. Mean-pool patch embeddings within each slide.
2. L2-normalize the slide vector.
3. Train a class-balanced multinomial linear probe at a SOURCE scale.
4. Evaluate the SAME held-out cases at all three TARGET scales.

Revision improvement:
---------------------
For each repeat, ONE shared StratifiedGroupKFold partition is generated
and reused across:

    all 8 PFMs
    × all 3 source scales
    × all 3 target scales

This removes unnecessary split variation between models/scales.

Primary downstream metric:
--------------------------
Pooled OOF balanced accuracy.

Magnification Generalization Gap:
---------------------------------
MGG(source -> target)
    = BA(source -> source)
      - BA(source -> target)

Higher MGG = larger magnification-shift penalty.

Probabilistic slide-level robustness:
-------------------------------------
shift_loss
    = NLL(source -> target)
      - NLL(source -> source)

shift_robustness
    = -mean(shift_loss)

Higher shift_robustness = more robust.

Representation-level association analyses
-----------------------------------------

OVERALL:
case × model (71 × 8 = 568)

Predictors:
- mean CKA
- relative retention
- mean retrieval similarity
- mean NPS

Outcome:
- mean shift robustness across six directed magnification shifts

PAIR-MATCHED:
case × model × undirected scale pair
71 × 8 × 3 = 1704

Predictors:
- pair-specific CKA
- pair-specific retrieval similarity
- pair-specific NPS

Outcome:
- robustness averaged over the two directions belonging to the
  same undirected magnification pair.

Association inference:
----------------------
To remove between-model baseline differences:

Overall analysis:
    percentile-rank predictor and outcome WITHIN MODEL.

Pair-matched analysis:
    percentile-rank predictor and outcome WITHIN MODEL × SCALE PAIR.

Then:
- pooled Spearman rho
- slide-cluster bootstrap 95% CI
- structured permutation preserving the full model/pair profile of a slide
- Holm correction within each planned association family
- conservative Holm correction across all seven associations
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


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


CKA_FILE = (
    RESULT_ROOT
    / "01b_cross_scale_cka_8pfm_fullpatch_per_case.csv"
)

RETENTION_FILE = (
    RESULT_ROOT
    / "02_relative_information_retention_8pfm_values.csv"
)

RETRIEVAL_FILE = (
    RESULT_ROOT
    / "07_cross_scale_retrieval_similarity_8pfm_values.csv"
)

NPS_CASE_FILE = (
    RESULT_ROOT
    / "16_neighborhood_preservation_score_8pfm_per_case.csv"
)

NPS_PAIR_FILE = (
    RESULT_ROOT
    / "16_neighborhood_preservation_score_8pfm_per_case_pair.csv"
)


PREFIX = "20_magnification_shift_robustness_8pfm"

OUT_SLIDE_META = RESULT_ROOT / f"{PREFIX}_slide_embedding_metadata.csv"
OUT_FOLDS = RESULT_ROOT / f"{PREFIX}_shared_fold_assignments.csv"
OUT_FOLD_PERF = RESULT_ROOT / f"{PREFIX}_fold_performance.csv"
OUT_PRED = RESULT_ROOT / f"{PREFIX}_out_of_fold_predictions.csv"
OUT_REPEAT = RESULT_ROOT / f"{PREFIX}_repeat_performance.csv"
OUT_GAPS = RESULT_ROOT / f"{PREFIX}_generalization_gaps.csv"
OUT_DIRECTION = RESULT_ROOT / f"{PREFIX}_direction_level_generalization_gaps.csv"
OUT_MATRIX = RESULT_ROOT / f"{PREFIX}_balanced_accuracy_matrix.csv"

OUT_SHIFT_DIRECTION = RESULT_ROOT / f"{PREFIX}_slide_direction_shift_loss.csv"
OUT_ROBUST = RESULT_ROOT / f"{PREFIX}_slide_shift_robustness.csv"
OUT_PAIR_ROBUST = RESULT_ROOT / f"{PREFIX}_slide_pair_shift_robustness.csv"

OUT_METRICS = RESULT_ROOT / f"{PREFIX}_overall_metrics_and_robustness.csv"
OUT_PAIR_METRICS = RESULT_ROOT / f"{PREFIX}_pair_matched_metrics_and_robustness.csv"

OUT_ASSOC = RESULT_ROOT / f"{PREFIX}_overall_associations.csv"
OUT_PAIR_ASSOC = RESULT_ROOT / f"{PREFIX}_pair_matched_associations.csv"

OUT_PROTOCOL = RESULT_ROOT / f"{PREFIX}_protocol.json"


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

EXPECTED_DIMS = {
    "phikon": 768,
    "uni": 1024,
    "virchow": 2560,
    "virchow2": 2560,
    "uni2": 1536,
    "midnight": 3072,
    "gigapath": 1536,
    "gigapath_flash": 384,
}

SCALES = [
    "40x",
    "10x",
    "2p5x",
]

SCALE_LABEL = {
    "40x": "40×",
    "10x": "10×",
    "2p5x": "2.5×",
}

EXPECTED_CASES = 71


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--n-folds",
        type=int,
        default=5,
    )

    p.add_argument(
        "--n-repeats",
        type=int,
        default=10,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
    )

    return p.parse_args()


# ============================================================
# GENERAL HELPERS
# ============================================================

def patient_id_from_case(case_id: str) -> str:

    parts = str(case_id).split("-")

    if len(parts) < 3:
        raise RuntimeError(
            f"Cannot derive patient ID: {case_id}"
        )

    return "-".join(
        parts[:3]
    )


def l2_normalize(v):

    v = np.asarray(
        v,
        dtype=np.float32,
    )

    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm


def canonical_pair(a, b):

    key = frozenset([
        str(a),
        str(b),
    ])

    mapping = {
        frozenset(["40x", "10x"]):
            "40x__10x",

        frozenset(["10x", "2p5x"]):
            "10x__2p5x",

        frozenset(["40x", "2p5x"]):
            "40x__2p5x",
    }

    if key not in mapping:
        raise RuntimeError(
            f"Unknown scale pair: {a}, {b}"
        )

    return mapping[key]


def pair_display(pair_key):

    mapping = {
        "40x__10x":
            "40× ↔ 10×",

        "10x__2p5x":
            "10× ↔ 2.5×",

        "40x__2p5x":
            "40× ↔ 2.5×",
    }

    return mapping[pair_key]


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


# ============================================================
# COHORT
# ============================================================

def build_cohort():

    retention = pd.read_csv(
        RETENTION_FILE
    )

    required = {
        "cancer",
        "case_id",
        "model",
        "relative_retention",
    }

    missing = required - set(
        retention.columns
    )

    if missing:
        raise RuntimeError(
            f"Retention missing columns: {missing}"
        )

    cohort = (
        retention[
            [
                "cancer",
                "case_id",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    cohort[
        "cancer"
    ] = pd.Categorical(
        cohort["cancer"],
        categories=CANCERS,
        ordered=True,
    )

    cohort = (
        cohort
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

    cohort[
        "cancer"
    ] = cohort[
        "cancer"
    ].astype(str)

    cohort[
        "patient_id"
    ] = cohort[
        "case_id"
    ].map(
        patient_id_from_case
    )

    if len(cohort) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected 71 cases, got {len(cohort)}"
        )

    for cancer, expected in EXPECTED_COUNTS.items():

        actual = int(
            np.sum(
                cohort["cancer"]
                == cancer
            )
        )

        if actual != expected:
            raise RuntimeError(
                f"{cancer}: {actual}/{expected}"
            )

    if (
        cohort["patient_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Multiple slides from the same patient "
            "detected in the 71-case cohort."
        )

    return cohort


# ============================================================
# SLIDE EMBEDDINGS
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


def build_slide_matrices(
    cohort,
):

    matrices = {}

    metadata_rows = []

    print(
        "\nBuilding scale-specific "
        "slide embeddings..."
    )

    for model_i, model in enumerate(
        MODELS,
        start=1,
    ):

        print(
            f"\n[MODEL {model_i}/8] "
            f"{MODEL_DISPLAY[model]}"
        )

        for scale in SCALES:

            vectors = []

            print(
                f"  {scale}: ",
                end="",
                flush=True,
            )

            for _, row in cohort.iterrows():

                cancer = row["cancer"]
                case_id = row["case_id"]

                path = embedding_path(
                    cancer,
                    case_id,
                    model,
                    scale,
                )

                if not path.exists():
                    raise FileNotFoundError(
                        path
                    )

                X = np.load(
                    path,
                    mmap_mode="r",
                )

                if X.ndim != 2:
                    raise RuntimeError(
                        f"{path}: shape={X.shape}"
                    )

                if (
                    X.shape[1]
                    != EXPECTED_DIMS[model]
                ):
                    raise RuntimeError(
                        f"{model}: dim "
                        f"{X.shape[1]} != "
                        f"{EXPECTED_DIMS[model]}"
                    )

                # Mean pool patches, then L2 normalize.
                v = np.mean(
                    X,
                    axis=0,
                    dtype=np.float64,
                )

                v = l2_normalize(
                    v
                )

                vectors.append(
                    v
                )

                metadata_rows.append({
                    "cancer":
                        cancer,

                    "case_id":
                        case_id,

                    "model":
                        model,

                    "model_display":
                        MODEL_DISPLAY[model],

                    "scale":
                        scale,

                    "n_patches":
                        int(X.shape[0]),

                    "embedding_dim":
                        int(X.shape[1]),
                })

                del X

            matrices[
                (model, scale)
            ] = np.vstack(
                vectors
            ).astype(
                np.float32
            )

            print(
                matrices[
                    (model, scale)
                ].shape
            )

    metadata = pd.DataFrame(
        metadata_rows
    )

    if len(metadata) != (
        EXPECTED_CASES
        * len(MODELS)
        * len(SCALES)
    ):
        raise RuntimeError(
            "Bad slide metadata row count."
        )

    # Same row count across scales within case/model.
    audit = (
        metadata.pivot_table(
            index=[
                "case_id",
                "model",
            ],
            columns="scale",
            values="n_patches",
            aggfunc="first",
        )
    )

    if not (
        (
            audit["40x"]
            == audit["10x"]
        )
        &
        (
            audit["40x"]
            == audit["2p5x"]
        )
    ).all():
        raise RuntimeError(
            "Patch row count mismatch "
            "across scales."
        )

    metadata.to_csv(
        OUT_SLIDE_META,
        index=False,
    )

    return matrices


# ============================================================
# SHARED REPEATED CV SPLITS
# ============================================================

def make_shared_splits(
    cohort,
    n_folds,
    n_repeats,
    seed,
):

    y = cohort[
        "cancer"
    ].to_numpy()

    groups = cohort[
        "patient_id"
    ].to_numpy()

    dummy = np.zeros(
        (
            len(cohort),
            1,
        )
    )

    all_splits = {}

    assignment_rows = []

    for repeat in range(
        n_repeats
    ):

        repeat_seed = (
            seed
            + repeat * 101
        )

        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=repeat_seed,
        )

        split_list = list(
            splitter.split(
                dummy,
                y,
                groups,
            )
        )

        # audit each case exactly once in test
        seen = np.zeros(
            len(cohort),
            dtype=int,
        )

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(
            split_list,
            start=1,
        ):

            seen[
                test_idx
            ] += 1

            for idx in test_idx:

                assignment_rows.append({
                    "repeat":
                        repeat,

                    "fold":
                        fold,

                    "case_index":
                        int(idx),

                    "case_id":
                        cohort.iloc[idx][
                            "case_id"
                        ],

                    "patient_id":
                        cohort.iloc[idx][
                            "patient_id"
                        ],

                    "cancer":
                        cohort.iloc[idx][
                            "cancer"
                        ],

                    "repeat_seed":
                        repeat_seed,
                })

        if not np.all(
            seen == 1
        ):
            raise RuntimeError(
                f"Repeat {repeat}: "
                "invalid OOF partition."
            )

        all_splits[
            repeat
        ] = split_list

    assignments = pd.DataFrame(
        assignment_rows
    )

    assignments.to_csv(
        OUT_FOLDS,
        index=False,
    )

    return all_splits


# ============================================================
# LINEAR PROBE
# ============================================================

def make_classifier(seed):

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=5000,
                tol=1e-4,
                random_state=seed,
            ),
        ),
    ])


# ============================================================
# CROSS-SCALE CV
# ============================================================

def run_cross_scale_cv(
    cohort,
    matrices,
    shared_splits,
    n_repeats,
    seed,
):

    y = cohort[
        "cancer"
    ].to_numpy()

    fold_rows = []
    pred_rows = []

    total_fit = (
        len(MODELS)
        * len(SCALES)
        * n_repeats
        * len(
            shared_splits[0]
        )
    )

    fit_counter = 0

    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
    )

    for model in MODELS:

        print(
            f"\nCV: {MODEL_DISPLAY[model]}"
        )

        for source in SCALES:

            for repeat in range(
                n_repeats
            ):

                for fold_zero, (
                    train_idx,
                    test_idx,
                ) in enumerate(
                    shared_splits[
                        repeat
                    ]
                ):

                    fold = (
                        fold_zero
                        + 1
                    )

                    fit_counter += 1

                    clf_seed = (
                        seed
                        + repeat * 101
                        + fold
                    )

                    clf = make_classifier(
                        clf_seed
                    )

                    clf.fit(
                        matrices[
                            (
                                model,
                                source,
                            )
                        ][train_idx],
                        y[
                            train_idx
                        ],
                    )

                    class_order = list(
                        clf.named_steps[
                            "classifier"
                        ].classes_
                    )

                    class_to_col = {
                        c: i
                        for i, c in enumerate(
                            class_order
                        )
                    }

                    for target in SCALES:

                        x_test = matrices[
                            (
                                model,
                                target,
                            )
                        ][test_idx]

                        true = y[
                            test_idx
                        ]

                        pred = clf.predict(
                            x_test
                        )

                        proba = clf.predict_proba(
                            x_test
                        )

                        p_true = np.array([
                            proba[
                                i,
                                class_to_col[
                                    true[i]
                                ],
                            ]
                            for i in range(
                                len(true)
                            )
                        ])

                        p_true = np.clip(
                            p_true,
                            1e-12,
                            1.0,
                        )

                        nll = -np.log(
                            p_true
                        )

                        bal = (
                            balanced_accuracy_score(
                                true,
                                pred,
                            )
                        )

                        macro = f1_score(
                            true,
                            pred,
                            labels=CANCERS,
                            average="macro",
                            zero_division=0,
                        )

                        fold_rows.append({
                            "model":
                                model,

                            "model_display":
                                MODEL_DISPLAY[
                                    model
                                ],

                            "source_scale":
                                source,

                            "target_scale":
                                target,

                            "repeat":
                                repeat,

                            "fold":
                                fold,

                            "n_train":
                                len(
                                    train_idx
                                ),

                            "n_test":
                                len(
                                    test_idx
                                ),

                            "balanced_accuracy":
                                bal,

                            "macro_f1":
                                macro,

                            "log_loss":
                                float(
                                    np.mean(
                                        nll
                                    )
                                ),
                        })

                        for pos, idx in enumerate(
                            test_idx
                        ):

                            pred_rows.append({
                                "case_id":
                                    cohort.iloc[idx][
                                        "case_id"
                                    ],

                                "patient_id":
                                    cohort.iloc[idx][
                                        "patient_id"
                                    ],

                                "cancer":
                                    cohort.iloc[idx][
                                        "cancer"
                                    ],

                                "model":
                                    model,

                                "model_display":
                                    MODEL_DISPLAY[
                                        model
                                    ],

                                "source_scale":
                                    source,

                                "target_scale":
                                    target,

                                "repeat":
                                    repeat,

                                "fold":
                                    fold,

                                "true_label":
                                    true[pos],

                                "predicted_label":
                                    pred[pos],

                                "correct":
                                    int(
                                        true[pos]
                                        == pred[pos]
                                    ),

                                "true_class_probability":
                                    float(
                                        p_true[pos]
                                    ),

                                "negative_log_likelihood":
                                    float(
                                        nll[pos]
                                    ),
                            })

                    if (
                        fit_counter % 50
                        == 0
                    ):

                        print(
                            f"  fits "
                            f"{fit_counter}/"
                            f"{total_fit}",
                            flush=True,
                        )

    return (
        pd.DataFrame(
            fold_rows
        ),
        pd.DataFrame(
            pred_rows
        ),
    )


# ============================================================
# REPEAT-LEVEL POOLED OOF PERFORMANCE
# ============================================================

def repeat_performance(
    predictions,
):

    rows = []

    group_cols = [
        "model",
        "model_display",
        "source_scale",
        "target_scale",
        "repeat",
    ]

    for key, g in predictions.groupby(
        group_cols,
        sort=False,
    ):

        (
            model,
            model_display,
            source,
            target,
            repeat,
        ) = key

        if len(g) != EXPECTED_CASES:
            raise RuntimeError(
                f"OOF rows !=71: "
                f"{key}: {len(g)}"
            )

        bal = balanced_accuracy_score(
            g["true_label"],
            g["predicted_label"],
        )

        macro = f1_score(
            g["true_label"],
            g["predicted_label"],
            labels=CANCERS,
            average="macro",
            zero_division=0,
        )

        rows.append({
            "model":
                model,

            "model_display":
                model_display,

            "source_scale":
                source,

            "target_scale":
                target,

            "repeat":
                repeat,

            "n_cases":
                len(g),

            "balanced_accuracy":
                float(bal),

            "macro_f1":
                float(macro),

            "log_loss":
                float(
                    g[
                        "negative_log_likelihood"
                    ].mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# GENERALIZATION GAPS
# ============================================================

def generalization_gaps(
    performance,
):

    within = (
        performance[
            performance[
                "source_scale"
            ]
            ==
            performance[
                "target_scale"
            ]
        ][
            [
                "model",
                "model_display",
                "source_scale",
                "repeat",
                "balanced_accuracy",
                "macro_f1",
                "log_loss",
            ]
        ]
        .rename(
            columns={
                "balanced_accuracy":
                    "within_balanced_accuracy",

                "macro_f1":
                    "within_macro_f1",

                "log_loss":
                    "within_log_loss",
            }
        )
    )

    cross = performance[
        performance[
            "source_scale"
        ]
        !=
        performance[
            "target_scale"
        ]
    ].copy()

    gaps = cross.merge(
        within,
        on=[
            "model",
            "model_display",
            "source_scale",
            "repeat",
        ],
        how="left",
        validate="many_to_one",
    )

    gaps[
        "mgg_balanced_accuracy"
    ] = (
        gaps[
            "within_balanced_accuracy"
        ]
        -
        gaps[
            "balanced_accuracy"
        ]
    )

    gaps[
        "mgg_macro_f1"
    ] = (
        gaps[
            "within_macro_f1"
        ]
        -
        gaps[
            "macro_f1"
        ]
    )

    gaps[
        "shift_log_loss"
    ] = (
        gaps[
            "log_loss"
        ]
        -
        gaps[
            "within_log_loss"
        ]
    )

    gaps[
        "direction"
    ] = (
        gaps[
            "source_scale"
        ].map(
            SCALE_LABEL
        )
        +
        " → "
        +
        gaps[
            "target_scale"
        ].map(
            SCALE_LABEL
        )
    )

    gaps[
        "pair_key"
    ] = [
        canonical_pair(
            a,
            b,
        )
        for a, b in zip(
            gaps[
                "source_scale"
            ],
            gaps[
                "target_scale"
            ],
        )
    ]

    return gaps


# ============================================================
# SLIDE-LEVEL SHIFT ROBUSTNESS
# ============================================================

def slide_shift_robustness(
    predictions,
):

    within = (
        predictions[
            predictions[
                "source_scale"
            ]
            ==
            predictions[
                "target_scale"
            ]
        ][
            [
                "case_id",
                "cancer",
                "model",
                "model_display",
                "source_scale",
                "repeat",
                "negative_log_likelihood",
            ]
        ]
        .rename(
            columns={
                "negative_log_likelihood":
                    "within_nll"
            }
        )
    )

    cross = predictions[
        predictions[
            "source_scale"
        ]
        !=
        predictions[
            "target_scale"
        ]
    ].copy()

    cross = cross.merge(
        within,
        on=[
            "case_id",
            "cancer",
            "model",
            "model_display",
            "source_scale",
            "repeat",
        ],
        how="left",
        validate="many_to_one",
    )

    cross[
        "shift_loss"
    ] = (
        cross[
            "negative_log_likelihood"
        ]
        -
        cross[
            "within_nll"
        ]
    )

    cross[
        "shift_robustness_direction"
    ] = (
        -cross[
            "shift_loss"
        ]
    )

    cross[
        "pair_key"
    ] = [
        canonical_pair(
            a,
            b,
        )
        for a, b in zip(
            cross[
                "source_scale"
            ],
            cross[
                "target_scale"
            ],
        )
    ]

    cross[
        "pair_display"
    ] = cross[
        "pair_key"
    ].map(
        pair_display
    )

    # Overall: all six directions × all repeats
    overall = (
        cross.groupby(
            [
                "case_id",
                "cancer",
                "model",
                "model_display",
            ],
            as_index=False,
        )
        .agg(
            mean_shift_loss=(
                "shift_loss",
                "mean",
            ),

            shift_robustness=(
                "shift_robustness_direction",
                "mean",
            ),

            n_shift_observations=(
                "shift_loss",
                "size",
            ),
        )
    )

    # Pair-matched:
    # two directions × all repeats
    pair = (
        cross.groupby(
            [
                "case_id",
                "cancer",
                "model",
                "model_display",
                "pair_key",
                "pair_display",
            ],
            as_index=False,
        )
        .agg(
            mean_pair_shift_loss=(
                "shift_loss",
                "mean",
            ),

            pair_shift_robustness=(
                "shift_robustness_direction",
                "mean",
            ),

            n_shift_observations=(
                "shift_loss",
                "size",
            ),
        )
    )

    return (
        cross,
        overall,
        pair,
    )


# ============================================================
# REPRESENTATION METRICS
# ============================================================

def prepare_overall_metrics():

    cka = pd.read_csv(
        CKA_FILE
    )

    retention = pd.read_csv(
        RETENTION_FILE
    )

    retrieval = pd.read_csv(
        RETRIEVAL_FILE
    )

    nps = pd.read_csv(
        NPS_CASE_FILE
    )

    cka_mean = (
        cka.groupby(
            [
                "cancer",
                "case_id",
                "model",
                "model_display",
            ],
            as_index=False,
        )[
            "cka"
        ]
        .mean()
        .rename(
            columns={
                "cka":
                    "global_alignment"
            }
        )
    )

    retention = retention[
        [
            "cancer",
            "case_id",
            "model",
            "model_display",
            "relative_retention",
        ]
    ].copy()

    retrieval_mean = (
        retrieval.groupby(
            [
                "cancer",
                "case_id",
                "model",
                "model_display",
            ],
            as_index=False,
        )[
            "retrieval_similarity"
        ]
        .mean()
        .rename(
            columns={
                "retrieval_similarity":
                    "cross_scale_retrieval"
            }
        )
    )

    nps = nps[
        [
            "cancer",
            "case_id",
            "model",
            "model_display",
            "neighborhood_preservation_score",
        ]
    ].rename(
        columns={
            "neighborhood_preservation_score":
                "neighborhood_preservation"
        }
    )

    keys = [
        "cancer",
        "case_id",
        "model",
        "model_display",
    ]

    merged = (
        cka_mean
        .merge(
            retention,
            on=keys,
            validate="one_to_one",
        )
        .merge(
            retrieval_mean,
            on=keys,
            validate="one_to_one",
        )
        .merge(
            nps,
            on=keys,
            validate="one_to_one",
        )
    )

    if len(merged) != (
        EXPECTED_CASES
        * len(MODELS)
    ):
        raise RuntimeError(
            f"Overall metric rows: "
            f"{len(merged)} !=568"
        )

    return merged


def prepare_pair_metrics():

    cka = pd.read_csv(
        CKA_FILE
    )

    retrieval = pd.read_csv(
        RETRIEVAL_FILE
    )

    nps = pd.read_csv(
        NPS_PAIR_FILE
    )

    for df in [
        cka,
        retrieval,
        nps,
    ]:

        df[
            "pair_key"
        ] = [
            canonical_pair(
                a,
                b,
            )
            for a, b in zip(
                df[
                    "scale_a"
                ],
                df[
                    "scale_b"
                ],
            )
        ]

    keys = [
        "cancer",
        "case_id",
        "model",
        "model_display",
        "pair_key",
    ]

    cka = cka[
        keys
        + [
            "cka"
        ]
    ].rename(
        columns={
            "cka":
                "pair_cka"
        }
    )

    retrieval = retrieval[
        keys
        + [
            "retrieval_similarity"
        ]
    ].rename(
        columns={
            "retrieval_similarity":
                "pair_retrieval_similarity"
        }
    )

    nps = nps[
        keys
        + [
            "nps"
        ]
    ].rename(
        columns={
            "nps":
                "pair_nps"
        }
    )

    merged = (
        cka
        .merge(
            retrieval,
            on=keys,
            validate="one_to_one",
        )
        .merge(
            nps,
            on=keys,
            validate="one_to_one",
        )
    )

    merged[
        "pair_display"
    ] = merged[
        "pair_key"
    ].map(
        pair_display
    )

    if len(merged) != (
        EXPECTED_CASES
        * len(MODELS)
        * 3
    ):
        raise RuntimeError(
            f"Pair metric rows: "
            f"{len(merged)} !=1704"
        )

    return merged


# ============================================================
# ASSOCIATION INFERENCE
# ============================================================

def structured_association(
    data,
    metric,
    outcome,
    strata_cols,
    entity_col,
    profile_cols,
    n_bootstrap,
    n_permutations,
    rng,
):

    x = data.copy()

    # Adjust for model / model×pair baseline differences
    x[
        "_metric_rank"
    ] = (
        x.groupby(
            strata_cols
        )[
            metric
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    x[
        "_outcome_rank"
    ] = (
        x.groupby(
            strata_cols
        )[
            outcome
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    rho_obs = float(
        spearmanr(
            x[
                "_metric_rank"
            ],
            x[
                "_outcome_rank"
            ],
        ).statistic
    )

    # --------------------------------------------------------
    # Cluster bootstrap by case
    # --------------------------------------------------------

    entities = (
        x[
            entity_col
        ]
        .drop_duplicates()
        .tolist()
    )

    clusters = {
        entity:
            x[
                x[
                    entity_col
                ]
                == entity
            ][
                [
                    "_metric_rank",
                    "_outcome_rank",
                ]
            ]
            .to_numpy(
                dtype=float
            )

        for entity in entities
    }

    boot_rhos = []

    for _ in range(
        n_bootstrap
    ):

        sampled = rng.choice(
            entities,
            size=len(
                entities
            ),
            replace=True,
        )

        arr = np.concatenate(
            [
                clusters[e]
                for e in sampled
            ],
            axis=0,
        )

        rho = spearmanr(
            arr[:, 0],
            arr[:, 1],
        ).statistic

        if np.isfinite(
            rho
        ):
            boot_rhos.append(
                float(rho)
            )

    ci_low, ci_high = np.quantile(
        boot_rhos,
        [
            0.025,
            0.975,
        ],
    )

    # --------------------------------------------------------
    # Structured permutation
    #
    # Entire outcome profile of one slide is moved to another
    # slide, preserving repeated model/pair structure.
    # --------------------------------------------------------

    metric_wide = x.pivot(
        index=entity_col,
        columns=profile_cols,
        values="_metric_rank",
    )

    outcome_wide = x.pivot(
        index=entity_col,
        columns=profile_cols,
        values="_outcome_rank",
    )

    common = (
        metric_wide.index
        .intersection(
            outcome_wide.index
        )
    )

    metric_wide = (
        metric_wide
        .loc[
            common
        ]
        .sort_index(
            axis=1
        )
    )

    outcome_wide = (
        outcome_wide
        .loc[
            common
        ]
        .reindex(
            columns=
            metric_wide.columns
        )
    )

    if (
        metric_wide
        .isna()
        .any()
        .any()
        or
        outcome_wide
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            f"Incomplete structured "
            f"association matrix: {metric}"
        )

    metric_arr = (
        metric_wide
        .to_numpy(
            dtype=float
        )
    )

    outcome_arr = (
        outcome_wide
        .to_numpy(
            dtype=float
        )
    )

    extreme = 0

    for _ in range(
        n_permutations
    ):

        perm = rng.permutation(
            len(common)
        )

        perm_rho = spearmanr(
            metric_arr.ravel(),
            outcome_arr[
                perm
            ].ravel(),
        ).statistic

        if (
            abs(
                perm_rho
            )
            >=
            abs(
                rho_obs
            )
            - 1e-12
        ):
            extreme += 1

    p_perm = (
        extreme
        + 1
    ) / (
        n_permutations
        + 1
    )

    return {
        "metric":
            metric,

        "outcome":
            outcome,

        "n_rows":
            len(x),

        "n_cases":
            len(
                entities
            ),

        "spearman_rho":
            rho_obs,

        "bootstrap_ci_low":
            float(
                ci_low
            ),

        "bootstrap_ci_high":
            float(
                ci_high
            ),

        "permutation_p":
            float(
                p_perm
            ),

        "n_bootstrap":
            n_bootstrap,

        "n_permutations":
            n_permutations,
    }


def run_associations(
    overall_data,
    pair_data,
    n_bootstrap,
    n_permutations,
    seed,
):

    rng = np.random.default_rng(
        seed + 90001
    )

    overall_metrics = [
        "global_alignment",
        "relative_retention",
        "cross_scale_retrieval",
        "neighborhood_preservation",
    ]

    overall_rows = []

    for metric in overall_metrics:

        print(
            f"  overall association: "
            f"{metric}"
        )

        overall_rows.append(
            structured_association(
                data=overall_data,
                metric=metric,
                outcome="shift_robustness",
                strata_cols=[
                    "model",
                ],
                entity_col="case_id",
                profile_cols=[
                    "model",
                ],
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                rng=rng,
            )
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    overall_df[
        "p_holm_family"
    ] = holm_adjust(
        overall_df[
            "permutation_p"
        ]
    )

    pair_metrics = [
        "pair_cka",
        "pair_retrieval_similarity",
        "pair_nps",
    ]

    pair_rows = []

    for metric in pair_metrics:

        print(
            f"  pair-matched association: "
            f"{metric}"
        )

        pair_rows.append(
            structured_association(
                data=pair_data,
                metric=metric,
                outcome="pair_shift_robustness",
                strata_cols=[
                    "model",
                    "pair_key",
                ],
                entity_col="case_id",
                profile_cols=[
                    "model",
                    "pair_key",
                ],
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                rng=rng,
            )
        )

    pair_df = pd.DataFrame(
        pair_rows
    )

    pair_df[
        "p_holm_family"
    ] = holm_adjust(
        pair_df[
            "permutation_p"
        ]
    )

    # Conservative sensitivity:
    # adjust all 7 planned associations together.
    all_p = np.concatenate([
        overall_df[
            "permutation_p"
        ].to_numpy(),

        pair_df[
            "permutation_p"
        ].to_numpy(),
    ])

    all_adj = holm_adjust(
        all_p
    )

    overall_df[
        "p_holm_all7"
    ] = all_adj[
        :len(
            overall_df
        )
    ]

    pair_df[
        "p_holm_all7"
    ] = all_adj[
        len(
            overall_df
        ):
    ]

    return (
        overall_df,
        pair_df,
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
        "FIGURE 8 — MAGNIFICATION-SHIFT "
        "DOWNSTREAM ROBUSTNESS"
    )

    print(
        "=" * 110
    )

    print(
        f"Folds        : {args.n_folds}"
    )

    print(
        f"Repeats      : {args.n_repeats}"
    )

    print(
        f"Bootstrap    : {args.n_bootstrap}"
    )

    print(
        f"Permutations : {args.n_permutations}"
    )

    print(
        "CV partitions: SHARED across "
        "all models and source scales"
    )


    # --------------------------------------------------------
    # Cohort + slide embeddings
    # --------------------------------------------------------

    cohort = build_cohort()

    print(
        "\nCohort:"
    )

    print(
        cohort[
            "cancer"
        ]
        .value_counts()
        .reindex(
            CANCERS
        )
        .to_string()
    )

    matrices = build_slide_matrices(
        cohort
    )


    # --------------------------------------------------------
    # Shared folds
    # --------------------------------------------------------

    shared_splits = (
        make_shared_splits(
            cohort,
            args.n_folds,
            args.n_repeats,
            args.seed,
        )
    )


    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    fold_perf, predictions = (
        run_cross_scale_cv(
            cohort,
            matrices,
            shared_splits,
            args.n_repeats,
            args.seed,
        )
    )

    fold_perf.to_csv(
        OUT_FOLD_PERF,
        index=False,
    )

    predictions.to_csv(
        OUT_PRED,
        index=False,
    )


    # --------------------------------------------------------
    # Repeat OOF performance
    # --------------------------------------------------------

    performance = repeat_performance(
        predictions
    )

    performance.to_csv(
        OUT_REPEAT,
        index=False,
    )


    # --------------------------------------------------------
    # Generalization gaps
    # --------------------------------------------------------

    gaps = generalization_gaps(
        performance
    )

    gaps.to_csv(
        OUT_GAPS,
        index=False,
    )

    direction_summary = (
        gaps.groupby(
            [
                "model",
                "model_display",
                "source_scale",
                "target_scale",
                "direction",
                "pair_key",
            ],
            as_index=False,
        )
        .agg(
            mean_mgg_balanced_accuracy=(
                "mgg_balanced_accuracy",
                "mean",
            ),

            std_mgg_balanced_accuracy=(
                "mgg_balanced_accuracy",
                "std",
            ),

            mean_mgg_macro_f1=(
                "mgg_macro_f1",
                "mean",
            ),

            mean_shift_log_loss=(
                "shift_log_loss",
                "mean",
            ),
        )
    )

    direction_summary.to_csv(
        OUT_DIRECTION,
        index=False,
    )


    # --------------------------------------------------------
    # 3×3 BA matrix summary
    # --------------------------------------------------------

    matrix_summary = (
        performance.groupby(
            [
                "model",
                "model_display",
                "source_scale",
                "target_scale",
            ],
            as_index=False,
        )
        .agg(
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),

            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),

            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),

            mean_log_loss=(
                "log_loss",
                "mean",
            ),
        )
    )

    matrix_summary.to_csv(
        OUT_MATRIX,
        index=False,
    )


    # --------------------------------------------------------
    # Slide-level robustness
    # --------------------------------------------------------

    (
        direction_shift,
        robustness,
        pair_robustness,
    ) = slide_shift_robustness(
        predictions
    )

    direction_shift.to_csv(
        OUT_SHIFT_DIRECTION,
        index=False,
    )

    robustness.to_csv(
        OUT_ROBUST,
        index=False,
    )

    pair_robustness.to_csv(
        OUT_PAIR_ROBUST,
        index=False,
    )

    if len(robustness) != (
        EXPECTED_CASES
        * len(MODELS)
    ):
        raise RuntimeError(
            "Overall robustness rows !=568"
        )

    if len(pair_robustness) != (
        EXPECTED_CASES
        * len(MODELS)
        * 3
    ):
        raise RuntimeError(
            "Pair robustness rows !=1704"
        )


    # --------------------------------------------------------
    # Representation metrics
    # --------------------------------------------------------

    metrics = (
        prepare_overall_metrics()
    )

    pair_metrics = (
        prepare_pair_metrics()
    )

    overall_merged = (
        metrics.merge(
            robustness[
                [
                    "case_id",
                    "cancer",
                    "model",
                    "shift_robustness",
                    "mean_shift_loss",
                ]
            ],
            on=[
                "case_id",
                "cancer",
                "model",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if len(overall_merged) != 568:
        raise RuntimeError(
            f"Overall merged rows "
            f"{len(overall_merged)} !=568"
        )

    overall_merged.to_csv(
        OUT_METRICS,
        index=False,
    )


    pair_merged = (
        pair_metrics.merge(
            pair_robustness[
                [
                    "case_id",
                    "cancer",
                    "model",
                    "pair_key",
                    "pair_shift_robustness",
                    "mean_pair_shift_loss",
                ]
            ],
            on=[
                "case_id",
                "cancer",
                "model",
                "pair_key",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if len(pair_merged) != 1704:
        raise RuntimeError(
            f"Pair merged rows "
            f"{len(pair_merged)} !=1704"
        )

    pair_merged.to_csv(
        OUT_PAIR_METRICS,
        index=False,
    )


    # --------------------------------------------------------
    # Associations
    # --------------------------------------------------------

    print(
        "\nRunning structured associations..."
    )

    (
        overall_assoc,
        pair_assoc,
    ) = run_associations(
        overall_merged,
        pair_merged,
        args.n_bootstrap,
        args.n_permutations,
        args.seed,
    )

    overall_assoc.to_csv(
        OUT_ASSOC,
        index=False,
    )

    pair_assoc.to_csv(
        OUT_PAIR_ASSOC,
        index=False,
    )


    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    protocol = {
        "analysis":
            "Magnification-shift downstream robustness",

        "protocol_id":
            "pathscalebench_native_fov_same_center_v1",

        "n_cases":
            EXPECTED_CASES,

        "models":
            MODELS,

        "scales":
            SCALES,

        "slide_embedding":
            (
                "mean pool patch embeddings "
                "within each scale, then L2 normalize"
            ),

        "classifier":
            {
                "type":
                    "LogisticRegression",

                "scaler":
                    "StandardScaler",

                "C":
                    1.0,

                "class_weight":
                    "balanced",

                "solver":
                    "lbfgs",

                "max_iter":
                    5000,
            },

        "cross_validation":
            {
                "type":
                    "StratifiedGroupKFold",

                "n_folds":
                    args.n_folds,

                "n_repeats":
                    args.n_repeats,

                "group":
                    "TCGA patient ID",

                "shared_partitions_across_models":
                    True,

                "shared_partitions_across_source_scales":
                    True,
            },

        "primary_performance":
            "pooled OOF balanced accuracy",

        "mgg":
            (
                "within-source balanced accuracy "
                "minus shifted-target balanced accuracy"
            ),

        "slide_shift_robustness":
            (
                "negative mean change in true-class NLL "
                "across all six directed scale shifts"
            ),

        "pair_shift_robustness":
            (
                "negative mean change in true-class NLL "
                "across the two directed shifts belonging "
                "to each undirected scale pair"
            ),

        "overall_associations":
            (
                "within-model percentile ranks; pooled "
                "Spearman; slide-cluster bootstrap; "
                "whole-slide model-profile permutation"
            ),

        "pair_matched_associations":
            (
                "within-model×scale-pair percentile ranks; "
                "pooled Spearman; slide-cluster bootstrap; "
                "whole-slide model×pair-profile permutation"
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
        "MEAN BALANCED ACCURACY "
        "BY MODEL × SOURCE × TARGET"
    )

    print(
        "=" * 110
    )

    for model in MODELS:

        x = (
            matrix_summary[
                matrix_summary[
                    "model"
                ]
                == model
            ]
            .pivot(
                index="source_scale",
                columns="target_scale",
                values="mean_balanced_accuracy",
            )
            .reindex(
                index=SCALES,
                columns=SCALES,
            )
        )

        print(
            f"\n{MODEL_DISPLAY[model]}"
        )

        print(
            x.round(4).to_string()
        )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "DIRECTION-LEVEL "
        "MAGNIFICATION GENERALIZATION GAP"
    )

    print(
        "=" * 110
    )

    gap_table = (
        direction_summary.pivot(
            index="model_display",
            columns="direction",
            values="mean_mgg_balanced_accuracy",
        )
    )

    print(
        gap_table
        .round(4)
        .to_string()
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "MEAN SHIFT ROBUSTNESS BY MODEL"
    )

    print(
        "=" * 110
    )

    robust_summary = (
        robustness.groupby(
            [
                "model",
                "model_display",
            ],
            as_index=False,
        )
        .agg(
            mean_shift_robustness=(
                "shift_robustness",
                "mean",
            ),

            median_shift_robustness=(
                "shift_robustness",
                "median",
            ),

            std_shift_robustness=(
                "shift_robustness",
                "std",
            ),
        )
        .sort_values(
            "mean_shift_robustness",
            ascending=False,
        )
    )

    print(
        robust_summary
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
        "OVERALL REPRESENTATION → "
        "SHIFT ROBUSTNESS ASSOCIATIONS"
    )

    print(
        "=" * 110
    )

    print(
        overall_assoc[
            [
                "metric",
                "spearman_rho",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_p",
                "p_holm_family",
                "p_holm_all7",
            ]
        ]
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
        "PAIR-MATCHED REPRESENTATION → "
        "SHIFT ROBUSTNESS ASSOCIATIONS"
    )

    print(
        "=" * 110
    )

    print(
        pair_assoc[
            [
                "metric",
                "spearman_rho",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_p",
                "p_holm_family",
                "p_holm_all7",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )


    print(
        "\nOutputs:"
    )

    for path in [
        OUT_SLIDE_META,
        OUT_FOLDS,
        OUT_FOLD_PERF,
        OUT_PRED,
        OUT_REPEAT,
        OUT_GAPS,
        OUT_DIRECTION,
        OUT_MATRIX,
        OUT_SHIFT_DIRECTION,
        OUT_ROBUST,
        OUT_PAIR_ROBUST,
        OUT_METRICS,
        OUT_PAIR_METRICS,
        OUT_ASSOC,
        OUT_PAIR_ASSOC,
        OUT_PROTOCOL,
    ]:
        print(path)


if __name__ == "__main__":
    main()
