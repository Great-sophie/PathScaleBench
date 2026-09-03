#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 8
Magnification-shift robustness + formal model statistics
+ criterion validity
=========================================================

A. Source-scale -> target-scale balanced accuracy
   8 PFMs × 3×3 matrices.

B. Magnification Generalization Gap (MGG)
   - six cross-scale directions per model
   - lower = better
   - formal paired model comparison:
       Friedman across the six matched shift directions
       pairwise Wilcoxon + Holm in supplementary output
   - CV repeats are NOT treated as independent units.

C. Probabilistic shift robustness
   - case-level values, 71 paired cases × 8 PFMs
   - higher = better
   - formal paired model comparison:
       Friedman across models
       pairwise Wilcoxon + Holm

D. Pair-matched criterion validity
   - case × model × scale-pair
   - retrieval(A,B) vs robustness to corresponding A<->B shift
   - within model × scale-pair percentile ranks
   - frozen structured association from Analysis 20:
       rho = 0.159148
       bootstrap CI = [0.058968, 0.264390]
       profile-permutation p = 0.000300
       Holm across all 7 association tests = 0.002100

No embeddings or classifiers are recomputed.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
)


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


PREFIX = "31b_figure8_magnification_robustness_FINAL_clean"


OUT_MGG_STATS = (
    RESULT_ROOT
    / f"{PREFIX}_mgg_model_statistics.csv"
)

OUT_MGG_PAIRWISE = (
    RESULT_ROOT
    / f"{PREFIX}_mgg_pairwise_wilcoxon_holm.csv"
)

OUT_SHIFT_STATS = (
    RESULT_ROOT
    / f"{PREFIX}_shift_robustness_model_statistics.csv"
)

OUT_SHIFT_PAIRWISE = (
    RESULT_ROOT
    / f"{PREFIX}_shift_robustness_pairwise_wilcoxon_holm.csv"
)

OUT_AUDIT = (
    RESULT_ROOT
    / f"{PREFIX}_audit.txt"
)

OUT_PNG = (
    FIG_ROOT
    / "Figure8_Magnification_Robustness_FINAL_clean_8PFM.png"
)

OUT_PDF = (
    FIG_ROOT
    / "Figure8_Magnification_Robustness_FINAL_clean_8PFM.pdf"
)

OUT_TIFF = (
    FIG_ROOT
    / "Figure8_Magnification_Robustness_FINAL_clean_8PFM_600dpi.tiff"
)


# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "midnight",
    "virchow2",
    "uni2",
    "phikon",
    "uni",
    "gigapath",
    "gigapath_flash",
    "virchow",
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


SCALE_DISPLAY = {
    "40x": "40×",
    "10x": "10×",
    "2p5x": "2.5×",
}


# ============================================================
# FROZEN BALANCED ACCURACY MATRICES
# ============================================================

BA = {

    "phikon": np.array([
        [0.7462, 0.6873, 0.1256],
        [0.5958, 0.8291, 0.2121],
        [0.2135, 0.4178, 0.6437],
    ]),

    "uni": np.array([
        [0.7489, 0.6111, 0.1406],
        [0.6413, 0.8289, 0.2074],
        [0.2968, 0.3963, 0.7067],
    ]),

    "virchow": np.array([
        [0.7252, 0.5367, 0.1287],
        [0.4417, 0.7819, 0.1404],
        [0.1902, 0.2875, 0.6517],
    ]),

    "virchow2": np.array([
        [0.7859, 0.7448, 0.5686],
        [0.6732, 0.8576, 0.7465],
        [0.4256, 0.7509, 0.8550],
    ]),

    "uni2": np.array([
        [0.8113, 0.6890, 0.6076],
        [0.5982, 0.8609, 0.8271],
        [0.3531, 0.6971, 0.8856],
    ]),

    "midnight": np.array([
        [0.8501, 0.8011, 0.7227],
        [0.8132, 0.8765, 0.8596],
        [0.7142, 0.8356, 0.8912],
    ]),

    "gigapath": np.array([
        [0.6788, 0.5815, 0.1665],
        [0.3704, 0.7991, 0.4047],
        [0.2916, 0.4652, 0.7312],
    ]),

    "gigapath_flash": np.array([
        [0.6636, 0.5387, 0.1660],
        [0.3646, 0.8070, 0.2727],
        [0.2245, 0.3758, 0.6841],
    ]),
}


# ============================================================
# FROZEN DIRECTION-LEVEL MGG
# ============================================================

DIRECTIONS = [
    "10→2.5",
    "10→40",
    "2.5→10",
    "2.5→40",
    "40→10",
    "40→2.5",
]


MGG = {

    "gigapath": [
        0.3944,
        0.4287,
        0.2660,
        0.4396,
        0.0972,
        0.5122,
    ],

    "gigapath_flash": [
        0.5342,
        0.4424,
        0.3083,
        0.4596,
        0.1249,
        0.4976,
    ],

    "midnight": [
        0.0169,
        0.0632,
        0.0556,
        0.1770,
        0.0490,
        0.1274,
    ],

    "phikon": [
        0.6170,
        0.2333,
        0.2260,
        0.4302,
        0.0590,
        0.6206,
    ],

    "uni": [
        0.6215,
        0.1876,
        0.3104,
        0.4099,
        0.1378,
        0.6083,
    ],

    "uni2": [
        0.0338,
        0.2627,
        0.1885,
        0.5325,
        0.1223,
        0.2037,
    ],

    "virchow": [
        0.6415,
        0.3403,
        0.3642,
        0.4615,
        0.1885,
        0.5965,
    ],

    "virchow2": [
        0.1112,
        0.1844,
        0.1041,
        0.4293,
        0.0411,
        0.2174,
    ],
}


# ============================================================
# FROZEN MODEL-MEAN PROBABILISTIC SHIFT ROBUSTNESS
# ============================================================

SHIFT_MEAN = {
    "midnight": -0.3188,
    "virchow2": -0.5797,
    "uni2": -0.5929,
    "phikon": -1.2119,
    "uni": -1.2275,
    "gigapath": -1.2885,
    "gigapath_flash": -2.0235,
    "virchow": -2.9459,
}


# ============================================================
# FROZEN PAIR-MATCHED CRITERION VALIDITY
# ============================================================

PAIR_MATCHED = {
    "rho": 0.159148,
    "ci_low": 0.058968,
    "ci_high": 0.264390,
    "p_perm": 0.000300,
    "p_holm_all7": 0.002100,
    "n": 1704,
}


# ============================================================
# HELPERS
# ============================================================

def normalize_model(x):

    s = (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "phikon": "phikon",
        "uni": "uni",
        "virchow": "virchow",
        "virchow2": "virchow2",

        "uni2": "uni2",
        "uni2_h": "uni2",

        "midnight": "midnight",
        "midnight_12k": "midnight",

        "gigapath": "gigapath",

        "gigapath_flash": "gigapath_flash",
        "gigapathflash": "gigapath_flash",
    }

    return aliases.get(
        s,
        s,
    )


def holm_adjust(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(
        pvalues
    )

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

    running = 0.0

    for i, p in enumerate(
        sorted_p
    ):

        value = (
            (m - i)
            * p
        )

        running = max(
            running,
            value,
        )

        adjusted_sorted[
            i
        ] = min(
            running,
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


def find_shift_case_file():

    """
    Find the Analysis-20 case-level probabilistic
    shift-robustness table.

    Required:
        model
        case identifier
        shift robustness / probabilistic robustness
    """

    patterns = [
        "20_magnification_shift_robustness_8pfm*.csv",
        "*shift_robustness*8pfm*.csv",
        "*slide_shift_robustness*.csv",
    ]

    candidates = []

    for pattern in patterns:

        candidates += sorted(
            RESULT_ROOT.glob(
                pattern
            )
        )

    seen = set()

    for path in candidates:

        if str(path) in seen:
            continue

        seen.add(
            str(path)
        )

        try:

            d = pd.read_csv(
                path,
                nrows=10,
            )

        except Exception:

            continue

        cols_lower = {
            c.lower():
                c
            for c in d.columns
        }

        model_col = next(
            (
                cols_lower[x]
                for x in [
                    "model",
                    "model_display",
                ]
                if x in cols_lower
            ),
            None,
        )

        case_col = next(
            (
                cols_lower[x]
                for x in [
                    "case_id",
                    "slide_id",
                    "wsi_id",
                    "case",
                ]
                if x in cols_lower
            ),
            None,
        )

        robustness_candidates = [
            c
            for c in d.columns
            if (
                "robust" in c.lower()
                and
                "rank" not in c.lower()
            )
        ]

        if (
            model_col is not None
            and
            case_col is not None
            and
            robustness_candidates
        ):

            return (
                path,
                model_col,
                case_col,
                robustness_candidates[0],
            )

    return None


def find_pair_matched_file():

    """
    Use the verified Analysis-21d plotting dataframe.

    71 cases × 8 PFMs × 3 scale pairs = 1704 rows.
    """

    path = (
        RESULT_ROOT
        / "21d_figure8D_pair_matched_retrieval_vs_shift_robustness_plot_data.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing verified Figure-8D plotting data:\n{path}"
        )


    d = pd.read_csv(
        path
    )


    required = [
        "case_id",
        "cancer",
        "model",
        "model_display",
        "pair_key",
        "pair_retrieval_similarity",
        "pair_shift_robustness",
        "retrieval_percentile",
        "robustness_percentile",
    ]


    missing = [
        c
        for c in required
        if c not in d.columns
    ]


    if missing:

        raise RuntimeError(
            f"Analysis-21d file missing columns: {missing}"
        )


    if len(d) != PAIR_MATCHED["n"]:

        raise RuntimeError(
            f"Expected {PAIR_MATCHED['n']} rows, "
            f"found {len(d)}"
        )


    x = pd.to_numeric(
        d[
            "retrieval_percentile"
        ],
        errors="coerce",
    )

    y = pd.to_numeric(
        d[
            "robustness_percentile"
        ],
        errors="coerce",
    )


    if (
        x.isna().any()
        or
        y.isna().any()
    ):

        raise RuntimeError(
            "NaN found in pair-matched percentile coordinates."
        )


    if not (
        x.between(
            0,
            1
        ).all()
        and
        y.between(
            0,
            1
        ).all()
    ):

        raise RuntimeError(
            "Pair-matched percentile coordinates outside [0,1]."
        )


    expected_pairs = {
        "40x__10x",
        "40x__2p5x",
        "10x__2p5x",
    }


    observed_pairs = set(
        d[
            "pair_key"
        ].astype(str)
    )


    if observed_pairs != expected_pairs:

        raise RuntimeError(
            f"Unexpected scale pairs: {observed_pairs}"
        )


    print(
        "\n[PAIR-MATCHED PLOT DATA RESOLVED]"
    )

    print(
        "file :",
        path
    )

    print(
        "shape:",
        d.shape
    )

    print(
        "x    : retrieval_percentile"
    )

    print(
        "y    : robustness_percentile"
    )

    print(
        "pairs:",
        sorted(
            observed_pairs
        )
    )

    print(
        f"x range: [{x.min():.6f}, {x.max():.6f}]"
    )

    print(
        f"y range: [{y.min():.6f}, {y.max():.6f}]"
    )


    return path


# ============================================================
# FORMAL MODEL STATISTICS — MGG
# ============================================================

mgg_matrix = np.column_stack(
    [
        np.asarray(
            MGG[
                model
            ],
            dtype=float,
        )
        for model in MODELS
    ]
)


mgg_arrays = [
    mgg_matrix[
        :,
        i
    ]
    for i in range(
        len(
            MODELS
        )
    )
]


mgg_chi2, mgg_p = (
    friedmanchisquare(
        *mgg_arrays
    )
)


mgg_W = (
    mgg_chi2
    /
    (
        len(
            DIRECTIONS
        )
        *
        (
            len(
                MODELS
            )
            - 1
        )
    )
)


mgg_stat_df = pd.DataFrame([{
    "analysis":
        "direction-level MGG",

    "n_blocks":
        len(
            DIRECTIONS
        ),

    "n_models":
        len(
            MODELS
        ),

    "friedman_chi_square":
        mgg_chi2,

    "df":
        len(
            MODELS
        )
        - 1,

    "p_value":
        mgg_p,

    "kendall_W":
        mgg_W,
}])


mgg_stat_df.to_csv(
    OUT_MGG_STATS,
    index=False,
)


# Pairwise Wilcoxon
mgg_pairwise = []

for a, b in combinations(
    MODELS,
    2,
):

    x = np.asarray(
        MGG[
            a
        ],
        dtype=float,
    )

    y = np.asarray(
        MGG[
            b
        ],
        dtype=float,
    )

    try:

        stat, p = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
        )

    except ValueError:

        stat = np.nan
        p = 1.0

    mgg_pairwise.append({
        "model_a":
            MODEL_DISPLAY[a],

        "model_b":
            MODEL_DISPLAY[b],

        "wilcoxon_stat":
            stat,

        "raw_p":
            p,

        "mean_mgg_a":
            x.mean(),

        "mean_mgg_b":
            y.mean(),

        "mean_difference_a_minus_b":
            x.mean()
            -
            y.mean(),
    })


mgg_pairwise = pd.DataFrame(
    mgg_pairwise
)

mgg_pairwise[
    "holm_p"
] = holm_adjust(
    mgg_pairwise[
        "raw_p"
    ]
)

mgg_pairwise[
    "significant_holm_0_05"
] = (
    mgg_pairwise[
        "holm_p"
    ]
    < 0.05
)

mgg_pairwise.to_csv(
    OUT_MGG_PAIRWISE,
    index=False,
)


# ============================================================
# FORMAL MODEL STATISTICS — SHIFT ROBUSTNESS
# ============================================================

shift_source = find_shift_case_file()

shift_case_df = None
shift_chi2 = np.nan
shift_p = np.nan
shift_W = np.nan


if shift_source is not None:

    (
        shift_path,
        shift_model_col,
        shift_case_col,
        shift_value_col,
    ) = shift_source

    raw_shift = pd.read_csv(
        shift_path
    )

    shift_case_df = pd.DataFrame({
        "model":
            raw_shift[
                shift_model_col
            ].map(
                normalize_model
            ),

        "case_id":
            raw_shift[
                shift_case_col
            ].astype(str),

        "shift_robustness":
            pd.to_numeric(
                raw_shift[
                    shift_value_col
                ],
                errors="coerce",
            ),
    })

    shift_case_df = (
        shift_case_df[
            shift_case_df[
                "model"
            ].isin(
                MODELS
            )
        ]
        .dropna()
        .copy()
    )


    # Collapse accidental duplicate rows only by mean.
    shift_case_df = (
        shift_case_df.groupby(
            [
                "case_id",
                "model",
            ],
            as_index=False,
        )
        .agg(
            shift_robustness=(
                "shift_robustness",
                "mean",
            )
        )
    )


    pivot = (
        shift_case_df.pivot(
            index="case_id",
            columns="model",
            values="shift_robustness",
        )
        .reindex(
            columns=MODELS
        )
        .dropna()
    )


    if len(
        pivot
    ) == 71:

        arrays = [
            pivot[
                model
            ].to_numpy(
                dtype=float
            )
            for model in MODELS
        ]


        shift_chi2, shift_p = (
            friedmanchisquare(
                *arrays
            )
        )


        shift_W = (
            shift_chi2
            /
            (
                len(
                    pivot
                )
                *
                (
                    len(
                        MODELS
                    )
                    - 1
                )
            )
        )


        pd.DataFrame([{
            "analysis":
                "case-level probabilistic shift robustness",

            "n_cases":
                len(
                    pivot
                ),

            "n_models":
                len(
                    MODELS
                ),

            "friedman_chi_square":
                shift_chi2,

            "df":
                len(
                    MODELS
                )
                - 1,

            "p_value":
                shift_p,

            "kendall_W":
                shift_W,

            "input_file":
                str(
                    shift_path
                ),

            "value_column":
                shift_value_col,
        }]).to_csv(
            OUT_SHIFT_STATS,
            index=False,
        )


        # Pairwise post hoc
        rows = []

        for a, b in combinations(
            MODELS,
            2,
        ):

            x = pivot[
                a
            ].to_numpy(
                dtype=float
            )

            y = pivot[
                b
            ].to_numpy(
                dtype=float
            )

            stat, p = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
            )

            rows.append({
                "model_a":
                    MODEL_DISPLAY[a],

                "model_b":
                    MODEL_DISPLAY[b],

                "wilcoxon_stat":
                    stat,

                "raw_p":
                    p,

                "mean_a":
                    x.mean(),

                "mean_b":
                    y.mean(),

                "mean_difference_a_minus_b":
                    x.mean()
                    -
                    y.mean(),
            })


        shift_pairwise = pd.DataFrame(
            rows
        )

        shift_pairwise[
            "holm_p"
        ] = holm_adjust(
            shift_pairwise[
                "raw_p"
            ]
        )

        shift_pairwise[
            "significant_holm_0_05"
        ] = (
            shift_pairwise[
                "holm_p"
            ]
            < 0.05
        )

        shift_pairwise.to_csv(
            OUT_SHIFT_PAIRWISE,
            index=False,
        )


# ============================================================
# CRITERION VALIDITY PLOT DATA
# ============================================================

pair_source = find_pair_matched_file()


pair_raw = pd.read_csv(
    pair_source
)


criterion_df = pd.DataFrame({
    "case_id":
        pair_raw[
            "case_id"
        ].astype(str),

    "cancer":
        pair_raw[
            "cancer"
        ].astype(str),

    "model":
        pair_raw[
            "model"
        ].astype(str),

    "model_display":
        pair_raw[
            "model_display"
        ].astype(str),

    "pair_key":
        pair_raw[
            "pair_key"
        ].astype(str),

    "retrieval_similarity":
        pd.to_numeric(
            pair_raw[
                "pair_retrieval_similarity"
            ],
            errors="coerce",
        ),

    "shift_robustness":
        pd.to_numeric(
            pair_raw[
                "pair_shift_robustness"
            ],
            errors="coerce",
        ),

    "retrieval_rank":
        pd.to_numeric(
            pair_raw[
                "retrieval_percentile"
            ],
            errors="coerce",
        ),

    "robustness_rank":
        pd.to_numeric(
            pair_raw[
                "robustness_percentile"
            ],
            errors="coerce",
        ),
})


criterion_df = criterion_df.dropna().copy()


if len(
    criterion_df
) != PAIR_MATCHED[
    "n"
]:

    raise RuntimeError(
        f"Expected {PAIR_MATCHED['n']} valid pair-matched rows; "
        f"found {len(criterion_df)}"
    )


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family":
        "DejaVu Sans",

    "font.size":
        9.0,

    "axes.titlesize":
        10.5,

    "axes.labelsize":
        9.2,

    "xtick.labelsize":
        8.0,

    "ytick.labelsize":
        8.0,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        13.8,
        9.25,
    ),
    facecolor="white",
)


outer = fig.add_gridspec(
    2,
    1,

    height_ratios=[
        1.16,
        1.0,
    ],

    hspace=0.42,
)


# ============================================================
# PANEL A — SOURCE → TARGET BALANCED ACCURACY
# ============================================================

gs_a = outer[
    0
].subgridspec(
    2,
    5,

    width_ratios=[
        1,
        1,
        1,
        1,
        0.045,
    ],

    hspace=0.30,
    wspace=0.25,
)


im = None
heat_axes = []


for i, model in enumerate(
    MODELS
):

    row = i // 4
    col = i % 4

    ax = fig.add_subplot(
        gs_a[
            row,
            col,
        ]
    )

    heat_axes.append(
        ax
    )

    matrix = BA[
        model
    ]


    im = ax.imshow(
        matrix,

        cmap="Blues",

        vmin=0.0,
        vmax=1.0,

        aspect="equal",

        interpolation="nearest",
    )


    for r in range(3):

        for c in range(3):

            value = matrix[
                r,
                c
            ]

            ax.text(
                c,
                r,

                f"{value:.2f}",

                ha="center",
                va="center",

                fontsize=7.7,

                fontweight=(
                    "bold"
                    if r == c
                    else "normal"
                ),

                color=(
                    "white"
                    if value >= 0.58
                    else "black"
                ),
            )


    # Explicitly mark same-scale evaluation.
    for k in range(3):

        rect = plt.Rectangle(
            (
                k - 0.5,
                k - 0.5,
            ),

            1,
            1,

            fill=False,

            edgecolor="black",

            linewidth=0.95,
        )

        ax.add_patch(
            rect
        )


    ax.set_xticks(
        range(3)
    )

    ax.set_xticklabels(
        [
            SCALE_DISPLAY[
                x
            ]
            for x in SCALES
        ]
    )


    ax.set_yticks(
        range(3)
    )

    ax.set_yticklabels(
        [
            SCALE_DISPLAY[
                x
            ]
            for x in SCALES
        ]
    )


    ax.set_title(
        MODEL_DISPLAY[
            model
        ],

        fontsize=9.2,

        fontweight="bold",

        pad=4,
    )


    ax.tick_params(
        length=0,

        labelsize=7.3,
    )


    for spine in ax.spines.values():

        spine.set_visible(
            False
        )


# Shared colorbar
cax = fig.add_subplot(
    gs_a[
        :,
        4,
    ]
)

cbar = fig.colorbar(
    im,
    cax=cax,
)

cbar.set_label(
    "Balanced accuracy",

    fontsize=8.3,
)

cbar.ax.tick_params(
    labelsize=7.5,
)


# Global Panel-A labels
fig.text(
    0.055,
    0.965,

    "A   Source-to-target magnification generalization",

    fontsize=11.3,

    fontweight="bold",

    ha="left",

    va="top",
)


fig.text(
    0.475,
    0.935,

    "Target (test) scale",

    ha="center",

    va="top",

    fontsize=8.6,

    color="0.32",
)


fig.text(
    0.035,
    0.755,

    "Source (train) scale",

    ha="center",

    va="center",

    rotation=90,

    fontsize=8.6,

    color="0.32",
)


# ============================================================
# LOWER PANELS
# ============================================================

lower = outer[
    1
].subgridspec(
    1,
    3,

    width_ratios=[
        1.0,
        1.0,
        1.18,
    ],

    wspace=0.36,
)


# ============================================================
# PANEL B — MAGNIFICATION GENERALIZATION GAP
# ============================================================

ax_b = fig.add_subplot(
    lower[
        0,
        0,
    ]
)


mgg_mean = {
    model:
        float(
            np.mean(
                MGG[
                    model
                ]
            )
        )
    for model in MODELS
}


mgg_order = sorted(
    MODELS,

    key=lambda m:
        mgg_mean[
            m
        ],
)


y = np.arange(
    len(
        mgg_order
    )
)


for yi, model in enumerate(
    mgg_order
):

    values = np.asarray(
        MGG[
            model
        ],
        dtype=float,
    )


    jitter = np.linspace(
        -0.14,
        0.14,
        len(
            values
        ),
    )


    # Six matched directional gaps.
    ax_b.scatter(
        values,
        yi + jitter,

        s=17,

        facecolor="#6FA8D1",

        edgecolor="none",

        alpha=0.48,

        zorder=2,
    )


    mean_value = float(
        values.mean()
    )


    ax_b.scatter(
        mean_value,
        yi,

        marker="D",

        s=53,

        facecolor="black",

        edgecolor="white",

        linewidth=0.6,

        zorder=4,
    )


    ax_b.text(
        mean_value + 0.012,
        yi,

        f"{mean_value:.3f}",

        fontsize=7.5,

        fontweight="bold",

        va="center",
    )


ax_b.set_yticks(
    y
)

ax_b.set_yticklabels(
    [
        MODEL_DISPLAY[
            m
        ]
        for m in mgg_order
    ]
)

ax_b.invert_yaxis()


ax_b.set_xlabel(
    "Magnification Generalization Gap (MGG)\n"
    "(lower is better)"
)


ax_b.set_title(
    "B   Magnification generalization gap",

    loc="left",

    fontweight="bold",

    pad=26,
)


if mgg_p < 0.001:

    mgg_ptext = "<0.001"

else:

    mgg_ptext = (
        f"{mgg_p:.3f}"
    )


# Formal statistic outside data cloud.
ax_b.text(
    0.00,
    1.015,

    (
        f"Friedman χ²(7) = {mgg_chi2:.2f}"
        f"  ·  p {mgg_ptext}"
        f"  ·  Kendall W = {mgg_W:.3f}"
    ),

    transform=ax_b.transAxes,

    fontsize=7.5,

    color="0.38",

    ha="left",

    va="bottom",
)


ax_b.grid(
    axis="x",

    alpha=0.075,

    linewidth=0.65,
)

ax_b.spines[
    "top"
].set_visible(False)

ax_b.spines[
    "right"
].set_visible(False)

ax_b.spines[
    "left"
].set_visible(False)

ax_b.tick_params(
    axis="y",
    length=0,
)


# ============================================================
# PANEL C — PROBABILISTIC SHIFT ROBUSTNESS
# ============================================================

ax_c = fig.add_subplot(
    lower[
        0,
        1,
    ]
)


shift_order = sorted(
    MODELS,

    key=lambda m:
        SHIFT_MEAN[
            m
        ],

    reverse=True,
)


y = np.arange(
    len(
        shift_order
    )
)


if shift_case_df is not None:

    # --------------------------------------------------------
    # Horizontal boxplot + paired case points
    # --------------------------------------------------------

    box_data = []

    for model in shift_order:

        vals = (
            shift_case_df.loc[
                shift_case_df[
                    "model"
                ]
                == model,
                "shift_robustness",
            ]
            .to_numpy(
                dtype=float
            )
        )

        box_data.append(
            vals
        )


    bp = ax_c.boxplot(
        box_data,

        positions=y,

        vert=False,

        widths=0.48,

        patch_artist=True,

        showfliers=False,

        medianprops={
            "color":
                "black",

            "linewidth":
                1.0,
        },

        whiskerprops={
            "color":
                "0.48",

            "linewidth":
                0.75,
        },

        capprops={
            "color":
                "0.48",

            "linewidth":
                0.75,
        },

        boxprops={
            "linewidth":
                0.8,

            "edgecolor":
                "#6FA8D1",
        },
    )


    for patch in bp[
        "boxes"
    ]:

        patch.set_facecolor(
            "#DCEAF5"
        )

        patch.set_alpha(
            0.72
        )


    # Case points + mean diamonds
    for yi, model in enumerate(
        shift_order
    ):

        vals = (
            shift_case_df.loc[
                shift_case_df[
                    "model"
                ]
                == model,
                "shift_robustness",
            ]
            .to_numpy(
                dtype=float
            )
        )


        rng = np.random.default_rng(
            2026
            +
            yi
        )


        jitter = rng.uniform(
            -0.13,
            0.13,

            size=len(
                vals
            ),
        )


        ax_c.scatter(
            vals,
            yi + jitter,

            s=7,

            facecolor="#6FA8D1",

            edgecolor="none",

            alpha=0.20,

            zorder=2,
        )


        mean_value = float(
            vals.mean()
        )


        # Ensure the case-level source reproduces the
        # audited frozen model mean.
        expected_mean = (
            SHIFT_MEAN[
                model
            ]
        )


        if abs(
            mean_value
            -
            expected_mean
        ) > 0.01:

            raise RuntimeError(
                f"Shift-robustness mean mismatch for "
                f"{model}: observed={mean_value:.6f}, "
                f"expected={expected_mean:.6f}"
            )


        ax_c.scatter(
            mean_value,
            yi,

            marker="D",

            s=54,

            facecolor="black",

            edgecolor="white",

            linewidth=0.6,

            zorder=5,
        )


        ax_c.text(
            mean_value + 0.06,
            yi,

            f"{mean_value:.3f}",

            fontsize=7.4,

            fontweight="bold",

            va="center",

            zorder=6,
        )


else:

    # Safe fallback — never invent case-level values.
    for yi, model in enumerate(
        shift_order
    ):

        mean_value = (
            SHIFT_MEAN[
                model
            ]
        )


        ax_c.scatter(
            mean_value,
            yi,

            marker="D",

            s=54,

            facecolor="black",

            edgecolor="white",

            linewidth=0.6,

            zorder=5,
        )


        ax_c.text(
            mean_value + 0.06,
            yi,

            f"{mean_value:.3f}",

            fontsize=7.4,

            fontweight="bold",

            va="center",
        )


ax_c.set_yticks(
    y
)

ax_c.set_yticklabels(
    [
        MODEL_DISPLAY[
            m
        ]
        for m in shift_order
    ]
)

ax_c.invert_yaxis()


ax_c.set_xlabel(
    "Probabilistic shift robustness\n"
    "(higher is better)"
)


ax_c.set_title(
    "C   Probabilistic shift robustness",

    loc="left",

    fontweight="bold",

    pad=26,
)


if np.isfinite(
    shift_p
):

    if shift_p < 0.001:

        shift_ptext = "<0.001"

    else:

        shift_ptext = (
            f"{shift_p:.3f}"
        )


    ax_c.text(
        0.00,
        1.015,

        (
            f"Friedman χ²(7) = {shift_chi2:.2f}"
            f"  ·  p {shift_ptext}"
            f"  ·  Kendall W = {shift_W:.3f}"
        ),

        transform=ax_c.transAxes,

        fontsize=7.5,

        color="0.38",

        ha="left",

        va="bottom",
    )


ax_c.grid(
    axis="x",

    alpha=0.075,

    linewidth=0.65,
)

ax_c.spines[
    "top"
].set_visible(False)

ax_c.spines[
    "right"
].set_visible(False)

ax_c.spines[
    "left"
].set_visible(False)

ax_c.tick_params(
    axis="y",
    length=0,
)


# ============================================================
# PANEL D — PAIR-MATCHED CRITERION VALIDITY
# ============================================================

ax_d = fig.add_subplot(
    lower[
        0,
        2,
    ]
)


PAIR_PLOT_STYLE = {
    "40x__10x": {
        "label":
            "40×↔10×",

        "color":
            "#D55E00",

        "marker":
            "o",
    },

    "10x__2p5x": {
        "label":
            "10×↔2.5×",

        "color":
            "#0072B2",

        "marker":
            "s",
    },

    "40x__2p5x": {
        "label":
            "40×↔2.5×",

        "color":
            "#009E73",

        "marker":
            "^",
    },
}


# Draw 568 points per scale pair.
for pair_key, style in PAIR_PLOT_STYLE.items():

    sub = criterion_df[
        criterion_df[
            "pair_key"
        ]
        == pair_key
    ]


    if len(
        sub
    ) != 568:

        raise RuntimeError(
            f"Expected 568 rows for {pair_key}; "
            f"found {len(sub)}"
        )


    ax_d.scatter(
        sub[
            "retrieval_rank"
        ],

        sub[
            "robustness_rank"
        ],

        s=10,

        marker=style[
            "marker"
        ],

        facecolor=style[
            "color"
        ],

        edgecolor="none",

        alpha=0.16,

        label=style[
            "label"
        ],

        zorder=2,
    )


# ------------------------------------------------------------
# Visual guide only
# Formal inference is Spearman + structured permutation.
# ------------------------------------------------------------

x = criterion_df[
    "retrieval_rank"
].to_numpy(
    dtype=float
)

yv = criterion_df[
    "robustness_rank"
].to_numpy(
    dtype=float
)


coef = np.polyfit(
    x,
    yv,
    1,
)


xx = np.linspace(
    0,
    1,
    100,
)


yy = (
    coef[
        0
    ]
    *
    xx
    +
    coef[
        1
    ]
)


ax_d.plot(
    xx,
    yy,

    linestyle="--",

    linewidth=0.9,

    color="0.45",

    alpha=0.68,

    zorder=1,
)


ax_d.set_xlim(
    0,
    1,
)

ax_d.set_ylim(
    0,
    1,
)


ax_d.set_xlabel(
    "Pair-matched retrieval percentile"
)

ax_d.set_ylabel(
    "Corresponding shift-robustness percentile"
)


ax_d.set_title(
    "D   Pair-matched criterion validity",

    loc="left",

    fontweight="bold",

    pad=35,
)


# Two-line formal statistical subtitle.
ax_d.text(
    0.00,
    1.015,

    (
        f"ρ = {PAIR_MATCHED['rho']:.3f}"
        f"  ·  95% CI "
        f"[{PAIR_MATCHED['ci_low']:.3f}, "
        f"{PAIR_MATCHED['ci_high']:.3f}]\n"
        f"permutation p = {PAIR_MATCHED['p_perm']:.4f}"
        f"  ·  Holm p = {PAIR_MATCHED['p_holm_all7']:.4f}"
    ),

    transform=ax_d.transAxes,

    fontsize=7.5,

    color="0.34",

    ha="left",

    va="bottom",

    linespacing=1.22,
)


ax_d.legend(
    frameon=False,

    loc="lower right",

    fontsize=7.1,

    handletextpad=0.3,

    labelspacing=0.35,
)


ax_d.grid(
    alpha=0.06,

    linewidth=0.65,
)

ax_d.spines[
    "top"
].set_visible(False)

ax_d.spines[
    "right"
].set_visible(False)


# ============================================================
# GLOBAL LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.955,
    top=0.925,
    bottom=0.085,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_TIFF,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={
        "compression": "tiff_lzw",
    },
)

plt.close(
    fig
)


# ============================================================
# AUDIT REPORT
# ============================================================

audit = []

audit.append(
    "FIGURE 8 AUDIT"
)

audit.append(
    ""
)

audit.append(
    "Mean MGG:"
)

for model in sorted(
    MODELS,
    key=lambda x:
        mgg_mean[x],
):

    audit.append(
        f"{MODEL_DISPLAY[model]:15s} "
        f"{mgg_mean[model]:.4f}"
    )


audit.append(
    ""
)

audit.append(
    "Mean probabilistic shift robustness:"
)

for model in shift_order:

    audit.append(
        f"{MODEL_DISPLAY[model]:15s} "
        f"{SHIFT_MEAN[model]:.4f}"
    )


audit.append(
    ""
)

audit.append(
    "Formal MGG model comparison:"
)

audit.append(
    f"Friedman chi2={mgg_chi2:.6f}, "
    f"df=7, p={mgg_p:.6g}, "
    f"Kendall W={mgg_W:.6f}"
)


if np.isfinite(
    shift_p
):

    audit.append(
        ""
    )

    audit.append(
        "Formal case-level shift-robustness model comparison:"
    )

    audit.append(
        f"Friedman chi2={shift_chi2:.6f}, "
        f"df=7, p={shift_p:.6g}, "
        f"Kendall W={shift_W:.6f}"
    )


audit.append(
    ""
)

audit.append(
    "Pair-matched criterion validity:"
)

audit.append(
    f"rho={PAIR_MATCHED['rho']:.6f}"
)

audit.append(
    f"CI=[{PAIR_MATCHED['ci_low']:.6f}, "
    f"{PAIR_MATCHED['ci_high']:.6f}]"
)

audit.append(
    f"permutation p={PAIR_MATCHED['p_perm']:.6f}"
)

audit.append(
    f"Holm all7={PAIR_MATCHED['p_holm_all7']:.6f}"
)


OUT_AUDIT.write_text(
    "\n".join(
        audit
    ),
    encoding="utf-8",
)


print(
    "\n".join(
        audit
    )
)

print(
    "\nOutputs:"
)

for path in [
    OUT_MGG_STATS,
    OUT_MGG_PAIRWISE,
    OUT_SHIFT_STATS,
    OUT_SHIFT_PAIRWISE,
    OUT_AUDIT,
    OUT_PNG,
    OUT_PDF,
    OUT_TIFF,
]:

    if path.exists():

        print(
            path
        )
