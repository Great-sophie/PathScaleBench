#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8-PFM native-FOV same-center full-patch cross-scale CKA.

Purpose
-------
Sensitivity analysis for the revision:
compare the current audited 1024-patch CKA protocol against a
full-patch protocol using ALL row-aligned native-FOV embeddings.

Important
---------
- No subsampling.
- No random patch selection.
- All three scales must have identical row counts per case/model.
- Uses audited native-FOV embeddings.
- Per-case linear CKA, then equal-weight averaging across cases.
- GPU implementation is chunked to avoid loading full centered
  embedding matrices onto GPU simultaneously.

Expected cohort
---------------
71 cases x 8 models x 3 scales = 1704 .npy files.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# CONSTANTS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
NATIVE_ROOT = SCRIPT_PATH.parents[1]

EMBED_ROOT = NATIVE_ROOT / "embeddings"

RESULT_ROOT = (
    NATIVE_ROOT
    / "reviewer_analysis"
    / "results"
)

RESULT_ROOT.mkdir(parents=True, exist_ok=True)

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

SCALE_DISPLAY = {
    "40x": "40×",
    "10x": "10×",
    "2p5x": "2.5×",
}

SCALE_PAIRS = [
    ("40x", "10x"),
    ("40x", "2p5x"),
    ("10x", "2p5x"),
]

EXPECTED_CASES = 71
EXPECTED_FILES = 71 * 8 * 3

SEED = 2026

PREFIX = "01b_cross_scale_cka_8pfm_fullpatch"

OUT_PER_CASE = RESULT_ROOT / f"{PREFIX}_per_case.csv"
OUT_SUMMARY = RESULT_ROOT / f"{PREFIX}_summary.csv"
OUT_BY_CANCER = RESULT_ROOT / f"{PREFIX}_by_cancer.csv"
OUT_PROTOCOL = RESULT_ROOT / f"{PREFIX}_protocol.json"
OUT_COMPARE = RESULT_ROOT / f"{PREFIX}_vs_1024.csv"

OLD_1024_SUMMARY = (
    RESULT_ROOT
    / "01_cross_scale_cka_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")

    return torch.device("cpu")


def scale_pair_display(a: str, b: str) -> str:
    return f"{SCALE_DISPLAY[a]}–{SCALE_DISPLAY[b]}"


def bootstrap_ci(
    values,
    seed=SEED,
    n_boot=2000,
):
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


# ============================================================
# MANIFEST DISCOVERY
# ============================================================

def build_manifest():
    rows = []

    for cancer_dir in sorted(EMBED_ROOT.iterdir()):
        if not cancer_dir.is_dir():
            continue

        cancer = cancer_dir.name

        for model in MODELS:
            suffix = f"_{model}_40x.npy"

            files40 = sorted(
                cancer_dir.glob(
                    f"*_{model}_40x.npy"
                )
            )

            for p40 in files40:
                name = p40.name

                if not name.endswith(suffix):
                    continue

                case_id = name[:-len(suffix)]

                paths = {
                    scale:
                    cancer_dir
                    / f"{case_id}_{model}_{scale}.npy"
                    for scale in SCALES
                }

                missing = [
                    str(p)
                    for p in paths.values()
                    if not p.exists()
                ]

                if missing:
                    raise RuntimeError(
                        f"Incomplete triplet: "
                        f"{cancer}/{case_id}/{model}\n"
                        + "\n".join(missing)
                    )

                rows.append({
                    "cancer": cancer,
                    "case_id": case_id,
                    "model": model,
                    "p40": str(paths["40x"]),
                    "p10": str(paths["10x"]),
                    "p25": str(paths["2p5x"]),
                })

    df = pd.DataFrame(rows)

    expected_triplets = (
        EXPECTED_CASES * len(MODELS)
    )

    if len(df) != expected_triplets:
        raise RuntimeError(
            f"Expected {expected_triplets} "
            f"case-model triplets, got {len(df)}"
        )

    n_files = len(df) * 3

    if n_files != EXPECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_FILES} embeddings, "
            f"got {n_files}"
        )

    counts = (
        df.groupby("model")
        .size()
        .to_dict()
    )

    for model in MODELS:
        if counts.get(model, 0) != EXPECTED_CASES:
            raise RuntimeError(
                f"{model}: expected "
                f"{EXPECTED_CASES} cases, "
                f"got {counts.get(model, 0)}"
            )

    return df.sort_values(
        ["model", "cancer", "case_id"]
    ).reset_index(drop=True)


# ============================================================
# FULL-PATCH CKA
# ============================================================

def open_triplet(row):
    X40 = np.load(
        row.p40,
        mmap_mode="r",
    )
    X10 = np.load(
        row.p10,
        mmap_mode="r",
    )
    X25 = np.load(
        row.p25,
        mmap_mode="r",
    )

    arrays = {
        "40x": X40,
        "10x": X10,
        "2p5x": X25,
    }

    dim_expected = EXPECTED_DIMS[row.model]

    shapes = {
        k: v.shape
        for k, v in arrays.items()
    }

    for scale, X in arrays.items():
        if X.ndim != 2:
            raise RuntimeError(
                f"{row.cancer}/{row.case_id}/"
                f"{row.model}/{scale}: "
                f"ndim={X.ndim}"
            )

        if X.shape[1] != dim_expected:
            raise RuntimeError(
                f"{row.cancer}/{row.case_id}/"
                f"{row.model}/{scale}: "
                f"shape={X.shape}, "
                f"expected dim={dim_expected}"
            )

    n40 = X40.shape[0]
    n10 = X10.shape[0]
    n25 = X25.shape[0]

    # HARD alignment requirement.
    # Unlike the old submission code, NEVER use min(n).
    if not (n40 == n10 == n25):
        raise RuntimeError(
            "ROW ALIGNMENT FAILURE\n"
            f"{row.cancer}/{row.case_id}/{row.model}\n"
            f"40x={n40}, 10x={n10}, 2p5x={n25}\n"
            f"shapes={shapes}"
        )

    if n40 < 2:
        raise RuntimeError(
            f"Too few aligned rows: {n40}"
        )

    return arrays, n40, dim_expected


def compute_means(
    arrays,
    n_rows,
    dim,
    chunk_size,
):
    """
    Compute per-scale feature means in float32,
    matching the old linear CKA convention as closely as possible,
    while streaming from mmap arrays.
    """

    sums = {
        scale: np.zeros(
            dim,
            dtype=np.float64,
        )
        for scale in SCALES
    }

    for start in range(
        0,
        n_rows,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_rows,
        )

        for scale in SCALES:
            x = np.asarray(
                arrays[scale][start:end],
                dtype=np.float32,
            )

            sums[scale] += x.sum(
                axis=0,
                dtype=np.float64,
            )

    means = {
        scale: (
            sums[scale] / float(n_rows)
        ).astype(np.float32)
        for scale in SCALES
    }

    return means


@torch.inference_mode()
def linear_cka_triplet_chunked(
    arrays,
    n_rows,
    dim,
    device,
    chunk_size,
):
    """
    Exact full-row linear CKA using chunked accumulation.

    For centered feature matrices X and Y:

        CKA(X,Y)
        = ||X^T Y||_F^2
          / (||X^T X||_F * ||Y^T Y||_F)

    All rows are used.
    No subsampling.
    """

    means_np = compute_means(
        arrays,
        n_rows,
        dim,
        chunk_size,
    )

    means = {
        scale: torch.from_numpy(
            means_np[scale]
        ).to(
            device=device,
            dtype=torch.float32,
        )
        for scale in SCALES
    }

    # Self cross-products
    G40 = torch.zeros(
        (dim, dim),
        dtype=torch.float32,
        device=device,
    )
    G10 = torch.zeros_like(G40)
    G25 = torch.zeros_like(G40)

    # Cross-scale cross-products
    C40_10 = torch.zeros_like(G40)
    C40_25 = torch.zeros_like(G40)
    C10_25 = torch.zeros_like(G40)

    for start in range(
        0,
        n_rows,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_rows,
        )

        x40 = torch.from_numpy(
            np.asarray(
                arrays["40x"][start:end],
                dtype=np.float32,
            ).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        x10 = torch.from_numpy(
            np.asarray(
                arrays["10x"][start:end],
                dtype=np.float32,
            ).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        x25 = torch.from_numpy(
            np.asarray(
                arrays["2p5x"][start:end],
                dtype=np.float32,
            ).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        x40.sub_(means["40x"])
        x10.sub_(means["10x"])
        x25.sub_(means["2p5x"])

        G40.addmm_(
            x40.T,
            x40,
            beta=1.0,
            alpha=1.0,
        )

        G10.addmm_(
            x10.T,
            x10,
            beta=1.0,
            alpha=1.0,
        )

        G25.addmm_(
            x25.T,
            x25,
            beta=1.0,
            alpha=1.0,
        )

        C40_10.addmm_(
            x40.T,
            x10,
            beta=1.0,
            alpha=1.0,
        )

        C40_25.addmm_(
            x40.T,
            x25,
            beta=1.0,
            alpha=1.0,
        )

        C10_25.addmm_(
            x10.T,
            x25,
            beta=1.0,
            alpha=1.0,
        )

        del x40, x10, x25

    n40 = torch.linalg.matrix_norm(
        G40,
        ord="fro",
    )
    n10 = torch.linalg.matrix_norm(
        G10,
        ord="fro",
    )
    n25 = torch.linalg.matrix_norm(
        G25,
        ord="fro",
    )

    def cka(C, na, nb):
        if (
            na.item() == 0.0
            or nb.item() == 0.0
        ):
            return np.nan

        num = torch.linalg.matrix_norm(
            C,
            ord="fro",
        ).square()

        den = na * nb

        return float(
            (num / den).item()
        )

    out = {
        ("40x", "10x"):
            cka(C40_10, n40, n10),

        ("40x", "2p5x"):
            cka(C40_25, n40, n25),

        ("10x", "2p5x"):
            cka(C10_25, n10, n25),
    }

    del (
        G40,
        G10,
        G25,
        C40_10,
        C40_25,
        C10_25,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out


# ============================================================
# SUMMARIES
# ============================================================

def summarize(per_case, n_boot):
    rows = []

    for model in MODELS:
        model_display = MODEL_DISPLAY[model]

        for a, b in SCALE_PAIRS:
            pair = scale_pair_display(a, b)

            sub = per_case[
                (per_case["model"] == model)
                & (per_case["scale_pair"] == pair)
            ]

            values = sub["cka"].to_numpy(
                dtype=np.float64
            )

            lo, hi = bootstrap_ci(
                values,
                seed=SEED,
                n_boot=n_boot,
            )

            rows.append({
                "model": model,
                "model_display": model_display,
                "scale_pair": pair,
                "n_cases": len(values),
                "mean_cka": float(
                    np.mean(values)
                ),
                "std_cka": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
                "median_cka": float(
                    np.median(values)
                ),
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_n_patches": float(
                    sub["n_patches"].mean()
                ),
            })

    return pd.DataFrame(rows)


def summarize_by_cancer(
    per_case,
    n_boot,
):
    rows = []

    for (
        cancer,
        model,
        pair,
    ), sub in per_case.groupby(
        [
            "cancer",
            "model",
            "scale_pair",
        ],
        observed=True,
    ):
        values = sub["cka"].to_numpy(
            dtype=np.float64
        )

        lo, hi = bootstrap_ci(
            values,
            seed=SEED,
            n_boot=n_boot,
        )

        rows.append({
            "cancer": cancer,
            "model": model,
            "model_display":
                MODEL_DISPLAY[model],
            "scale_pair": pair,
            "n_cases": len(values),
            "mean_cka":
                float(np.mean(values)),
            "std_cka":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ) if len(values) > 1
                else np.nan,
            "median_cka":
                float(np.median(values)),
            "ci95_low": lo,
            "ci95_high": hi,
            "mean_n_patches":
                float(
                    sub["n_patches"].mean()
                ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 1024 vs FULL COMPARISON
# ============================================================

def make_comparison(full_summary):
    if not OLD_1024_SUMMARY.exists():
        print(
            "[INFO] 1024-patch summary not found; "
            "comparison skipped."
        )
        return None

    old = pd.read_csv(
        OLD_1024_SUMMARY
    )

    required = {
        "model_display",
        "scale_pair",
        "mean_cka",
    }

    if not required.issubset(
        old.columns
    ):
        print(
            "[INFO] Existing 1024 summary columns "
            "do not match expected schema; "
            "comparison skipped."
        )
        return None

    old = old[
        [
            "model_display",
            "scale_pair",
            "mean_cka",
        ]
    ].rename(
        columns={
            "mean_cka":
                "cka_1024"
        }
    )

    full = full_summary[
        [
            "model_display",
            "scale_pair",
            "mean_cka",
        ]
    ].rename(
        columns={
            "mean_cka":
                "cka_fullpatch"
        }
    )

    comp = old.merge(
        full,
        on=[
            "model_display",
            "scale_pair",
        ],
        how="inner",
    )

    comp["delta_full_minus_1024"] = (
        comp["cka_fullpatch"]
        - comp["cka_1024"]
    )

    comp["abs_delta"] = (
        comp[
            "delta_full_minus_1024"
        ].abs()
    )

    comp.to_csv(
        OUT_COMPARE,
        index=False,
    )

    return comp


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help=(
            "Rows processed per GPU chunk. "
            "This is NOT subsampling; all rows "
            "are still used."
        ),
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--models",
        nargs="*",
        choices=MODELS,
        default=None,
    )

    return p.parse_args()


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    selected_models = (
        args.models
        if args.models
        else MODELS
    )

    print("=" * 88)
    print(
        "8-PFM NATIVE-FOV SAME-CENTER "
        "FULL-PATCH CROSS-SCALE CKA"
    )
    print("=" * 88)

    print(
        f"Embedding root : {EMBED_ROOT}"
    )
    print(
        f"Device         : {device}"
    )

    if device.type == "cuda":
        print(
            "GPU            : "
            + torch.cuda.get_device_name(0)
        )

    print(
        f"Chunk size     : {args.chunk_size}"
    )
    print(
        "Patch sampling : NONE "
        "(all aligned rows)"
    )
    print(
        f"Seed           : {SEED} "
        "(bootstrap only)"
    )
    print(
        f"Models         : "
        f"{', '.join(selected_models)}"
    )
    print("=" * 88)

    manifest = build_manifest()

    manifest = manifest[
        manifest["model"].isin(
            selected_models
        )
    ].reset_index(drop=True)

    print(
        f"Case-model triplets: "
        f"{len(manifest)}"
    )

    records = []

    t_run = time.perf_counter()

    for idx, row in enumerate(
        manifest.itertuples(
            index=False
        ),
        start=1,
    ):
        t0 = time.perf_counter()

        arrays, n_rows, dim = open_triplet(
            row
        )

        print(
            f"\n[{idx:>3d}/{len(manifest)}] "
            f"{row.cancer}/"
            f"{row.case_id} | "
            f"{MODEL_DISPLAY[row.model]} | "
            f"rows={n_rows:,} | "
            f"dim={dim}"
        )

        values = linear_cka_triplet_chunked(
            arrays=arrays,
            n_rows=n_rows,
            dim=dim,
            device=device,
            chunk_size=args.chunk_size,
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        for a, b in SCALE_PAIRS:
            cka_value = values[(a, b)]

            if not np.isfinite(
                cka_value
            ):
                raise RuntimeError(
                    f"Non-finite CKA: "
                    f"{row.cancer}/"
                    f"{row.case_id}/"
                    f"{row.model}/"
                    f"{a}-{b}"
                )

            records.append({
                "cancer": row.cancer,
                "case_id": row.case_id,
                "model": row.model,
                "model_display":
                    MODEL_DISPLAY[
                        row.model
                    ],
                "scale_a": a,
                "scale_b": b,
                "scale_pair":
                    scale_pair_display(
                        a,
                        b,
                    ),
                "n_patches": n_rows,
                "cka": cka_value,
            })

        print(
            "    "
            + " | ".join(
                [
                    (
                        f"{scale_pair_display(a,b)}="
                        f"{values[(a,b)]:.4f}"
                    )
                    for a, b
                    in SCALE_PAIRS
                ]
            )
            + f" | {elapsed:.1f}s"
        )

    per_case = pd.DataFrame(
        records
    )

    per_case.to_csv(
        OUT_PER_CASE,
        index=False,
    )

    summary = summarize(
        per_case,
        n_boot=args.bootstrap,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    by_cancer = summarize_by_cancer(
        per_case,
        n_boot=args.bootstrap,
    )

    by_cancer.to_csv(
        OUT_BY_CANCER,
        index=False,
    )

    comparison = make_comparison(
        summary
    )

    runtime = (
        time.perf_counter()
        - t_run
    )

    protocol = {
        "protocol":
            "native_fov_same_center_fullpatch_cka",
        "embedding_root":
            str(EMBED_ROOT),
        "n_expected_cases":
            EXPECTED_CASES,
        "models":
            selected_models,
        "scales":
            SCALES,
        "scale_pairs":
            [
                list(x)
                for x
                in SCALE_PAIRS
            ],
        "sampling":
            "none_all_rows",
        "row_alignment":
            (
                "hard requirement: "
                "40x == 10x == 2p5x "
                "row counts per case/model"
            ),
        "cka":
            (
                "linear centered CKA; "
                "full aligned rows; "
                "chunked matrix accumulation"
            ),
        "chunk_size":
            args.chunk_size,
        "device":
            str(device),
        "bootstrap_replicates":
            args.bootstrap,
        "seed":
            SEED,
        "runtime_sec":
            runtime,
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
    print(
        "FULL-PATCH CKA COMPLETE"
    )
    print("=" * 88)

    show_cols = [
        "model_display",
        "scale_pair",
        "n_cases",
        "mean_cka",
        "ci95_low",
        "ci95_high",
        "mean_n_patches",
    ]

    print(
        summary[
            show_cols
        ].round(4).to_string(
            index=False
        )
    )

    print(
        f"\nRuntime: "
        f"{runtime / 60:.1f} min"
    )

    print("\nOutputs:")
    print(OUT_PER_CASE)
    print(OUT_SUMMARY)
    print(OUT_BY_CANCER)
    print(OUT_PROTOCOL)

    if comparison is not None:
        print(OUT_COMPARE)

        print(
            "\n"
            + "=" * 88
        )
        print(
            "FULL-PATCH vs 1024-PATCH"
        )
        print(
            "=" * 88
        )

        print(
            comparison.sort_values(
                "abs_delta",
                ascending=False,
            ).round(4).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
