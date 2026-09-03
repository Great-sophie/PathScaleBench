#!/usr/bin/env python3
"""
PathScaleBench audited native-FOV production embedding extractor (5 PFMs).

Locked production settings from completed FP32-vs-AMP/batch benchmark:
  phikon          FP32  batch=8    dim=768
  uni             FP32  batch=24   dim=1024
  virchow2        FP32  batch=6    dim=2560
  gigapath        FP32  batch=4    dim=1536
  gigapath_flash  FP32  batch=50   dim=384

Spatial protocol:
  protocol_id = pathscalebench_native_fov_same_center_v1
  40x   = WSI level 0, read 224 x 224
  10x   = WSI level 1, read 224 x 224
  2p5x  = WSI level 2, read 224 x 224
  alignment = same tissue center

Key safeguards:
- coordinates are NEVER recomputed here;
- reads come only from audited native-FOV triplet manifests;
- model loaders/preprocessing/features are imported from the already smoke-tested
  native_fov_7model_smoke_test.py;
- FP32 only; no AMP, no quantization;
- no patch PNGs are written;
- one model loaded at a time;
- per-scale output uses a temporary .npy memmap then atomic rename;
- output audit sidecar stores protocol/model/manifest provenance;
- resume skips only structurally validated PASS outputs;
- any failure aborts the current case/model; no silent row skipping.

Expected location:
  scripts/extraction/

Default outputs:
  native_fov/embeddings/<CANCER>/
    <CASE>_<MODEL>_40x.npy
    <CASE>_<MODEL>_10x.npy
    <CASE>_<MODEL>_2p5x.npy
    <CASE>_<MODEL>_AUDIT.json

Recommended workflow:
  # 1) preflight
  python scripts/extract_native_fov_embeddings_5pfm_audited.py --preflight

  # 2) one-case all-model production pilot
  python scripts/extract_native_fov_embeddings_5pfm_audited.py \
      --cancers BLCA --max-cases 1

  # 3) full run / resume
  python scripts/extract_native_fov_embeddings_5pfm_audited.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import native_fov_8model_smoke_test as smoke
except Exception as e:
    raise RuntimeError(
        "Cannot import scripts/native_fov_7model_smoke_test.py. "
        "Place this extractor in the same scripts directory."
    ) from e


PROTOCOL_ID = "pathscalebench_native_fov_same_center_v1"

MODELS = [
    "virchow",
]

EXPECTED_DIMS = {
    "virchow": 2560,
}

PRODUCTION_BATCH = {
    "virchow": 16,
}

SCALES = ["40x", "10x", "2p5x"]
LEVELS = {"40x": 0, "10x": 1, "2p5x": 2}
READ_SIZE = 224

DEFAULT_CANCER_ORDER = [
    "BLCA", "BRCA", "COAD", "HNSC",
    "KIRC", "LUAD", "STAD", "UCEC",
]

ROOT = smoke.ROOT
MANIFEST_ROOT = smoke.MANIFEST_ROOT
OUT_ROOT = ROOT / "embeddings"
LOG_ROOT = ROOT / "logs"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def local_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def human_bytes(n):
    x = float(n)
    for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(x) < 1024.0 or u == "TiB":
            return f"{x:.2f} {u}"
        x /= 1024.0


def expected_output_bytes(n_rows, dim):
    return int(n_rows) * int(dim) * 4 * 3


def output_paths(cancer, case_id, model):
    d = OUT_ROOT / cancer
    d.mkdir(parents=True, exist_ok=True)
    p = {s: d / f"{case_id}_{model}_{s}.npy" for s in SCALES}
    p["audit"] = d / f"{case_id}_{model}_AUDIT.json"
    return p


def partial_path(final_path):
    return final_path.with_name(final_path.name + ".partial")


def discover_cases(cancers=None, exact_cases=None):
    cancer_order = cancers or [
        c for c in DEFAULT_CANCER_ORDER
        if (MANIFEST_ROOT / c).exists()
    ]
    exact = set(exact_cases or [])
    found = []

    for cancer in cancer_order:
        cdir = MANIFEST_ROOT / cancer
        if not cdir.exists():
            raise FileNotFoundError(cdir)

        suffix = "_native_fov_triplets.csv"
        for trip in sorted(cdir.glob(f"*{suffix}")):
            case_id = trip.name[:-len(suffix)]
            if exact and case_id not in exact:
                continue
            geom = cdir / f"{case_id}_native_fov_geometry.json"
            if not geom.exists():
                raise FileNotFoundError(geom)
            found.append((cancer, case_id, trip, geom))

    if exact:
        seen = {x[1] for x in found}
        missing = sorted(exact - seen)
        if missing:
            raise RuntimeError(
                "Requested cases not found: " + ", ".join(missing)
            )

    if not found:
        raise RuntimeError("No native-FOV cases discovered.")
    return found


def load_full_manifest(cancer, case_id, trip_path, geom_path):
    with geom_path.open("r", encoding="utf-8") as f:
        geom = json.load(f)

    if geom.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(
            f"{cancer}/{case_id}: protocol mismatch "
            f"{geom.get('protocol_id')!r}"
        )

    df = pd.read_csv(trip_path)
    expected_n = int(geom["native_valid_triplets"])
    if len(df) != expected_n:
        raise RuntimeError(
            f"{cancer}/{case_id}: rows {len(df)} != {expected_n}"
        )

    required = ["native_row"]
    for s in SCALES:
        required += [
            f"{s}_level",
            f"{s}_read_size",
            f"{s}_x_level0",
            f"{s}_y_level0",
        ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{cancer}/{case_id}: missing columns {missing}"
        )

    rows = df["native_row"].to_numpy(dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError(
            f"{cancer}/{case_id}: duplicate native_row values"
        )

    for s in SCALES:
        if not np.all(
            df[f"{s}_level"].to_numpy(dtype=np.int64) == LEVELS[s]
        ):
            raise RuntimeError(f"{cancer}/{case_id}/{s}: wrong level")
        if not np.all(
            df[f"{s}_read_size"].to_numpy(dtype=np.int64) == READ_SIZE
        ):
            raise RuntimeError(
                f"{cancer}/{case_id}/{s}: read_size != 224"
            )

    return df, geom


def audit_slide(slide, geom, df, cancer, case_id):
    if slide.level_count < 3:
        raise RuntimeError(
            f"{cancer}/{case_id}: WSI has <3 levels"
        )

    current_ds = [float(x) for x in slide.level_downsamples]
    manifest_ds = [float(x) for x in geom["slide_level_downsamples"]]

    for level in (0, 1, 2):
        rel = abs(current_ds[level] - manifest_ds[level]) / max(
            abs(manifest_ds[level]), 1e-12
        )
        if rel > 1e-4:
            raise RuntimeError(
                f"{cancer}/{case_id}: pyramid changed at L{level}: "
                f"{current_ds[level]} vs {manifest_ds[level]}"
            )

    width0, height0 = map(int, slide.dimensions)
    for s in SCALES:
        fov0 = READ_SIZE * current_ds[LEVELS[s]]
        x = df[f"{s}_x_level0"].to_numpy(dtype=np.int64)
        y = df[f"{s}_y_level0"].to_numpy(dtype=np.int64)
        bad = (
            (x < 0) | (y < 0) |
            (x.astype(float) + fov0 > width0 + 1e-6) |
            (y.astype(float) + fov0 > height0 + 1e-6)
        )
        if bad.any():
            raise RuntimeError(
                f"{cancer}/{case_id}/{s}: "
                f"{int(bad.sum())} OOB reads"
            )


def validate_completed(
    cancer,
    case_id,
    model,
    n_rows,
    trip_sha,
    geom_sha,
    verify_hash=False,
):
    p = output_paths(cancer, case_id, model)
    if not p["audit"].exists():
        return False, "audit_missing"

    try:
        with p["audit"].open("r", encoding="utf-8") as f:
            a = json.load(f)
    except Exception as e:
        return False, f"audit_unreadable:{e!r}"

    checks = [
        a.get("status") == "PASS",
        a.get("protocol_id") == PROTOCOL_ID,
        a.get("model") == model,
        int(a.get("n_rows", -1)) == int(n_rows),
        int(a.get("embedding_dim", -1)) == EXPECTED_DIMS[model],
        a.get("precision") == "fp32",
        int(a.get("batch_size", -1)) == PRODUCTION_BATCH[model],
        a.get("native_triplets_sha256") == trip_sha,
        a.get("native_geometry_sha256") == geom_sha,
    ]
    if not all(checks):
        return False, "audit_metadata_mismatch"

    outputs = a.get("outputs", {})
    for s in SCALES:
        fpath = p[s]
        if not fpath.exists():
            return False, f"{s}_missing"
        try:
            arr = np.load(fpath, mmap_mode="r")
        except Exception as e:
            return False, f"{s}_unreadable:{e!r}"
        if arr.shape != (n_rows, EXPECTED_DIMS[model]):
            return False, f"{s}_shape_mismatch:{arr.shape}"
        if arr.dtype != np.float32:
            return False, f"{s}_dtype_mismatch:{arr.dtype}"

        meta = outputs.get(s, {})
        if int(meta.get("file_size_bytes", -1)) != fpath.stat().st_size:
            return False, f"{s}_file_size_mismatch"

        if verify_hash:
            expected = meta.get("sha256")
            if not expected:
                return False, f"{s}_sha256_missing"
            if sha256_file(fpath) != expected:
                return False, f"{s}_sha256_mismatch"

    return True, "validated_complete"


def remove_partials(p):
    for s in SCALES:
        q = partial_path(p[s])
        if q.exists():
            q.unlink()


def extract_scale(
    slide,
    df,
    scale,
    adapter,
    batch_size,
    out_tmp,
    n_rows,
    dim,
    device,
    progress_every,
):
    mmap = open_memmap(
        out_tmp,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, dim),
    )

    t0 = time.perf_counter()
    read_sec = preprocess_sec = forward_sec = write_sec = 0.0
    next_report = progress_every

    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch_df = df.iloc[start:end]

        x0 = time.perf_counter()
        images = smoke.read_images(slide, batch_df, scale)
        read_sec += time.perf_counter() - x0

        x0 = time.perf_counter()
        batch = adapter.tensors(images)
        sync_cuda(device)
        preprocess_sec += time.perf_counter() - x0

        sync_cuda(device)
        x0 = time.perf_counter()
        z = adapter.forward(batch, amp=False)
        sync_cuda(device)
        forward_sec += time.perf_counter() - x0

        arr = (
            z.detach().cpu().numpy()
            .astype(np.float32, copy=False)
        )
        expected_shape = (end - start, dim)
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"{scale}: {arr.shape} != {expected_shape}"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{scale}: NaN/Inf rows [{start}:{end})"
            )

        x0 = time.perf_counter()
        mmap[start:end] = arr
        write_sec += time.perf_counter() - x0

        del images, batch, z, arr

        if end >= next_report or end == n_rows:
            elapsed = time.perf_counter() - t0
            rate = end / max(elapsed, 1e-9)
            eta = (n_rows - end) / max(rate, 1e-9)
            print(
                f"    {scale:4s} {end:>8,d}/{n_rows:,} "
                f"({100*end/n_rows:6.2f}%) | "
                f"{rate:7.2f} rows/s | ETA {eta/60:7.1f} min",
                flush=True,
            )
            while next_report <= end:
                next_report += progress_every

    mmap.flush()
    del mmap

    total_sec = time.perf_counter() - t0

    # Post-write structural + finite validation before atomic commit.
    check = np.load(out_tmp, mmap_mode="r")
    if check.shape != (n_rows, dim):
        raise RuntimeError(
            f"{scale}: temp shape changed to {check.shape}"
        )
    if check.dtype != np.float32:
        raise RuntimeError(
            f"{scale}: temp dtype changed to {check.dtype}"
        )
    for i in range(0, n_rows, 8192):
        if not np.isfinite(check[i:i+8192]).all():
            raise RuntimeError(
                f"{scale}: temp output contains NaN/Inf"
            )
    del check

    return {
        "total_sec": float(total_sec),
        "read_sec": float(read_sec),
        "preprocess_sec": float(preprocess_sec),
        "forward_sec": float(forward_sec),
        "write_sec": float(write_sec),
        "rows_per_sec_end_to_end": float(
            n_rows / max(total_sec, 1e-9)
        ),
    }


def run_case_model(
    cancer,
    case_id,
    trip_path,
    geom_path,
    model,
    device,
    adapter=None,
    model_load_sec=None,
    force=False,
    verify_existing_hash=False,
    output_hash=True,
    progress_every=5000,
):
    t_case = time.perf_counter()

    df, geom = load_full_manifest(
        cancer, case_id, trip_path, geom_path
    )
    n_rows = len(df)
    dim = EXPECTED_DIMS[model]
    batch_size = PRODUCTION_BATCH[model]
    trip_sha = sha256_file(trip_path)
    geom_sha = sha256_file(geom_path)

    p = output_paths(cancer, case_id, model)

    if not force:
        valid, why = validate_completed(
            cancer,
            case_id,
            model,
            n_rows,
            trip_sha,
            geom_sha,
            verify_hash=verify_existing_hash,
        )
        if valid:
            print(
                f"[SKIP] {cancer}/{case_id} | {model} | {why}",
                flush=True,
            )
            return {
                "cancer": cancer,
                "case_id": case_id,
                "model": model,
                "status": "SKIP_VALIDATED",
                "n_rows": n_rows,
                "embedding_dim": dim,
                "batch_size": batch_size,
                "seconds": 0.0,
                "message": why,
            }

    remove_partials(p)
    if force and p["audit"].exists():
        p["audit"].unlink()

    wsi_path = smoke.load_wsi_path(cancer, case_id)

    print(
        f"\n[CASE]  {cancer}/{case_id}\n"
        f"[MODEL] {model} | FP32 | batch={batch_size} | dim={dim}\n"
        f"[ROWS]  {n_rows:,}\n"
        f"[WSI]   {wsi_path}",
        flush=True,
    )

    owns_adapter = adapter is None
    slide = None

    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        if adapter is None:
            t0 = time.perf_counter()
            adapter = smoke.build_adapter(model, device)
            model_load_sec = time.perf_counter() - t0
        elif model_load_sec is None:
            model_load_sec = 0.0

        if int(adapter.dim) != dim:
            raise RuntimeError(
                f"{model}: adapter.dim={adapter.dim} != {dim}"
            )

        slide = smoke.openslide.OpenSlide(str(wsi_path))
        audit_slide(slide, geom, df, cancer, case_id)

        scale_stats = {}
        outputs = {}

        for s in SCALES:
            final_path = p[s]
            temp_path = partial_path(final_path)
            if temp_path.exists():
                temp_path.unlink()

            print(f"  [EXTRACT] {s} -> {final_path.name}", flush=True)

            stats = extract_scale(
                slide=slide,
                df=df,
                scale=s,
                adapter=adapter,
                batch_size=batch_size,
                out_tmp=temp_path,
                n_rows=n_rows,
                dim=dim,
                device=device,
                progress_every=progress_every,
            )

            os.replace(temp_path, final_path)

            meta = {
                "path": str(final_path),
                "shape": [n_rows, dim],
                "dtype": "float32",
                "file_size_bytes": int(final_path.stat().st_size),
            }
            if output_hash:
                print(f"  [HASH] {s}", flush=True)
                meta["sha256"] = sha256_file(final_path)
            else:
                meta["sha256"] = None

            scale_stats[s] = stats
            outputs[s] = meta

            print(
                f"  [PASS] {s} | "
                f"{stats['rows_per_sec_end_to_end']:.2f} rows/s | "
                f"{stats['total_sec']/60:.1f} min",
                flush=True,
            )

        peak_alloc = peak_reserved = None
        if device.type == "cuda":
            peak_alloc = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved(device) / 1024**3
            )

        total_sec = time.perf_counter() - t_case

        audit = {
            "status": "PASS",
            "created_utc": utc_now(),
            "protocol_id": PROTOCOL_ID,
            "cancer": cancer,
            "case_id": case_id,
            "model": model,
            "precision": "fp32",
            "amp": False,
            "batch_size": batch_size,
            "embedding_dim": dim,
            "n_rows": n_rows,
            "n_scale_forwards": n_rows * 3,
            "spatial_protocol": {
                "40x": {"level": 0, "read_size": 224},
                "10x": {"level": 1, "read_size": 224},
                "2p5x": {"level": 2, "read_size": 224},
                "alignment": "same tissue center",
                "coordinate_source": (
                    "audited native-FOV triplet manifest; not recomputed"
                ),
            },
            "model_provenance": adapter.provenance,
            "model_load_seconds_shared": model_load_sec,
            "model_reused_across_cases": not owns_adapter,
            "native_triplets": str(trip_path),
            "native_triplets_sha256": trip_sha,
            "native_geometry": str(geom_path),
            "native_geometry_sha256": geom_sha,
            "wsi_path": str(wsi_path),
            "wsi_dimensions_level0": list(map(int, slide.dimensions)),
            "wsi_level_downsamples": [
                float(x) for x in slide.level_downsamples
            ],
            "scale_stats": scale_stats,
            "outputs": outputs,
            "peak_cuda_allocated_gb": peak_alloc,
            "peak_cuda_reserved_gb": peak_reserved,
            "total_seconds": total_sec,
            "software": smoke.software_info(),
            "production_lock": {
                "precision": "fp32",
                "batch_size": batch_size,
                "source": (
                    "RTX 4090 FP32 native-FOV throughput benchmark; "
                    "AMP remains disabled because it failed the pre-specified "
                    "conservative max-relative-L2 engineering gate"
                ),
            },
        }

        json_dump(audit, p["audit"])

        valid, why = validate_completed(
            cancer,
            case_id,
            model,
            n_rows,
            trip_sha,
            geom_sha,
            verify_hash=output_hash,
        )
        if not valid:
            raise RuntimeError(
                f"Final resume validation failed: {why}"
            )

        print(
            f"[DONE] {cancer}/{case_id} | {model} | "
            f"{total_sec/60:.1f} min",
            flush=True,
        )

        return {
            "cancer": cancer,
            "case_id": case_id,
            "model": model,
            "status": "PASS",
            "n_rows": n_rows,
            "embedding_dim": dim,
            "batch_size": batch_size,
            "seconds": total_sec,
            "message": "validated_complete",
        }

    except Exception as e:
        remove_partials(p)

        fail_audit = {
            "status": "FAIL",
            "created_utc": utc_now(),
            "protocol_id": PROTOCOL_ID,
            "cancer": cancer,
            "case_id": case_id,
            "model": model,
            "precision": "fp32",
            "batch_size": batch_size,
            "embedding_dim": dim,
            "n_rows": n_rows,
            "native_triplets": str(trip_path),
            "native_triplets_sha256": trip_sha,
            "native_geometry": str(geom_path),
            "native_geometry_sha256": geom_sha,
            "wsi_path": str(wsi_path),
            "error_type": type(e).__name__,
            "error": repr(e),
            "software": smoke.software_info(),
        }
        json_dump(fail_audit, p["audit"])

        print(
            f"[FAIL] {cancer}/{case_id} | {model} | "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        return {
            "cancer": cancer,
            "case_id": case_id,
            "model": model,
            "status": "FAIL",
            "n_rows": n_rows,
            "embedding_dim": dim,
            "batch_size": batch_size,
            "seconds": time.perf_counter() - t_case,
            "message": f"{type(e).__name__}: {e}",
        }

    finally:
        if slide is not None:
            slide.close()
        if owns_adapter and adapter is not None:
            del adapter
        cleanup_cuda()


def preflight(cases, models):
    rows = []
    total_bytes = 0
    total_triplets = 0

    print("=" * 100)
    print("PATHSCALEBENCH 5-PFM NATIVE-FOV PRODUCTION PREFLIGHT")
    print("=" * 100)

    for cancer, case_id, trip, geom_path in cases:
        df, geom = load_full_manifest(
            cancer, case_id, trip, geom_path
        )
        n = len(df)
        total_triplets += n

        wsi_path = smoke.load_wsi_path(cancer, case_id)
        slide = smoke.openslide.OpenSlide(str(wsi_path))
        try:
            audit_slide(slide, geom, df, cancer, case_id)
        finally:
            slide.close()

        b = sum(
            expected_output_bytes(n, EXPECTED_DIMS[m])
            for m in models
        )
        total_bytes += b

        rows.append({
            "cancer": cancer,
            "case_id": case_id,
            "n_triplets": n,
            "n_scale_reads_per_model": n * 3,
            "projected_output_bytes": b,
            "wsi_path": str(wsi_path),
            "status": "PASS",
        })

    usage = shutil.disk_usage(OUT_ROOT.parent)
    free_bytes = int(usage.free)

    print(f"cases                    : {len(cases)}")
    print(f"triplets total           : {total_triplets:,}")
    print(f"models                   : {', '.join(models)}")
    print(f"projected new embeddings : {human_bytes(total_bytes)}")
    print(f"free disk at native_fov  : {human_bytes(free_bytes)}")

    need = int(total_bytes * 1.10)
    if free_bytes < need:
        raise RuntimeError(
            f"Insufficient disk with 10% margin: "
            f"need {human_bytes(need)}, free {human_bytes(free_bytes)}"
        )

    print("preflight                : PASS")
    print("=" * 100)
    return pd.DataFrame(rows)


def write_summary(rows):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / "extraction_virchow_case_summary.csv"
    new = pd.DataFrame(rows)

    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["cancer", "case_id", "model"],
            keep="last",
        )
    else:
        merged = new

    if not merged.empty:
        corder = {c: i for i, c in enumerate(DEFAULT_CANCER_ORDER)}
        morder = {m: i for i, m in enumerate(MODELS)}
        merged["_c"] = merged["cancer"].map(corder).fillna(999)
        merged["_m"] = merged["model"].map(morder).fillna(999)
        merged = (
            merged.sort_values(["_c", "case_id", "_m"])
            .drop(columns=["_c", "_m"])
            .reset_index(drop=True)
        )

    merged.to_csv(path, index=False)
    return path


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
    )
    p.add_argument(
        "--cancers",
        nargs="+",
        choices=DEFAULT_CANCER_ORDER,
        default=None,
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Exact case IDs.",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Run only first N filtered cases; useful for pilot.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    p.add_argument(
        "--preflight",
        action="store_true",
    )
    p.add_argument(
        "--force",
        action="store_true",
    )
    p.add_argument(
        "--verify-existing-hash",
        action="store_true",
        help="Re-hash existing embeddings when deciding whether to skip.",
    )
    p.add_argument(
        "--no-output-hash",
        action="store_true",
        help="Skip SHA256 of newly generated embedding arrays.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=5000,
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.max_cases is not None and args.max_cases < 1:
        raise RuntimeError("--max-cases must be >=1")
    if args.progress_every < 1:
        raise RuntimeError("--progress-every must be >=1")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cases = discover_cases(args.cancers, args.cases)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PATHSCALEBENCH AUDITED NATIVE-FOV PRODUCTION EXTRACTION — 5 PFMs")
    print("=" * 100)
    print("protocol :", PROTOCOL_ID)
    print("device   :", device)
    if device.type == "cuda":
        print("GPU      :", torch.cuda.get_device_name(device))
    print("models   :", ", ".join(args.models))
    print("cases    :", len(cases))
    print("precision: FP32 for all models")
    print(
        "batches  : "
        + ", ".join(
            f"{m}={PRODUCTION_BATCH[m]}"
            for m in args.models
        )
    )
    print("=" * 100)

    pf = preflight(cases, args.models)
    preflight_path = LOG_ROOT / "extraction_virchow_preflight.csv"
    pf.to_csv(preflight_path, index=False)

    if args.preflight:
        print("Preflight-only requested; exiting.")
        print("summary:", preflight_path)
        return

    run_id = local_stamp()
    started = utc_now()
    rows = []

    try:
        for model in args.models:
            print("\n" + "=" * 100)
            print(f"[LOAD ONCE] {model}")
            print("=" * 100, flush=True)

            cleanup_cuda()

            t0 = time.perf_counter()
            adapter = smoke.build_adapter(model, device)
            model_load_sec = time.perf_counter() - t0

            if int(adapter.dim) != EXPECTED_DIMS[model]:
                raise RuntimeError(
                    f"{model}: adapter.dim={adapter.dim} "
                    f"!= {EXPECTED_DIMS[model]}"
                )

            print(
                f"[MODEL READY] {model} | "
                f"load={model_load_sec:.2f}s | "
                f"batch={PRODUCTION_BATCH[model]}",
                flush=True,
            )

            try:
                for cancer, case_id, trip, geom in cases:
                    r = run_case_model(
                        cancer=cancer,
                        case_id=case_id,
                        trip_path=trip,
                        geom_path=geom,
                        model=model,
                        device=device,
                        adapter=adapter,
                        model_load_sec=model_load_sec,
                        force=args.force,
                        verify_existing_hash=args.verify_existing_hash,
                        output_hash=not args.no_output_hash,
                        progress_every=args.progress_every,
                    )

                    rows.append(r)
                    summary_path = write_summary(rows)

                    if r["status"] == "FAIL" and args.fail_fast:
                        raise RuntimeError(
                            f"Fail-fast: {cancer}/{case_id}/{model}"
                        )

            finally:
                del adapter
                cleanup_cuda()

    finally:
        summary_path = write_summary(rows)
        run_audit = {
            "run_id": run_id,
            "started_utc": started,
            "finished_utc": utc_now(),
            "protocol_id": PROTOCOL_ID,
            "models": args.models,
            "n_cases": len(cases),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
            "precision": {m: "fp32" for m in args.models},
            "batch_size": {
                m: PRODUCTION_BATCH[m] for m in args.models
            },
            "force": bool(args.force),
            "verify_existing_hash": bool(args.verify_existing_hash),
            "output_hash": not args.no_output_hash,
            "results": rows,
            "summary_csv": str(summary_path),
            "preflight_csv": str(preflight_path),
            "software": smoke.software_info(),
        }
        run_json = LOG_ROOT / f"extraction_virchow_run_{run_id}.json"
        json_dump(run_audit, run_json)

        print("\n" + "=" * 100)
        print("RUN SUMMARY")
        print("=" * 100)
        if rows:
            sdf = pd.DataFrame(rows)
            print("status counts:", sdf["status"].value_counts().to_dict())
        else:
            print("No extraction attempted.")
        print("case summary:", summary_path)
        print("run audit   :", run_json)


if __name__ == "__main__":
    main()
