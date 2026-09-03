#!/usr/bin/env python3
"""
PathScaleBench 5-PFM RTX 4090 shared-read native-FOV extractor.

Purpose
-------
Reduce repeated OpenSlide decoding while preserving the locked scientific
computation exactly:
  * audited native-FOV coordinates are unchanged;
  * 224x224 reads at WSI levels 0 / 1 / 2 are unchanged;
  * model checkpoints, preprocessors, feature definitions and FP32 are unchanged;
  * per-model microbatch sizes are unchanged;
  * CUDA forwards are sequential (no concurrent CUDA streams).

Optimization
------------
All selected models are resident on the GPU. For each case/scale, one raw RGB
chunk is decoded once from the WSI, then the same PIL images are passed to each
model using its locked microbatch size.

Default test outputs are intentionally isolated from production outputs:
  native_fov/embeddings_sharedread_test/<CANCER>/...

This script imports and reuses the audited helpers from
extract_native_fov_embeddings_5pfm_4090.py and
native_fov_7model_smoke_test.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import extract_native_fov_embeddings_5pfm_4090 as base
import native_fov_7model_smoke_test as smoke
import native_fov_uni2_midnight_adapters as models2



DEFAULT_RAW_CHUNK = 128


# ================================================================
# UNI2-h + Midnight-12k model lock
# ================================================================

base.MODELS = [
    "uni2",
    "midnight",
]

base.EXPECTED_DIMS = {
    "uni2": 1536,
    "midnight": 3072,
}

# Conservative pilot settings.
# Lock final production batch sizes only after RTX4090 smoke/throughput audit.
base.PRODUCTION_BATCH = {
    "uni2": 1,
    "midnight": 1,
}


def write_summary_2pfm(rows):
    base.LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        base.LOG_ROOT
        / "extraction_uni2_midnight_case_summary.csv"
    )

    new = pd.DataFrame(rows)

    if path.exists():
        old = pd.read_csv(path)

        merged = pd.concat(
            [old, new],
            ignore_index=True,
        )

        merged = merged.drop_duplicates(
            subset=[
                "cancer",
                "case_id",
                "model",
            ],
            keep="last",
        )
    else:
        merged = new

    if not merged.empty:

        corder = {
            c: i
            for i, c
            in enumerate(base.DEFAULT_CANCER_ORDER)
        }

        morder = {
            m: i
            for i, m
            in enumerate(base.MODELS)
        }

        merged["_c"] = (
            merged["cancer"]
            .map(corder)
            .fillna(999)
        )

        merged["_m"] = (
            merged["model"]
            .map(morder)
            .fillna(999)
        )

        merged = (
            merged
            .sort_values(
                ["_c", "case_id", "_m"]
            )
            .drop(
                columns=["_c", "_m"]
            )
            .reset_index(drop=True)
        )

    merged.to_csv(
        path,
        index=False,
    )

    return path


# Make all inherited resume/summary logic use the isolated 2-PFM summary.
base.write_summary = write_summary_2pfm


def configure_output_roots(output_root: str | None, log_root: str | None):
    if output_root:
        out = Path(output_root).expanduser()
        if not out.is_absolute():
            out = (base.ROOT / out).resolve()
    else:
        out = base.ROOT / "embeddings_sharedread_test"

    if log_root:
        log = Path(log_root).expanduser()
        if not log.is_absolute():
            log = (base.ROOT / log).resolve()
    else:
        log = base.ROOT / "logs_sharedread_test"

    base.OUT_ROOT = out
    base.LOG_ROOT = log
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    return out, log


def validate_raw_chunk(raw_chunk: int, models):
    if raw_chunk < 1:
        raise RuntimeError("--raw-chunk must be >=1")
    bad = [m for m in models if raw_chunk % base.PRODUCTION_BATCH[m] != 0]
    if bad:
        raise RuntimeError(
            "To preserve baseline microbatch boundaries exactly, --raw-chunk "
            "must be divisible by every selected model batch. Bad models: "
            + ", ".join(f"{m}(bs={base.PRODUCTION_BATCH[m]})" for m in bad)
        )


def validate_temp_array(path: Path, n_rows: int, dim: int, scale: str, model: str):
    arr = np.load(path, mmap_mode="r")
    if arr.shape != (n_rows, dim):
        raise RuntimeError(
            f"{model}/{scale}: temp shape {arr.shape} != {(n_rows, dim)}"
        )
    if arr.dtype != np.float32:
        raise RuntimeError(
            f"{model}/{scale}: temp dtype {arr.dtype} != float32"
        )
    for i in range(0, n_rows, 8192):
        if not np.isfinite(arr[i:i + 8192]).all():
            raise RuntimeError(
                f"{model}/{scale}: temp output contains NaN/Inf"
            )
    del arr


def extract_scale_shared(
    *,
    slide,
    df,
    scale,
    active_models,
    adapters,
    paths,
    n_rows,
    device,
    raw_chunk,
    progress_every,
    output_hash,
):
    """Decode each raw WSI patch once, then run all active models sequentially."""

    mmaps = {}
    temp_paths = {}
    stats = {}
    outputs = {}

    for model in active_models:
        dim = base.EXPECTED_DIMS[model]
        final_path = paths[model][scale]
        temp_path = base.partial_path(final_path)
        if temp_path.exists():
            temp_path.unlink()
        temp_paths[model] = temp_path
        mmaps[model] = open_memmap(
            temp_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, dim),
        )
        stats[model] = {
            "read_sec": 0.0,
            "preprocess_sec": 0.0,
            "forward_sec": 0.0,
            "write_sec": 0.0,
        }

    t_scale = time.perf_counter()
    shared_read_sec = 0.0
    next_report = progress_every

    try:
        for start in range(0, n_rows, raw_chunk):
            end = min(start + raw_chunk, n_rows)
            rows = df.iloc[start:end]

            x0 = time.perf_counter()
            images = smoke.read_images(slide, rows, scale)
            read_dt = time.perf_counter() - x0
            shared_read_sec += read_dt

            # Sequential model execution is deliberate. It changes only I/O
            # scheduling, not the model computation or batch boundaries.
            for model in active_models:
                adapter = adapters[model]
                bs = base.PRODUCTION_BATCH[model]
                dim = base.EXPECTED_DIMS[model]
                st = stats[model]

                # Attribute the one shared read to each model for compatibility
                # with the baseline timing schema; shared_read_sec is also stored
                # explicitly in the returned stats.
                st["read_sec"] += read_dt

                for local_start in range(0, len(images), bs):
                    local_end = min(local_start + bs, len(images))
                    global_start = start + local_start
                    global_end = start + local_end
                    sub_images = images[local_start:local_end]

                    x0 = time.perf_counter()
                    batch = adapter.tensors(sub_images)
                    base.sync_cuda(device)
                    st["preprocess_sec"] += time.perf_counter() - x0

                    # Keep the same synchronization structure used by the
                    # validated baseline extractor.
                    base.sync_cuda(device)
                    x0 = time.perf_counter()
                    z = adapter.forward(batch, amp=False)
                    base.sync_cuda(device)
                    st["forward_sec"] += time.perf_counter() - x0

                    arr = z.detach().cpu().numpy().astype(np.float32, copy=False)
                    expected_shape = (global_end - global_start, dim)
                    if arr.shape != expected_shape:
                        raise RuntimeError(
                            f"{model}/{scale}: {arr.shape} != {expected_shape}"
                        )
                    if not np.isfinite(arr).all():
                        raise RuntimeError(
                            f"{model}/{scale}: NaN/Inf rows "
                            f"[{global_start}:{global_end})"
                        )

                    x0 = time.perf_counter()
                    mmaps[model][global_start:global_end] = arr
                    st["write_sec"] += time.perf_counter() - x0

                    del batch, z, arr, sub_images

            del images

            if end >= next_report or end == n_rows:
                elapsed = time.perf_counter() - t_scale
                rate = end / max(elapsed, 1e-9)
                eta = (n_rows - end) / max(rate, 1e-9)
                print(
                    f"    {scale:4s} {end:>8,d}/{n_rows:,} "
                    f"({100 * end / n_rows:6.2f}%) | "
                    f"shared wall {rate:7.2f} rows/s | ETA {eta / 60:7.1f} min",
                    flush=True,
                )
                while next_report <= end:
                    next_report += progress_every

        shared_scale_wall_sec = time.perf_counter() - t_scale

        # Flush, validate, then atomically commit every active model output.
        for model in active_models:
            mmaps[model].flush()
            del mmaps[model]
            validate_temp_array(
                temp_paths[model],
                n_rows,
                base.EXPECTED_DIMS[model],
                scale,
                model,
            )

        for model in active_models:
            final_path = paths[model][scale]
            os.replace(temp_paths[model], final_path)

            meta = {
                "path": str(final_path),
                "shape": [n_rows, base.EXPECTED_DIMS[model]],
                "dtype": "float32",
                "file_size_bytes": int(final_path.stat().st_size),
            }
            if output_hash:
                print(f"  [HASH] {model}/{scale}", flush=True)
                meta["sha256"] = base.sha256_file(final_path)
            else:
                meta["sha256"] = None
            outputs[model] = meta

        result_stats = {}
        for model in active_models:
            st = stats[model]
            effective_total = (
                st["read_sec"]
                + st["preprocess_sec"]
                + st["forward_sec"]
                + st["write_sec"]
            )
            result_stats[model] = {
                "total_sec": float(effective_total),
                "read_sec": float(st["read_sec"]),
                "shared_read_sec": float(shared_read_sec),
                "preprocess_sec": float(st["preprocess_sec"]),
                "forward_sec": float(st["forward_sec"]),
                "write_sec": float(st["write_sec"]),
                "shared_scale_wall_sec": float(shared_scale_wall_sec),
                "rows_per_sec_end_to_end": float(
                    n_rows / max(effective_total, 1e-9)
                ),
            }

        return result_stats, outputs, shared_scale_wall_sec, shared_read_sec

    except Exception:
        for model, mmap in list(mmaps.items()):
            try:
                mmap.flush()
            except Exception:
                pass
        mmaps.clear()
        for model in active_models:
            q = temp_paths.get(model)
            if q is not None and q.exists():
                try:
                    q.unlink()
                except Exception:
                    pass
        raise


def run_case_shared(
    *,
    cancer,
    case_id,
    trip_path,
    geom_path,
    models,
    adapters,
    model_load_sec,
    device,
    raw_chunk,
    force=False,
    verify_existing_hash=False,
    output_hash=True,
    progress_every=5000,
):
    t_case = time.perf_counter()
    df, geom = base.load_full_manifest(cancer, case_id, trip_path, geom_path)
    n_rows = len(df)
    trip_sha = base.sha256_file(trip_path)
    geom_sha = base.sha256_file(geom_path)
    wsi_path = smoke.load_wsi_path(cancer, case_id)

    rows = []
    active_models = []
    paths = {}

    for model in models:
        p = base.output_paths(cancer, case_id, model)
        paths[model] = p

        if not force:
            valid, why = base.validate_completed(
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
                rows.append({
                    "cancer": cancer,
                    "case_id": case_id,
                    "model": model,
                    "status": "SKIP_VALIDATED",
                    "n_rows": n_rows,
                    "embedding_dim": base.EXPECTED_DIMS[model],
                    "batch_size": base.PRODUCTION_BATCH[model],
                    "seconds": 0.0,
                    "message": why,
                })
                continue

        base.remove_partials(p)
        if force and p["audit"].exists():
            p["audit"].unlink()
        active_models.append(model)

    if not active_models:
        return rows

    print(
        f"\n[CASE SHARED] {cancer}/{case_id}\n"
        f"[ROWS]        {n_rows:,}\n"
        f"[WSI]         {wsi_path}\n"
        f"[MODELS]      {', '.join(active_models)}\n"
        f"[RAW CHUNK]   {raw_chunk}",
        flush=True,
    )

    slide = None
    scale_stats = {m: {} for m in active_models}
    outputs = {m: {} for m in active_models}
    shared_scale_wall = {}
    shared_read = {}

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        slide = smoke.openslide.OpenSlide(str(wsi_path))
        base.audit_slide(slide, geom, df, cancer, case_id)

        for scale in base.SCALES:
            print(
                f"  [EXTRACT SHARED] {scale} | models={len(active_models)}",
                flush=True,
            )
            st_by_model, out_by_model, scale_wall, scale_read = extract_scale_shared(
                slide=slide,
                df=df,
                scale=scale,
                active_models=active_models,
                adapters=adapters,
                paths=paths,
                n_rows=n_rows,
                device=device,
                raw_chunk=raw_chunk,
                progress_every=progress_every,
                output_hash=output_hash,
            )
            shared_scale_wall[scale] = scale_wall
            shared_read[scale] = scale_read
            for model in active_models:
                scale_stats[model][scale] = st_by_model[model]
                outputs[model][scale] = out_by_model[model]
                print(
                    f"  [PASS] {model}/{scale} | "
                    f"effective {st_by_model[model]['rows_per_sec_end_to_end']:.2f} rows/s",
                    flush=True,
                )

        peak_alloc = peak_reserved = None
        if device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

        case_wall_sec = time.perf_counter() - t_case

        for model in active_models:
            effective_model_sec = sum(
                scale_stats[model][s]["total_sec"] for s in base.SCALES
            )
            audit = {
                "status": "PASS",
                "created_utc": base.utc_now(),
                "protocol_id": base.PROTOCOL_ID,
                "cancer": cancer,
                "case_id": case_id,
                "model": model,
                "precision": "fp32",
                "amp": False,
                "batch_size": base.PRODUCTION_BATCH[model],
                "embedding_dim": base.EXPECTED_DIMS[model],
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
                "model_provenance": adapters[model].provenance,
                "model_load_seconds_shared": model_load_sec[model],
                "model_reused_across_cases": True,
                "native_triplets": str(trip_path),
                "native_triplets_sha256": trip_sha,
                "native_geometry": str(geom_path),
                "native_geometry_sha256": geom_sha,
                "wsi_path": str(wsi_path),
                "wsi_dimensions_level0": list(map(int, slide.dimensions)),
                "wsi_level_downsamples": [
                    float(x) for x in slide.level_downsamples
                ],
                "scale_stats": scale_stats[model],
                "outputs": outputs[model],
                "peak_cuda_allocated_gb_shared": peak_alloc,
                "peak_cuda_reserved_gb_shared": peak_reserved,
                "total_seconds": float(effective_model_sec),
                "shared_case_wall_seconds": float(case_wall_sec),
                "shared_execution": {
                    "enabled": True,
                    "raw_chunk_size": int(raw_chunk),
                    "resident_models": list(models),
                    "active_models_for_case": list(active_models),
                    "concurrent_cuda_streams": False,
                    "shared_scale_wall_seconds": shared_scale_wall,
                    "shared_read_seconds": shared_read,
                },
                "software": smoke.software_info(),
                "production_lock": {
                    "precision": "fp32",
                    "batch_size": base.PRODUCTION_BATCH[model],
                    "source": (
                        "RTX 4090 FP32 throughput benchmark; shared-read changes "
                        "only WSI I/O scheduling; model microbatch boundaries, "
                        "preprocessing and forwards are unchanged"
                    ),
                },
            }
            base.json_dump(audit, paths[model]["audit"])

            valid, why = base.validate_completed(
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
                    f"Final resume validation failed for {model}: {why}"
                )

            print(
                f"[DONE] {cancer}/{case_id} | {model} | "
                f"shared_case={case_wall_sec / 60:.2f} min",
                flush=True,
            )
            rows.append({
                "cancer": cancer,
                "case_id": case_id,
                "model": model,
                "status": "PASS",
                "n_rows": n_rows,
                "embedding_dim": base.EXPECTED_DIMS[model],
                "batch_size": base.PRODUCTION_BATCH[model],
                "seconds": case_wall_sec,
                "message": "validated_complete_sharedread",
            })

        return rows

    except Exception as e:
        for model in active_models:
            p = paths[model]
            base.remove_partials(p)
            fail_audit = {
                "status": "FAIL",
                "created_utc": base.utc_now(),
                "protocol_id": base.PROTOCOL_ID,
                "cancer": cancer,
                "case_id": case_id,
                "model": model,
                "precision": "fp32",
                "batch_size": base.PRODUCTION_BATCH[model],
                "embedding_dim": base.EXPECTED_DIMS[model],
                "n_rows": n_rows,
                "native_triplets": str(trip_path),
                "native_triplets_sha256": trip_sha,
                "native_geometry": str(geom_path),
                "native_geometry_sha256": geom_sha,
                "wsi_path": str(wsi_path),
                "shared_execution": True,
                "raw_chunk_size": raw_chunk,
                "error_type": type(e).__name__,
                "error": repr(e),
                "software": smoke.software_info(),
            }
            base.json_dump(fail_audit, p["audit"])
            rows.append({
                "cancer": cancer,
                "case_id": case_id,
                "model": model,
                "status": "FAIL",
                "n_rows": n_rows,
                "embedding_dim": base.EXPECTED_DIMS[model],
                "batch_size": base.PRODUCTION_BATCH[model],
                "seconds": time.perf_counter() - t_case,
                "message": f"{type(e).__name__}: {e}",
            })
        print(
            f"[FAIL] {cancer}/{case_id} shared-read | {type(e).__name__}: {e}",
            flush=True,
        )
        return rows

    finally:
        if slide is not None:
            slide.close()


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--models", nargs="+", choices=base.MODELS, default=base.MODELS)
    p.add_argument(
        "--cancers",
        nargs="+",
        choices=base.DEFAULT_CANCER_ORDER,
        default=None,
    )
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verify-existing-hash", action="store_true")
    p.add_argument("--no-output-hash", action="store_true")
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--raw-chunk", type=int, default=DEFAULT_RAW_CHUNK)
    p.add_argument(
        "--output-root",
        default=None,
        help=(
            "Absolute path or path relative to native_fov/. Default: "
            "native_fov/embeddings_sharedread_test"
        ),
    )
    p.add_argument(
        "--log-root",
        default=None,
        help=(
            "Absolute path or path relative to native_fov/. Default: "
            "native_fov/logs_sharedread_test"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.max_cases is not None and args.max_cases < 1:
        raise RuntimeError("--max-cases must be >=1")
    if args.progress_every < 1:
        raise RuntimeError("--progress-every must be >=1")

    validate_raw_chunk(args.raw_chunk, args.models)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out_root, log_root = configure_output_roots(args.output_root, args.log_root)

    cases = base.discover_cases(args.cancers, args.cases)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]

    print("=" * 100)
    print("PATHSCALEBENCH RTX4090 SHARED-READ NATIVE-FOV EXTRACTION — UNI2-h + Midnight-12k")
    print("=" * 100)
    print("protocol    :", base.PROTOCOL_ID)
    print("device      :", device)
    if device.type == "cuda":
        print("GPU         :", torch.cuda.get_device_name(device))
    print("models      :", ", ".join(args.models))
    print("cases       :", len(cases))
    print("precision   : FP32 for all models")
    print("raw chunk   :", args.raw_chunk)
    print(
        "microbatches: "
        + ", ".join(f"{m}={base.PRODUCTION_BATCH[m]}" for m in args.models)
    )
    print("output root :", out_root)
    print("log root    :", log_root)
    print("=" * 100)

    pf = base.preflight(cases, args.models)
    preflight_path = base.LOG_ROOT / "extraction_uni2_midnight_sharedread_preflight.csv"
    pf.to_csv(preflight_path, index=False)
    if args.preflight:
        print("Preflight-only requested; exiting.")
        print("summary:", preflight_path)
        return

    run_id = base.local_stamp()
    started = base.utc_now()
    rows = []
    adapters = {}
    model_load_sec = {}

    try:
        # Keep all selected models resident. This uses the 48 GiB GPU for model
        # residency while retaining sequential, deterministic FP32 forwards.
        print("\n" + "=" * 100)
        print("[LOAD ALL MODELS ONCE]")
        print("=" * 100, flush=True)
        base.cleanup_cuda()

        for model in args.models:
            t0 = time.perf_counter()
            adapter = models2.build_adapter(model, device)
            dt = time.perf_counter() - t0
            if int(adapter.dim) != base.EXPECTED_DIMS[model]:
                raise RuntimeError(
                    f"{model}: adapter.dim={adapter.dim} != {base.EXPECTED_DIMS[model]}"
                )
            adapters[model] = adapter
            model_load_sec[model] = dt
            print(
                f"[MODEL RESIDENT] {model} | load={dt:.2f}s | "
                f"microbatch={base.PRODUCTION_BATCH[model]}",
                flush=True,
            )

        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            reserv = torch.cuda.memory_reserved(device) / 1024**3
            print(
                f"[RESIDENT MEMORY] allocated={alloc:.2f} GiB | "
                f"reserved={reserv:.2f} GiB",
                flush=True,
            )

        for cancer, case_id, trip, geom in cases:
            case_rows = run_case_shared(
                cancer=cancer,
                case_id=case_id,
                trip_path=trip,
                geom_path=geom,
                models=args.models,
                adapters=adapters,
                model_load_sec=model_load_sec,
                device=device,
                raw_chunk=args.raw_chunk,
                force=args.force,
                verify_existing_hash=args.verify_existing_hash,
                output_hash=not args.no_output_hash,
                progress_every=args.progress_every,
            )
            rows.extend(case_rows)
            summary_path = base.write_summary(rows)

            if args.fail_fast and any(r["status"] == "FAIL" for r in case_rows):
                raise RuntimeError(f"Fail-fast: {cancer}/{case_id}")

    finally:
        adapters.clear()
        base.cleanup_cuda()
        summary_path = base.write_summary(rows)
        run_audit = {
            "run_id": run_id,
            "started_utc": started,
            "finished_utc": base.utc_now(),
            "protocol_id": base.PROTOCOL_ID,
            "models": args.models,
            "n_cases": len(cases),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
            "precision": {m: "fp32" for m in args.models},
            "batch_size": {m: base.PRODUCTION_BATCH[m] for m in args.models},
            "raw_chunk_size": args.raw_chunk,
            "shared_read": True,
            "concurrent_cuda_streams": False,
            "force": bool(args.force),
            "verify_existing_hash": bool(args.verify_existing_hash),
            "output_hash": not args.no_output_hash,
            "output_root": str(base.OUT_ROOT),
            "results": rows,
            "summary_csv": str(summary_path),
            "preflight_csv": str(preflight_path),
            "software": smoke.software_info(),
        }
        run_json = base.LOG_ROOT / f"extraction_uni2_midnight_sharedread_run_{run_id}.json"
        base.json_dump(run_audit, run_json)

        print("\n" + "=" * 100)
        print("SHARED-READ RUN SUMMARY")
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
