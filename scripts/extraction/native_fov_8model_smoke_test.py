#!/usr/bin/env python3
"""
PathScaleBench: audited 7-model native-FOV smoke test.

Models
------
Original/reference:
  phikon
  uni

Reviewer-requested additions:
  virchow2
  rudolfv2
  gigapath
  rudolfv2s
  gigapath_flash

Purpose
-------
Validate, on a small deterministic subset of the already visually/audited
native-FOV triplets, that each PFM:

  1) receives exactly the same raw native-FOV tissue regions,
  2) uses its official model-specific preprocessing,
  3) produces the documented tile-level representation,
  4) returns finite embeddings of the expected dimension,
  5) is deterministic/repeatable on a reference tile,
  6) can run within the selected hardware/precision configuration.

Spatial protocol is NOT recomputed here:
  40x   = level 0, read 224 x 224
  10x   = level 1, read 224 x 224
  2p5x  = level 2, read 224 x 224
  alignment = same tissue center

Default case is the previously visually-QC'd BLCA case.
Default n=8 anchors => 24 raw tiles/model.

No quantization is used.
Default precision is FP32 for the smoke test. Optional --amp enables CUDA
autocast without changing stored embeddings (always saved as float32).

Outputs
-------
native_fov/
    smoke_test_7models/<MODEL>/
        <MODEL>_40x.npy
        <MODEL>_10x.npy
        <MODEL>_2p5x.npy
        SMOKE_AUDIT.json

Run models one-by-one first. This avoids one gated/large model blocking all
others and keeps GPU memory controlled.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import openslide
import PIL
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from transformers import AutoImageProcessor, AutoModel
import transformers


REPO_ROOT = Path(__file__).resolve().parents[2]

ROOT = Path(
    os.environ.get(
        "PATHSCALEBENCH_NATIVE_FOV_ROOT",
        REPO_ROOT / "native_fov",
    )
)

MANIFEST_ROOT = Path(
    os.environ.get(
        "PATHSCALEBENCH_MANIFEST_ROOT",
        ROOT / "manifests",
    )
)

WSI_MAP = Path(
    os.environ.get(
        "PATHSCALEBENCH_WSI_MAP",
        REPO_ROOT / "manifests" / "wsi_paths.csv",
    )
)

QC_ROOT = ROOT / "visual_qc"

SMOKE_ROOT = ROOT / "smoke_test_8models"

UNI_CKPT = Path(
    os.environ.get(
        "UNI_CKPT",
        "",
    )
)

DEFAULT_CANCER = "BLCA"
DEFAULT_CASE = (
    "TCGA-CU-A0YN-01Z-00-DX1."
    "6C5EAAAF-8F14-49D3-8FFC-9BDAE56CAFD0"
)

SCALES = ["40x", "10x", "2p5x"]
LEVELS = {"40x": 0, "10x": 1, "2p5x": 2}
EXPECTED_READ = 224
PROTOCOL_ID = "pathscalebench_native_fov_same_center_v1"

MODEL_KEYS = [
    "phikon",
    "uni",
    "virchow",
    "virchow2",
    "rudolfv2",
    "gigapath",
    "rudolfv2s",
    "gigapath_flash",
]

EXPECTED_DIMS = {
    "phikon": 768,
    "uni": 1024,
    "virchow": 2560,
    "virchow2": 2560,
    "rudolfv2": 1536,
    "gigapath": 1536,
    "rudolfv2s": 384,
    "gigapath_flash": 384,
}

DEFAULT_BATCH = {
    "phikon": 16,
    "uni": 8,
    "virchow": 2,
    "virchow2": 2,
    "rudolfv2": 1,
    "gigapath": 1,
    "rudolfv2s": 16,
    "gigapath_flash": 16,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def first_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    raise RuntimeError(
        f"None of {names} found; columns={list(df.columns)}"
    )


def software_info():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "transformers": transformers.__version__,
        "PIL": PIL.__version__,
        "openslide_python": getattr(openslide, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


# ------------------------------------------------------------------
# Spatial data
# ------------------------------------------------------------------

def load_wsi_path(cancer, case_id):
    if not WSI_MAP.exists():
        raise FileNotFoundError(WSI_MAP)

    df = pd.read_csv(WSI_MAP)
    cc = first_col(df, ["cancer", "CANCER", "tumor_type"])
    kc = first_col(df, ["case_id", "case", "slide_id", "wsi_id"])
    pc = first_col(df, ["resolved_wsi_path", "resolved_path", "wsi_path"])

    hit = df[
        (df[cc].astype(str) == cancer)
        & (df[kc].astype(str) == case_id)
    ]

    if len(hit) != 1:
        raise RuntimeError(
            f"Expected one WSI mapping for {cancer}/{case_id}, got {len(hit)}"
        )

    p = Path(str(hit.iloc[0][pc]))
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_native_manifest(cancer, case_id, n):
    mdir = MANIFEST_ROOT / cancer

    geom_path = mdir / f"{case_id}_native_fov_geometry.json"
    trip_path = mdir / f"{case_id}_native_fov_triplets.csv"

    if not geom_path.exists():
        raise FileNotFoundError(geom_path)
    if not trip_path.exists():
        raise FileNotFoundError(trip_path)

    with open(geom_path, "r", encoding="utf-8") as f:
        geom = json.load(f)

    if geom.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(
            f"Wrong protocol: {geom.get('protocol_id')}"
        )

    all_trip = pd.read_csv(trip_path)

    if len(all_trip) != int(geom["native_valid_triplets"]):
        raise RuntimeError("Geometry/triplet count mismatch")

    # Prefer the exact 50-anchor set that was manually visually inspected.
    selected50_path = (
        QC_ROOT / cancer / case_id / "selected_50_triplets.csv"
    )

    if selected50_path.exists():
        selected50 = pd.read_csv(selected50_path)

        rows = set(
            selected50["native_row"].astype(int).tolist()
        )
        subset = all_trip[
            all_trip["native_row"].astype(int).isin(rows)
        ].copy()

        subset = subset.sort_values("native_row").reset_index(drop=True)

        if len(subset) != 50:
            raise RuntimeError(
                f"Visual-QC manifest should resolve to 50 rows, got {len(subset)}"
            )

        # Deterministic even coverage of the 50 visually-QC'd anchors.
        idx = np.linspace(
            0, len(subset) - 1, n, dtype=int
        )
        smoke = subset.iloc[idx].copy().reset_index(drop=True)
        source = str(selected50_path)
    else:
        # Fallback only if QC file is missing.
        idx = np.linspace(
            0, len(all_trip) - 1, n, dtype=int
        )
        smoke = all_trip.iloc[idx].copy().reset_index(drop=True)
        source = str(trip_path)

    if len(smoke) != n:
        raise RuntimeError(f"Expected {n} smoke rows, got {len(smoke)}")

    for scale in SCALES:
        if not np.all(
            smoke[f"{scale}_level"].to_numpy() == LEVELS[scale]
        ):
            raise RuntimeError(f"{scale}: wrong level")

        if not np.all(
            smoke[f"{scale}_read_size"].to_numpy() == EXPECTED_READ
        ):
            raise RuntimeError(f"{scale}: read size is not 224")

    return smoke, geom, geom_path, trip_path, source


def audit_wsi(slide, geom, smoke):
    if slide.level_count < 3:
        raise RuntimeError("WSI has fewer than 3 pyramid levels")

    current_ds = [float(v) for v in slide.level_downsamples]
    manifest_ds = [float(v) for v in geom["slide_level_downsamples"]]

    for level in (0, 1, 2):
        rel = abs(current_ds[level] - manifest_ds[level]) / max(
            abs(manifest_ds[level]), 1e-12
        )
        if rel > 1e-4:
            raise RuntimeError(
                f"WSI pyramid changed at L{level}: relerr={rel:.3e}"
            )

    width0, height0 = map(int, slide.dimensions)

    for scale in SCALES:
        ds = current_ds[LEVELS[scale]]
        fov0 = 224.0 * ds

        x = smoke[f"{scale}_x_level0"].to_numpy(dtype=np.int64)
        y = smoke[f"{scale}_y_level0"].to_numpy(dtype=np.int64)

        bad = (
            (x < 0)
            | (y < 0)
            | (x.astype(float) + fov0 > width0 + 1e-6)
            | (y.astype(float) + fov0 > height0 + 1e-6)
        )

        if bad.any():
            raise RuntimeError(
                f"{scale}: {int(bad.sum())} smoke reads are OOB"
            )


def read_images(slide, rows, scale):
    level = LEVELS[scale]
    images = []

    xcol = f"{scale}_x_level0"
    ycol = f"{scale}_y_level0"

    xs = rows[xcol].to_numpy(dtype=np.int64)
    ys = rows[ycol].to_numpy(dtype=np.int64)

    for x, y in zip(xs, ys):
        x = int(x)
        y = int(y)

        im = slide.read_region(
            (x, y),
            level,
            (224, 224),
        ).convert("RGB")

        if im.mode != "RGB" or im.size != (224, 224):
            raise RuntimeError(
                f"Bad OpenSlide read: {im.mode}, {im.size}"
            )

        images.append(im)

    return images


# ------------------------------------------------------------------
# Model adapters
# ------------------------------------------------------------------

class Adapter:
    def __init__(
        self,
        key,
        model,
        preprocess_kind,
        preprocess,
        feature_fn,
        dim,
        provenance,
        device,
    ):
        self.key = key
        self.model = model
        self.preprocess_kind = preprocess_kind
        self.preprocess = preprocess
        self.feature_fn = feature_fn
        self.dim = dim
        self.provenance = provenance
        self.device = device

    def tensors(self, images):
        if self.preprocess_kind == "hf_processor":
            batch = self.preprocess(
                images=images,
                return_tensors="pt",
            )
            return {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }

        xs = [self.preprocess(im) for im in images]
        x = torch.stack(xs, 0)
        if tuple(x.shape[1:]) != (3, 224, 224):
            raise RuntimeError(
                f"{self.key}: preprocessing output {x.shape}, expected Bx3x224x224"
            )
        if not torch.isfinite(x).all():
            raise RuntimeError(f"{self.key}: preprocessing produced NaN/Inf")
        return {"pixel_values": x.to(self.device, non_blocking=True)}

    def forward(self, batch, amp=False):
        if self.preprocess_kind == "hf_processor":
            call = lambda: self.model(**batch)
        else:
            x = batch["pixel_values"]
            call = lambda: self.model(x)

        ctx = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if amp and self.device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode(), ctx:
            raw = call()
            z = self.feature_fn(raw)

        if not torch.is_tensor(z):
            raise RuntimeError(f"{self.key}: feature_fn did not return Tensor")
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise RuntimeError(
                f"{self.key}: got {tuple(z.shape)}, expected Bx{self.dim}"
            )
        if not torch.isfinite(z).all():
            raise RuntimeError(f"{self.key}: embedding contains NaN/Inf")

        return z.float()


def build_adapter(key, device):
    if key == "phikon":
        repo = "owkin/phikon"
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo)
        model.eval().to(device)

        def feat(out):
            return out.last_hidden_state[:, 0, :]

        return Adapter(
            key, model, "hf_processor", processor, feat, 768,
            {
                "repo": repo,
                "loader": "AutoImageProcessor + AutoModel",
                "feature": "last_hidden_state[:,0,:] (CLS)",
                "expected_dim": 768,
                "preprocessor": repr(processor),
            },
            device,
        )

    if key == "uni":
        if not UNI_CKPT.exists():
            raise FileNotFoundError(UNI_CKPT)

        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        state = torch.load(UNI_CKPT, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        msg = model.load_state_dict(state, strict=True)
        if (
            getattr(msg, "missing_keys", [])
            or getattr(msg, "unexpected_keys", [])
        ):
            raise RuntimeError(f"UNI strict load failed: {msg}")

        model.eval().to(device)

        tfm = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        return Adapter(
            key, model, "tensor_transform", tfm,
            lambda out: out,
            1024,
            {
                "repo": "MahmoodLab/UNI",
                "checkpoint": str(UNI_CKPT),
                "checkpoint_sha256": sha256_file(UNI_CKPT),
                "strict_load": True,
                "feature": "model(image)",
                "expected_dim": 1024,
                "transform": repr(tfm),
            },
            device,
        )

    if key == "virchow":
        repo = "paige-ai/Virchow"

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model.eval().to(device)

        cfg = resolve_data_config(
            model.pretrained_cfg,
            model=model,
        )
        tfm = create_transform(**cfg)

        def feat(out):
            if (
                not torch.is_tensor(out)
                or out.ndim != 3
                or out.shape[1:] != (257, 1280)
            ):
                raise RuntimeError(
                    f"Virchow expected Bx257x1280, "
                    f"got {getattr(out, 'shape', None)}"
                )

            patch = out[:, 1:, :]

            if patch.shape[1] != 256:
                raise RuntimeError(
                    "Virchow patch-token count != 256"
                )

            return torch.cat(
                [
                    out[:, 0, :],
                    patch.mean(dim=1),
                ],
                dim=1,
            )

        return Adapter(
            key,
            model,
            "tensor_transform",
            tfm,
            feat,
            2560,
            {
                "repo": repo,
                "loader": "timm hf-hub",
                "feature": (
                    "concat(CLS token 0, mean patch tokens 1:); "
                    "no register tokens"
                ),
                "expected_dim": 2560,
                "pretrained_cfg": dict(model.pretrained_cfg),
                "resolved_data_config": dict(cfg),
                "transform": repr(tfm),
            },
            device,
        )

    if key == "virchow2":
        repo = "paige-ai/Virchow2"
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model.eval().to(device)

        cfg = resolve_data_config(
            model.pretrained_cfg,
            model=model,
        )
        tfm = create_transform(**cfg)

        def feat(out):
            if (
                not torch.is_tensor(out)
                or out.ndim != 3
                or out.shape[1:] != (261, 1280)
            ):
                raise RuntimeError(
                    f"Virchow2 expected Bx261x1280, got {getattr(out, 'shape', None)}"
                )
            patch = out[:, 5:, :]
            if patch.shape[1] != 256:
                raise RuntimeError("Virchow2 patch-token count != 256")
            return torch.cat(
                [out[:, 0, :], patch.mean(dim=1)],
                dim=1,
            )

        return Adapter(
            key, model, "tensor_transform", tfm, feat, 2560,
            {
                "repo": repo,
                "loader": "timm hf-hub",
                "feature": "concat(CLS token 0, mean patch tokens 5:); registers 1-4 excluded",
                "expected_dim": 2560,
                "pretrained_cfg": dict(model.pretrained_cfg),
                "resolved_data_config": dict(cfg),
                "transform": repr(tfm),
            },
            device,
        )

    if key in ("rudolfv2", "rudolfv2s"):
        repo = (
            "Aignostics/RudolfV-2"
            if key == "rudolfv2"
            else "Aignostics/RudolfV-2-S"
        )
        dim = 1536 if key == "rudolfv2" else 384

        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(
            repo,
            trust_remote_code=True,
        )
        model.eval().to(device)

        def feat(out):
            if not hasattr(out, "pooler_output"):
                raise RuntimeError(
                    f"{key}: output lacks pooler_output"
                )
            return out.pooler_output

        return Adapter(
            key, model, "hf_processor", processor, feat, dim,
            {
                "repo": repo,
                "loader": "AutoImageProcessor + AutoModel(trust_remote_code=True)",
                "feature": "pooler_output (CLS)",
                "expected_dim": dim,
                "preprocessor": repr(processor),
            },
            device,
        )

    if key == "gigapath":
        repo = "prov-gigapath/prov-gigapath"

        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
        )
        model.eval().to(device)

        tfm = transforms.Compose([
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        return Adapter(
            key, model, "tensor_transform", tfm,
            lambda out: out,
            1536,
            {
                "repo": repo,
                "loader": "timm hf_hub tile encoder",
                "feature": "tile_encoder(image)",
                "expected_dim": 1536,
                "transform": repr(tfm),
            },
            device,
        )

    if key == "gigapath_flash":
        repo = "prov-gigapath/prov-gigapath-flash"

        loader = None
        try:
            import gigapath.tile_encoder as tile_encoder
            model = tile_encoder.create_model(
                "hf_hub:prov-gigapath/prov-gigapath-flash"
            )
            loader = "gigapath.tile_encoder.create_model"
        except ImportError as e:
            raise RuntimeError(
                "GigaPath-Flash official README requires the Prov-GigaPath "
                "package to register its tile architecture. Install the official "
                "prov-gigapath package in this environment before smoke testing "
                "gigapath_flash."
            ) from e

        model.eval().to(device)

        tfm = transforms.Compose([
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        return Adapter(
            key, model, "tensor_transform", tfm,
            lambda out: out,
            384,
            {
                "repo": repo,
                "loader": loader,
                "feature": "tile_encoder(image)",
                "expected_dim": 384,
                "transform": repr(tfm),
            },
            device,
        )

    raise ValueError(key)


# ------------------------------------------------------------------
# Smoke run
# ------------------------------------------------------------------

def cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(den, 1e-30))


def run_one_model(
    key,
    cancer,
    case_id,
    n,
    batch_size,
    device,
    amp,
):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    smoke, geom, geom_path, trip_path, selection_source = (
        load_native_manifest(cancer, case_id, n)
    )
    wsi_path = load_wsi_path(cancer, case_id)

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        audit_wsi(slide, geom, smoke)

        print("\n" + "=" * 100)
        print(f"MODEL       : {key}")
        print(f"DEVICE      : {device}")
        print(f"AMP         : {amp}")
        print(f"ANCHORS     : {n}")
        print(f"RAW TILES   : {n * 3}")
        print(f"WSI         : {wsi_path}")
        print("=" * 100)

        t0 = time.perf_counter()
        adapter = build_adapter(key, device)
        load_sec = time.perf_counter() - t0

        if adapter.dim != EXPECTED_DIMS[key]:
            raise RuntimeError(
                f"{key}: adapter dim mismatch {adapter.dim}"
            )

        out_dir = SMOKE_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)

        scale_results = {}
        scale_times = {}

        # Repeatability reference: first smoke tile at 40x, same exact tensor.
        ref_imgs = read_images(slide, smoke.iloc[:1], "40x")
        ref_batch = adapter.tensors(ref_imgs)

        z1 = adapter.forward(ref_batch, amp=amp).detach().cpu().numpy()
        z2 = adapter.forward(ref_batch, amp=amp).detach().cpu().numpy()

        repeat_cos = cosine(z1[0], z2[0])
        repeat_max_abs = float(np.max(np.abs(z1 - z2)))

        if repeat_cos < 0.999999:
            raise RuntimeError(
                f"{key}: repeatability cosine too low: {repeat_cos}"
            )

        del ref_batch, z1, z2, ref_imgs

        for scale in SCALES:
            chunks = []
            t1 = time.perf_counter()

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                imgs = read_images(
                    slide,
                    smoke.iloc[start:end],
                    scale,
                )
                batch = adapter.tensors(imgs)
                z = adapter.forward(
                    batch,
                    amp=amp,
                )

                chunks.append(
                    z.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                del imgs, batch, z

            arr = np.concatenate(chunks, axis=0)
            sec = time.perf_counter() - t1

            if arr.shape != (n, adapter.dim):
                raise RuntimeError(
                    f"{key}/{scale}: {arr.shape} != {(n, adapter.dim)}"
                )
            if arr.dtype != np.float32:
                raise RuntimeError(
                    f"{key}/{scale}: dtype={arr.dtype}"
                )
            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{key}/{scale}: NaN/Inf"
                )

            out_path = out_dir / f"{key}_{scale}.npy"
            np.save(out_path, arr)

            scale_results[scale] = {
                "path": str(out_path),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "mean_l2_norm": float(
                    np.linalg.norm(arr, axis=1).mean()
                ),
            }
            scale_times[scale] = sec

            print(
                f"[PASS] {scale:4s} "
                f"{arr.shape} {arr.dtype} "
                f"time={sec:.2f}s "
                f"mean_norm={scale_results[scale]['mean_l2_norm']:.4f}"
            )

        peak_gb = None
        if device.type == "cuda":
            peak_gb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

        audit = {
            "status": "PASS",
            "created_utc": utc_now(),
            "model_key": key,
            "model_provenance": adapter.provenance,
            "expected_embedding_dim": EXPECTED_DIMS[key],
            "protocol_id": PROTOCOL_ID,
            "cancer": cancer,
            "case_id": case_id,
            "n_anchors": n,
            "n_raw_tiles": n * 3,
            "selection_source": selection_source,
            "native_geometry": str(geom_path),
            "native_geometry_sha256": sha256_file(geom_path),
            "native_triplets": str(trip_path),
            "native_triplets_sha256": sha256_file(trip_path),
            "wsi_path": str(wsi_path),
            "spatial_protocol": {
                "40x": {"level": 0, "read_size": 224},
                "10x": {"level": 1, "read_size": 224},
                "2p5x": {"level": 2, "read_size": 224},
                "alignment": "same_center",
            },
            "device": str(device),
            "amp": bool(amp),
            "batch_size": int(batch_size),
            "model_load_seconds": load_sec,
            "scale_seconds": scale_times,
            "peak_cuda_memory_gb": peak_gb,
            "repeatability": {
                "cosine_similarity": repeat_cos,
                "max_absolute_difference": repeat_max_abs,
            },
            "outputs": scale_results,
            "software": software_info(),
        }

        audit_path = out_dir / "SMOKE_AUDIT.json"
        json_dump(audit, audit_path)

        print(
            f"[PASS] repeatability cosine={repeat_cos:.12f}, "
            f"max_abs={repeat_max_abs:.3e}"
        )
        if peak_gb is not None:
            print(f"[INFO] peak CUDA allocated={peak_gb:.3f} GiB")
        print(f"[PASS] audit -> {audit_path}")

        return audit

    finally:
        slide.close()


def cleanup_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        required=True,
        choices=MODEL_KEYS + ["all"],
    )
    p.add_argument(
        "--cancer",
        default=DEFAULT_CANCER,
    )
    p.add_argument(
        "--case",
        default=DEFAULT_CASE,
    )
    p.add_argument(
        "--n",
        type=int,
        default=8,
        help="Smoke anchors. Default 8 -> 24 raw tiles/model.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override conservative model-specific smoke batch size.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="CUDA fp16 autocast. Default smoke test is FP32.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.n < 1 or args.n > 50:
        raise RuntimeError("--n must be between 1 and 50")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    models = MODEL_KEYS if args.model == "all" else [args.model]

    summary = []

    for key in models:
        bs = args.batch_size or DEFAULT_BATCH[key]

        try:
            audit = run_one_model(
                key=key,
                cancer=args.cancer,
                case_id=args.case,
                n=args.n,
                batch_size=bs,
                device=device,
                amp=args.amp,
            )

            summary.append({
                "model": key,
                "status": "PASS",
                "dim": EXPECTED_DIMS[key],
                "device": str(device),
                "amp": args.amp,
                "batch_size": bs,
                "peak_cuda_memory_gb": audit["peak_cuda_memory_gb"],
            })

        except Exception as e:
            summary.append({
                "model": key,
                "status": "FAIL",
                "dim": EXPECTED_DIMS[key],
                "device": str(device),
                "amp": args.amp,
                "batch_size": bs,
                "error": repr(e),
            })

            print("\n" + "!" * 100)
            print(f"[FAIL] {key}: {type(e).__name__}: {e}")
            print("!" * 100)

            if args.model != "all":
                raise

        finally:
            cleanup_model()

    new_df = pd.DataFrame(summary)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = SMOKE_ROOT / "SMOKE_7MODEL_SUMMARY.csv"

    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        summary_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )
        summary_df = (
            summary_df
            .drop_duplicates(
                subset=["model"],
                keep="last"
            )
            .reset_index(drop=True)
        )
    else:
        summary_df = new_df

    model_order = {
        "phikon": 0,
        "uni": 1,
        "virchow2": 2,
        "rudolfv2": 3,
        "gigapath": 4,
        "rudolfv2s": 5,
        "gigapath_flash": 6,
    }

    summary_df["_order"] = (
        summary_df["model"]
        .map(model_order)
        .fillna(999)
    )

    summary_df = (
        summary_df
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 100)
    print("7-MODEL NATIVE-FOV SMOKE SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("\nsummary:", summary_path)


if __name__ == "__main__":
    main()
