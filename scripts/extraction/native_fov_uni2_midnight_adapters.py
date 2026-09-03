#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

import torch
import timm
from torchvision import transforms
from transformers import AutoModel

from native_fov_7model_smoke_test import Adapter, sha256_file


# ================================================================
# Model locations on RTX4090 server
# Can be overridden through environment variables.
# ================================================================

HF_HOME = Path(
    os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface"),
    )
).expanduser()

UNI2_CACHE = Path(
    os.environ.get(
        "UNI2_CACHE",
        str(HF_HOME / "hub" / "models--MahmoodLab--UNI2-h"),
    )
).expanduser()

MIDNIGHT_DIR_ENV = os.environ.get("MIDNIGHT_DIR")
MIDNIGHT_DIR = (
    Path(MIDNIGHT_DIR_ENV).expanduser()
    if MIDNIGHT_DIR_ENV
    else None
)


def resolve_hf_snapshot(cache_root: Path) -> Path:
    """
    Resolve:
        models--ORG--MODEL/
            refs/main
            snapshots/<commit>/
    """
    ref = cache_root / "refs" / "main"

    if not ref.exists():
        raise FileNotFoundError(
            f"Hugging Face refs/main missing: {ref}"
        )

    commit = ref.read_text().strip()

    if not commit:
        raise RuntimeError(
            f"Empty Hugging Face refs/main: {ref}"
        )

    snapshot = cache_root / "snapshots" / commit

    if not snapshot.exists():
        raise FileNotFoundError(
            f"Hugging Face snapshot missing: {snapshot}"
        )

    return snapshot


# ================================================================
# UNI2-h
# ================================================================

def build_uni2(device):

    snapshot = resolve_hf_snapshot(UNI2_CACHE)
    ckpt = snapshot / "pytorch_model.bin"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"UNI2-h checkpoint missing: {ckpt}"
        )

    model = timm.create_model(
        "vit_giant_patch14_224",
        pretrained=False,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )

    state = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    msg = model.load_state_dict(
        state,
        strict=True,
    )

    if (
        getattr(msg, "missing_keys", [])
        or getattr(msg, "unexpected_keys", [])
    ):
        raise RuntimeError(
            f"UNI2-h strict load failed: {msg}"
        )

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
        "uni2",
        model,
        "tensor_transform",
        tfm,
        lambda out: out,
        1536,
        {
            "repo": "MahmoodLab/UNI2-h",
            "checkpoint": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "loader": (
                "timm vit_giant_patch14_224; "
                "offline local checkpoint"
            ),
            "strict_load": True,
            "feature": "model(image)",
            "expected_dim": 1536,
            "transform": repr(tfm),
        },
        device,
    )


# ================================================================
# Midnight-12k
# ================================================================

def build_midnight(device):

    if MIDNIGHT_DIR is None or not MIDNIGHT_DIR.exists():
        raise FileNotFoundError(
            "Set MIDNIGHT_DIR to the local Midnight-12k model directory."
        )

    model = AutoModel.from_pretrained(
        str(MIDNIGHT_DIR),
        local_files_only=True,
    )

    hidden = int(
        getattr(model.config, "hidden_size", -1)
    )

    if hidden != 1536:
        raise RuntimeError(
            f"Midnight hidden_size={hidden}, expected 1536"
        )

    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    def feat(out):

        if not hasattr(out, "last_hidden_state"):
            raise RuntimeError(
                "Midnight output lacks last_hidden_state"
            )

        x = out.last_hidden_state

        if x.ndim != 3:
            raise RuntimeError(
                f"Midnight token tensor shape invalid: {tuple(x.shape)}"
            )

        if x.shape[-1] != 1536:
            raise RuntimeError(
                f"Midnight hidden dim={x.shape[-1]}, expected 1536"
            )

        # Representation used in this benchmark:
        # CLS + mean(all patch tokens)
        cls = x[:, 0, :]
        patch = x[:, 1:, :]

        # 224 / 14 = 16 -> 16x16 = 256 patch tokens.
        if patch.shape[1] != 256:
            raise RuntimeError(
                "Midnight expected 256 patch tokens, "
                f"got {patch.shape[1]}"
            )

        z = torch.cat(
            [
                cls,
                patch.mean(dim=1),
            ],
            dim=1,
        )

        if z.shape[1] != 3072:
            raise RuntimeError(
                f"Midnight embedding dim={z.shape[1]}, expected 3072"
            )

        return z

    return Adapter(
        "midnight",
        model,
        "tensor_transform",
        tfm,
        feat,
        3072,
        {
            "repo": "kaiko-ai/midnight",
            "checkpoint_dir": str(MIDNIGHT_DIR),
            "loader": (
                "transformers.AutoModel.from_pretrained("
                "local_files_only=True)"
            ),
            "feature": (
                "concat(CLS, mean(all 256 patch tokens))"
            ),
            "hidden_dim": 1536,
            "expected_dim": 3072,
            "transform": repr(tfm),
        },
        device,
    )


# ================================================================
# Unified interface
# ================================================================

def build_adapter(key, device):

    if key == "uni2":
        return build_uni2(device)

    if key == "midnight":
        return build_midnight(device)

    raise ValueError(
        f"Unsupported UNI2/Midnight model key: {key}"
    )
