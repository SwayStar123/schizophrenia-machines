"""Shared config. Import this BEFORE any huggingface/diffusers/datasets import:
it redirects every HF cache into ./hf_cache so the whole project is self-contained
and can be deleted in one go.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "hf_cache"
HF_CACHE.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # Windows without dev mode can't symlink
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

VAE_REPO = "black-forest-labs/FLUX.2-small-decoder"  # ungated FLUX.2 encoder + small decoder
DATASET_REPO = "korexyz/celeba-hq-256x256"

IMG_SIZE = 128
LATENTS_DIR = ROOT / "latents"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"
for d in (LATENTS_DIR, CKPT_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)
