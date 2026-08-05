"""Download the FLUX.2 VAE and CelebA-HQ 256 dataset into the repo-local HF cache."""
import common  # noqa: F401  (must be first: sets HF_HOME to ./hf_cache)

from huggingface_hub import snapshot_download
from datasets import load_dataset

print("Downloading VAE:", common.VAE_REPO)
snapshot_download(
    common.VAE_REPO,
    allow_patterns=["config.json", "diffusion_pytorch_model.safetensors", "README.md"],
)
print("VAE done.")

print("Downloading dataset:", common.DATASET_REPO)
ds = load_dataset(common.DATASET_REPO, split="train")
print("Dataset done:", ds)
print("Features:", ds.features)
