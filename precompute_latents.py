"""Encode CelebA-HQ (resized to 128x128) with the FLUX.2 VAE encoder and store all
latents plus per-channel normalization stats. Training then never touches the VAE.
"""
import common

import torch
from datasets import load_dataset
from diffusers import AutoencoderKLFlux2
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

tf = transforms.Compose([
    transforms.Resize(common.IMG_SIZE, antialias=True),
    transforms.CenterCrop(common.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def collate(batch):
    return torch.stack([tf(item["image"].convert("RGB")) for item in batch])


def main():
    device = "cuda"
    dtype = torch.bfloat16

    vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=dtype).to(device).eval()
    vae.requires_grad_(False)

    ds = load_dataset(common.DATASET_REPO, split="train")
    loader = DataLoader(ds, batch_size=64, num_workers=4, collate_fn=collate,
                        persistent_workers=True)

    chunks = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="encoding", mininterval=10):
            imgs = imgs.to(device, dtype)
            posterior = vae.encode(imgs).latent_dist
            lat = posterior.mode()  # deterministic; cleaner for controlled experiments
            chunks.append(lat.float().cpu())

    latents = torch.cat(chunks)
    print("latents:", tuple(latents.shape))

    mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    std = latents.std(dim=(0, 2, 3), keepdim=True)
    print("channel mean range:", mean.min().item(), mean.max().item())
    print("channel std range:", std.min().item(), std.max().item())

    torch.save(latents.to(torch.float16), common.LATENTS_DIR / "latents.pt")
    torch.save({"mean": mean, "std": std}, common.LATENTS_DIR / "stats.pt")
    print("saved to", common.LATENTS_DIR)


if __name__ == "__main__":
    main()
