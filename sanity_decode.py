"""Round-trip check: decode a few precomputed latents and save alongside stats."""
import common

import torch
from diffusers import AutoencoderKLFlux2
from torchvision.utils import make_grid, save_image

device = "cuda"
latents = torch.load(common.LATENTS_DIR / "latents.pt").float()
print("latents:", tuple(latents.shape), "dtype fp16 on disk")

vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()
with torch.no_grad():
    imgs = vae.decode(latents[:8].to(device, torch.bfloat16)).sample.float()
imgs = (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()
save_image(make_grid(imgs, nrow=8), common.OUT_DIR / "sanity_roundtrip.png")
print("wrote outputs/sanity_roundtrip.png")
