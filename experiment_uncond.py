"""Probe the noise-UNconditioned model trained on t ∈ {1.0, 0.9}.

It receives no t input, so the only way to know whether an input is pure noise
(marginal std 1.0) or a 90%-noised face (marginal std ~0.906) is the input itself —
in d=8192 the norm separates the two almost perfectly. Does it learn that check?

Probes (one step each, same 16 noises as previous figures where applicable):
  A) pure noise, jump with step 1.0            -> if it inferred "t=1", mean face
  B) real images noised to 0.9, jump with 0.9  -> if it inferred "t=0.9", signal reading
  C) pure noise RESCALED by 0.906 (masquerading as the t=0.9 marginal), jump 0.9
     -> if the sanity check is just a norm detector, hallucinations return
  D) two-step sampling 1.0 -> 0.9 -> 0.0

Outputs: fig11_uncond.png (rows A/B/C/D) + metrics appended.
"""
import common

import torch
from diffusers import AutoencoderKLFlux2
from torchvision.utils import make_grid, save_image

from dit import TinyDiT

device = "cuda"
torch.manual_seed(1234)  # same noise16 as all previous figures

stats = torch.load(common.LATENTS_DIR / "stats.pt")
mean_s, std_s = stats["mean"].to(device), stats["std"].to(device)
latents_n = ((torch.load(common.LATENTS_DIR / "latents.pt").float() - stats["mean"]) / stats["std"])
C, H, W = latents_n.shape[1:]
noise = torch.randn(16, C, H, W, device=device)

ck = torch.load(common.CKPT_DIR / "dit_uncond.pt", map_location=device)
m = TinyDiT(in_channels=C, grid=H).to(device)
m.load_state_dict(ck["ema"])
m.eval().requires_grad_(False)

vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()


@torch.no_grad()
def vel(x):
    return m(x, torch.zeros(len(x), device=device))  # conditioning frozen, as in training


@torch.no_grad()
def decode(lat_n):
    lat = lat_n.to(device) * std_s + mean_s
    imgs = vae.decode(lat.to(torch.bfloat16)).sample.float()
    return (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()


def pairwise(f):
    f = f.flatten(1).cpu()
    d = torch.cdist(f, f)
    return (d.sum() / (len(f) * (len(f) - 1))).item()


mean_lat = latents_n.mean(0).to(device)

def dist_to_mean(x):
    return (x - mean_lat).flatten(1).norm(dim=1).mean().item()


# A) pure noise, presented as t=1
a = noise - 1.0 * vel(noise)

# B) real images noised to 0.9
real_idx = torch.arange(16) * 1000 + 17
x0_real = latents_n[real_idx].to(device)
xt_real = 0.1 * x0_real + 0.9 * torch.randn_like(x0_real)
b = xt_real - 0.9 * vel(xt_real)

# C) pure noise rescaled to masquerade as the t=0.9 marginal
sigma_t09 = (0.81 + 0.01) ** 0.5  # ~0.906
c = (noise * sigma_t09) - 0.9 * vel(noise * sigma_t09)

# D) two-step sampling: 1.0 -> 0.9 -> 0.0
x09 = noise - 0.1 * vel(noise)
d = x09 - 0.9 * vel(x09)

rows = torch.cat([a[:8], b[:8], c[:8], d[:8]])
save_image(make_grid(decode(rows), nrow=8, padding=2), common.OUT_DIR / "fig11_uncond.png")
print("wrote fig11_uncond.png (rows: pure noise / real 0.9-noised / rescaled noise / two-step)")

lines = [
    f"uncond_pure_noise_diversity: {pairwise(a):.2f}  dist_to_mean: {dist_to_mean(a):.2f}",
    f"uncond_real_t09_diversity: {pairwise(b):.2f}  dist_to_mean: {dist_to_mean(b):.2f}",
    f"uncond_rescaled_noise_diversity: {pairwise(c):.2f}  dist_to_mean: {dist_to_mean(c):.2f}",
    f"uncond_two_step_diversity: {pairwise(d):.2f}  dist_to_mean: {dist_to_mean(d):.2f}",
]
with open(common.OUT_DIR / "metrics.txt", "a") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
