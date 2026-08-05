"""One extra number: t09 model's output diversity on REAL 90%-noised images vs on
pure noise, plus mean distance from the dataset-mean latent in both cases.
"""
import common

import torch

from dit import TinyDiT

device = "cuda"
torch.manual_seed(7)

stats = torch.load(common.LATENTS_DIR / "stats.pt")
latents = torch.load(common.LATENTS_DIR / "latents.pt").float()
latents_n = (latents - stats["mean"]) / stats["std"]

ck = torch.load(common.CKPT_DIR / "dit_t09.pt", map_location=device)
m = TinyDiT(in_channels=ck["config"]["in_channels"], grid=ck["config"]["grid"]).to(device)
m.load_state_dict(ck["ema"])
m.eval().requires_grad_(False)


@torch.no_grad()
def one_step(x, t):
    return x - t * m(x, torch.full((len(x),), t, device=device))


def pairwise(f):
    f = f.flatten(1)
    d = torch.cdist(f, f)
    return (d.sum() / (len(f) * (len(f) - 1))).item()


n = 64
x0 = latents_n[torch.randperm(len(latents_n))[:n]].to(device)
real_recon = one_step(0.1 * x0 + 0.9 * torch.randn_like(x0), 0.9)
noise_recon = one_step(torch.randn_like(x0), 0.9)
mean_lat = latents_n.mean(0).to(device)

print(f"t09 on real-noised inputs : diversity {pairwise(real_recon.cpu()):.2f}, "
      f"dist-to-mean {(real_recon - mean_lat).flatten(1).norm(dim=1).mean():.2f}")
print(f"t09 on pure noise         : diversity {pairwise(noise_recon.cpu()):.2f}, "
      f"dist-to-mean {(noise_recon - mean_lat).flatten(1).norm(dim=1).mean():.2f}")
