"""Two-step sampling t=1.0 -> t=0.9 -> t=0.0, comparing:
  A) the single-t models strung together (t1 model for the first step, t09 for the second)
  B) the full model on the exact same schedule
  C) reference: t09 model fed pure noise directly at t=0.9 (no first step)

Same noise inputs everywhere. Outputs:
  fig9_chain_comparison.png  rows: A / B / C over 8 shared noises
  chain metrics appended to outputs/metrics.txt
"""
import common

import torch
from diffusers import AutoencoderKLFlux2
from torchvision.utils import make_grid, save_image

from dit import TinyDiT

device = "cuda"
torch.manual_seed(1234)

stats = torch.load(common.LATENTS_DIR / "stats.pt")
mean_s, std_s = stats["mean"].to(device), stats["std"].to(device)
latents = torch.load(common.LATENTS_DIR / "latents.pt").float()
latents_n = (latents - stats["mean"]) / stats["std"]

vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()
vae.requires_grad_(False)


def load_model(name):
    ck = torch.load(common.CKPT_DIR / f"dit_{name}.pt", map_location=device)
    m = TinyDiT(in_channels=ck["config"]["in_channels"], grid=ck["config"]["grid"]).to(device)
    m.load_state_dict(ck["ema"])
    return m.eval().requires_grad_(False)


@torch.no_grad()
def decode(lat_n):
    lat = lat_n.to(device) * std_s + mean_s
    imgs = vae.decode(lat.to(torch.bfloat16)).sample.float()
    return (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()


@torch.no_grad()
def vel(model, x, t):
    return model(x, torch.full((len(x),), t, device=device))


def pairwise(f):
    f = f.flatten(1).cpu()
    d = torch.cdist(f, f)
    return (d.sum() / (len(f) * (len(f) - 1))).item()


m_t1, m_t09, m_full = load_model("t1"), load_model("t09"), load_model("full")

C, H, W = latents.shape[1:]
n = 64
x1 = torch.randn(n, C, H, W, device=device)

# A) chained single-t models
x09_chain = x1 - 0.1 * vel(m_t1, x1, 1.0)
x0_chain = x09_chain - 0.9 * vel(m_t09, x09_chain, 0.9)

# B) full model, same 2-step schedule
x09_full = x1 - 0.1 * vel(m_full, x1, 1.0)
x0_full2 = x09_full - 0.9 * vel(m_full, x09_full, 0.9)

# C) reference: t09 fed the raw noise directly
x0_direct = x1 - 0.9 * vel(m_t09, x1, 0.9)

# D) t09 model taking BOTH steps (its own hallucination on pure noise for step 1)
x09_t09 = x1 - 0.1 * vel(m_t09, x1, 0.9)
x0_t09both = x09_t09 - 0.9 * vel(m_t09, x09_t09, 0.9)

rows = torch.cat([x0_chain[:8], x0_full2[:8], x0_direct[:8], x0_t09both[:8]])
imgs = decode(rows)
save_image(make_grid(imgs, nrow=8, padding=2), common.OUT_DIR / "fig9_chain_comparison.png")
print("wrote fig9_chain_comparison.png (rows: chained t1->t09 / full 2-step / t09 direct / t09 both steps)")

lines = [
    f"chain_t1_then_t09_diversity: {pairwise(x0_chain):.4f}",
    f"full_2step_diversity: {pairwise(x0_full2):.4f}",
    f"t09_direct_pure_noise_diversity: {pairwise(x0_direct):.4f}",
    f"t09_two_steps_diversity: {pairwise(x0_t09both):.4f}",
    f"chain_vs_direct_mean_shift: {(x0_chain - x0_direct).flatten(1).norm(dim=1).mean().item():.4f}",
    f"t09_two_steps_vs_direct_mean_shift: {(x0_t09both - x0_direct).flatten(1).norm(dim=1).mean().item():.4f}",
]
with open(common.OUT_DIR / "metrics.txt", "a") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
