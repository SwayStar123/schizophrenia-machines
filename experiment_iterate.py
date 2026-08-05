"""How many times can we run the t09 model on its own output, and where does it end up?

Iterate small Euler steps at fixed t=0.9:  x <- x - 0.1*v(x)
(each step plants ~11% of the current prediction back into the state)

Outputs:
  fig12_iterate_A.png  rows: 4 seeds, cols: implied x_hat0 at n = 1,2,3,5,10,20,50,100
  convergence + diversity + nearest-train stats printed
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
latents_n = ((torch.load(common.LATENTS_DIR / "latents.pt").float() - stats["mean"]) / stats["std"])
C, H, W = latents_n.shape[1:]
X = latents_n.flatten(1).to(device)

noise = torch.randn(64, C, H, W, device=device)
show = [1, 4, 0, 2]  # rows for the figure
SNAPS = [1, 2, 3, 5, 10, 20, 50, 100]
N_ITER = 100

ck = torch.load(common.CKPT_DIR / "dit_t09.pt", map_location=device)
m = TinyDiT(in_channels=C, grid=H).to(device)
m.load_state_dict(ck["ema"])
m.eval().requires_grad_(False)

vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()


@torch.no_grad()
def vel(x):
    return m(x, torch.full((len(x),), 0.9, device=device))


@torch.no_grad()
def decode(lat_n):
    lat = lat_n.to(device) * std_s + mean_s
    imgs = vae.decode(lat.to(torch.bfloat16)).sample.float()
    return (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()


def pairwise(f):
    f = f.flatten(1).cpu()
    d = torch.cdist(f, f)
    return (d.sum() / (len(f) * (len(f) - 1))).item()


def nn_dist(q):
    return torch.cdist(q.flatten(1), X).min(1).values.mean().item()


x = noise.clone()
snaps = {}
deltas = []
with torch.no_grad():
    for n in range(1, N_ITER + 1):
        x_new = x - 0.1 * vel(x)
        deltas.append((x_new - x).flatten(1).norm(dim=1).mean().item())
        x = x_new
        if n in SNAPS:
            snaps[n] = (x.clone(), x - 0.9 * vel(x))

cols = [snaps[n][1] for n in SNAPS]
grid = torch.stack([c[show] for c in cols], dim=1).flatten(0, 1)
imgs = torch.cat([decode(grid[i:i + 8]) for i in range(0, len(grid), 8)])
save_image(make_grid(imgs, nrow=len(SNAPS), padding=2), common.OUT_DIR / "fig12_iterate_A.png")

final_belief = snaps[N_ITER][1]
print("step deltas n=1,2,3,5,10,20,50,100: " + ", ".join(f"{deltas[n-1]:.1f}" for n in SNAPS))
print(f"state norm (per-dim std) at n=100: {snaps[N_ITER][0].std().item():.3f}")
print(f"diversity of prediction at n=1: {pairwise(noise - 0.9*vel(noise)):.1f}, "
      f"at n=100: {pairwise(final_belief):.1f}")
print(f"nearest-train dist of prediction at n=100: {nn_dist(final_belief):.1f} "
      f"(one-step: ~69, optimal denoiser: ~33, real NN: ~88)")
print("wrote fig12_iterate_A.png")
