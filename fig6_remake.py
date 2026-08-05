"""Remake fig6: two seeds (rows 2 and 5 of the old figure), columns interleaving
the model's actual input state with its implied x0-prediction:

  x̂0@t=1.0 | state@0.9 | x̂0@0.9 | state@0.8 | x̂0@0.8 | ... | state@0.2 | x̂0@0.2 | final
"""
import common

import torch
from diffusers import AutoencoderKLFlux2
from torchvision.utils import make_grid, save_image

from dit import TinyDiT

device = "cuda"
torch.manual_seed(1234)  # same noise16 as every other figure

stats = torch.load(common.LATENTS_DIR / "stats.pt")
mean_s, std_s = stats["mean"].to(device), stats["std"].to(device)
latents = torch.load(common.LATENTS_DIR / "latents.pt")
C, H, W = latents.shape[1:]

noise16 = torch.randn(16, C, H, W, device=device)
x = noise16[[1, 4]].clone()  # 2nd and 2nd-last rows of the old 6-row figure

ck = torch.load(common.CKPT_DIR / "dit_full.pt", map_location=device)
m = TinyDiT(in_channels=C, grid=H).to(device)
m.load_state_dict(ck["ema"])
m.eval().requires_grad_(False)

vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()


@torch.no_grad()
def decode(lat_n):
    lat = lat_n.to(device) * std_s + mean_s
    imgs = vae.decode(lat.to(torch.bfloat16)).sample.float()
    return (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()


snap_ts = [0.9, 0.8, 0.6, 0.4, 0.2]
steps = 64
ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
want = sorted(snap_ts, reverse=True)
cols = []  # list of [2, C, H, W] latents in display order

with torch.no_grad():
    for i in range(steps):
        t = ts[i].item()
        v = m(x, torch.full((len(x),), t, device=device))
        if i == 0:
            cols.append(x - t * v)  # x̂0 at t=1.0
        elif want and t <= want[0] + 1e-6:
            cols.append(x.clone())      # the state the model actually receives
            cols.append(x - t * v)      # what it believes is hiding in it
            want.pop(0)
        x = x + (ts[i + 1] - ts[i]) * v
cols.append(x)  # final sample

grid = torch.stack(cols, dim=1).flatten(0, 1)  # row-major: seed rows, interleaved cols
imgs = torch.cat([decode(grid[i:i + 8]) for i in range(0, len(grid), 8)])
save_image(make_grid(imgs, nrow=len(cols), padding=2), common.OUT_DIR / "fig6_x0_trajectory.png")
print(f"wrote fig6_x0_trajectory.png ({len(cols)} cols x 2 rows)")
