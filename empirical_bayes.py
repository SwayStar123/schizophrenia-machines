"""The make-or-break experiment: compare the trained t=0.9 model against the EXACT
optimal denoiser for the finite training set, on the same pure-noise inputs.

For xt = 0.1*x0 + 0.9*eps, the empirical-Bayes optimal prediction is
    E[x0 | xt] = sum_i softmax_i( -||xt - 0.1*x_i||^2 / (2*0.81) ) * x_i
i.e. a posterior-weighted average over all 28k training latents. If the trained
model were only interpolating memorized posterior structure, its one-step outputs
should match this retrieval machine. If its outputs are NOT near any training
image, the diversity is manufactured by the network — generalization/hallucination.

Outputs:
  fig10_empirical_bayes.png  rows: optimal x0-hat / optimal's top-1 training image /
                             trained t09 x0-hat / model output's nearest training image
  numbers appended to outputs/metrics.txt
"""
import common

import torch
from diffusers import AutoencoderKLFlux2
from torchvision.utils import make_grid, save_image

from dit import TinyDiT

device = "cuda"
torch.manual_seed(1234)  # same seed => same noise16 as experiments.py / fig2

stats = torch.load(common.LATENTS_DIR / "stats.pt")
mean_s, std_s = stats["mean"].to(device), stats["std"].to(device)
latents_n = ((torch.load(common.LATENTS_DIR / "latents.pt").float() - stats["mean"]) / stats["std"])

C, H, W = latents_n.shape[1:]
noise = torch.randn(16, C, H, W, device=device)

X = latents_n.flatten(1).to(device)          # [N, d]
N, d = X.shape
Xsq = (X * X).sum(1)                         # [N]
T, S2 = 0.9, 0.81                            # noise scale, its variance

# ---- empirical-Bayes optimal denoiser ----
xt = noise.flatten(1)                        # pure noise presented as "t=0.9 states"
logits = (0.1 * (X @ xt.T) - 0.005 * Xsq[:, None]) / S2   # [N, 16], const in i dropped
w = torch.softmax(logits, dim=0)
x0_opt = (w.T @ X).reshape(-1, C, H, W)

ess = 1.0 / (w * w).sum(0)                   # effective #training images per input
top1 = w.argmax(0)
logit_std = logits.std(0).mean()

# ---- trained t09 model ----
ck = torch.load(common.CKPT_DIR / "dit_t09.pt", map_location=device)
m = TinyDiT(in_channels=C, grid=H).to(device)
m.load_state_dict(ck["ema"])
m.eval().requires_grad_(False)
with torch.no_grad():
    x0_model = noise - T * m(noise, torch.full((16,), T, device=device))

# ---- nearest training image to each output ----
def nn_dist(q):
    dmat = torch.cdist(q.flatten(1), X)      # [16, N]
    val, idx = dmat.min(1)
    return val, idx

d_model, nn_model = nn_dist(x0_model)
d_opt, nn_opt = nn_dist(x0_opt)

# reference scale: nearest-neighbor distance among real training latents
probe = X[torch.randperm(N)[:512]]
dref = torch.cdist(probe, X)
dref.fill_diagonal_(float("inf")) if dref.shape[0] == dref.shape[1] else None
# probe rows are training points, so min over full set is 0 at itself; mask via topk
ref_nn = dref.topk(2, largest=False).values[:, 1]

def pairwise(f):
    f = f.flatten(1)
    dm = torch.cdist(f, f)
    return (dm.sum() / (len(f) * (len(f) - 1))).item()

# ---- figure ----
vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()

@torch.no_grad()
def decode(lat_n):
    lat = lat_n.to(device) * std_s + mean_s
    imgs = vae.decode(lat.to(torch.bfloat16)).sample.float()
    return (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()

k = 8
rows = torch.cat([x0_opt[:k], latents_n[top1[:k].cpu()].to(device),
                  x0_model[:k], latents_n[nn_model[:k].cpu()].to(device)])
save_image(make_grid(decode(rows), nrow=k, padding=2), common.OUT_DIR / "fig10_empirical_bayes.png")
print("wrote fig10_empirical_bayes.png")

lines = [
    f"eb_logit_std_nats: {logit_std.item():.2f}",
    f"eb_effective_num_train_images: mean {ess.mean().item():.1f}, median {ess.median().item():.1f}, max {ess.max().item():.1f}",
    f"eb_top1_weight: mean {w.max(0).values.mean().item():.3f}",
    f"eb_optimal_diversity: {pairwise(x0_opt.cpu()):.2f}",
    f"t09_model_diversity_same_inputs: {pairwise(x0_model.cpu()):.2f}",
    f"dist_optimal_output_to_nearest_train: {d_opt.mean().item():.2f}",
    f"dist_model_output_to_nearest_train: {d_model.mean().item():.2f}",
    f"ref_nn_dist_between_real_train_latents: {ref_nn.mean().item():.2f}",
    f"model_nn_equals_optimal_top1: {(nn_model == top1).float().mean().item():.2f}",
]
with open(common.OUT_DIR / "metrics.txt", "a") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
