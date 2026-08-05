"""All experiments for the "schizophrenia machines" article.

Figures land in ./outputs:
  fig1_t1_onestep_grid.png      16 one-step samples from the t=1-only model (mean face, xN)
  fig1b_dataset_mean.png        decode of the actual dataset-mean latent, for comparison
  fig2_t09_onestep_grid.png     16 one-step samples from the t=0.9-only model fed PURE noise
  fig3_t09_real_recon.png       t=0.9 model on genuinely 90%-noised real images (in-distribution)
  fig4_full_onestep_t1.png      full model, single step from t=1 (also the mean face)
  fig5_full_euler_grid.png      full model, 64-step Euler samples
  fig6_x0_trajectory.png        full model: implied x0-prediction at each t during sampling
  fig8_real_t09_continue.png    full model: real image noised to t=0.9, sampling continued
  metrics.txt                   diversity numbers quoted in the article
"""
import common

import torch
from diffusers import AutoencoderKLFlux2
from torchvision.utils import make_grid, save_image

from dit import TinyDiT

device = "cuda"
torch.manual_seed(1234)

# ---------- shared assets ----------
stats = torch.load(common.LATENTS_DIR / "stats.pt")
mean_s, std_s = stats["mean"].to(device), stats["std"].to(device)
latents = torch.load(common.LATENTS_DIR / "latents.pt").float()
latents_n = ((latents - stats["mean"]) / stats["std"])  # normalized, cpu

vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()
vae.requires_grad_(False)


def load_model(name):
    path = common.CKPT_DIR / f"dit_{name}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location=device)
    m = TinyDiT(in_channels=ck["config"]["in_channels"], grid=ck["config"]["grid"]).to(device)
    m.load_state_dict(ck["ema"])
    print(f"loaded {name} @ step {ck.get('step', '?')}")
    return m.eval().requires_grad_(False)


@torch.no_grad()
def decode(lat_n):
    """normalized latent -> [0,1] image tensor on cpu"""
    lat = lat_n.to(device) * std_s + mean_s
    imgs = vae.decode(lat.to(torch.bfloat16)).sample.float()
    return (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()


def save_grid(lat_n, path, nrow=4, bs=16):
    imgs = torch.cat([decode(lat_n[i:i + bs]) for i in range(0, len(lat_n), bs)])
    save_image(make_grid(imgs, nrow=nrow, padding=2), common.OUT_DIR / path)
    print("wrote", path)


@torch.no_grad()
def one_step(model, x, t):
    tb = torch.full((len(x),), t, device=device)
    return x - t * model(x, tb)


@torch.no_grad()
def euler(model, x, t_start=1.0, steps=64, snapshot_ts=()):
    """Integrate t_start -> 0. Optionally snapshot the implied x0-prediction at given ts."""
    ts = torch.linspace(t_start, 0.0, steps + 1, device=device)
    snaps, want = {}, sorted(snapshot_ts, reverse=True)
    for i in range(steps):
        t, tn = ts[i].item(), ts[i + 1].item()
        v = model(x, torch.full((len(x),), t, device=device))
        if want and t <= want[0] + 1e-6:
            snaps[want.pop(0)] = x - t * v  # implied x0 = x_t - t*v
        x = x + (tn - t) * v
    return x, snaps


def pairwise_dist(lat_n):
    f = lat_n.flatten(1)
    d = torch.cdist(f, f)
    n = len(f)
    return (d.sum() / (n * (n - 1))).item()


metrics = {}

noise16 = torch.randn(16, latents.shape[1], latents.shape[2], latents.shape[3], device=device)
real_idx = torch.arange(8) * 1000 + 17

# ---------- Exp 1: t=1-only model, one step. 16 different noises -> same mean face ----------
m_t1 = load_model("t1")
if m_t1 is not None:
    x0_t1 = one_step(m_t1, noise16, 1.0)
    save_grid(x0_t1, "fig1_t1_onestep_grid.png")
    metrics["t1_onestep_pairwise_latent_dist"] = pairwise_dist(x0_t1.cpu())
    save_grid(latents_n.mean(0, keepdim=True), "fig1b_dataset_mean.png", nrow=1)

# ---------- Exp 2: t=0.9-only model fed PURE noise (train/test mismatch) ----------
m_t09 = load_model("t09")
if m_t09 is not None:
    x0_t09 = one_step(m_t09, noise16, 0.9)  # same 16 noises as exp 1
    save_grid(x0_t09, "fig2_t09_onestep_grid.png")
    metrics["t09_onestep_pairwise_latent_dist"] = pairwise_dist(x0_t09.cpu())

    # ---------- Exp 3: t=0.9 model on genuinely 90%-noised real images ----------
    x0_real = latents_n[real_idx].to(device)
    eps = torch.randn_like(x0_real)
    xt_real = 0.1 * x0_real + 0.9 * eps
    recon = one_step(m_t09, xt_real, 0.9)
    rows = torch.cat([x0_real, xt_real, recon])  # originals / noised state / one-step recon
    save_grid(rows.cpu(), "fig3_t09_real_recon.png", nrow=8)

# ---------- Exp 4: full model ----------
m_full = load_model("full")
if m_full is not None:
    x0_full1 = one_step(m_full, noise16, 1.0)
    save_grid(x0_full1, "fig4_full_onestep_t1.png")
    metrics["full_onestep_pairwise_latent_dist"] = pairwise_dist(x0_full1.cpu())

    samples, _ = euler(m_full, noise16, steps=64)
    save_grid(samples, "fig5_full_euler_grid.png")
    metrics["full_euler_pairwise_latent_dist"] = pairwise_dist(samples.cpu())
    metrics["real_data_pairwise_latent_dist"] = pairwise_dist(latents_n[:512])

    # x0-prediction trajectory: what the model "believes" the image is, at each t
    snap_ts = [1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2]
    traj_noise = noise16[:6]
    final, snaps = euler(m_full, traj_noise, steps=64, snapshot_ts=snap_ts)
    cols = [snaps[t] for t in snap_ts] + [final]
    strip = torch.stack(cols, dim=1).flatten(0, 1)  # row per seed, col per t
    save_grid(strip.cpu(), "fig6_x0_trajectory.png", nrow=len(cols))

    # real image -> noise to t=0.9 -> continue sampling: pose survives
    x0_real8 = latents_n[real_idx].to(device)
    xt09 = 0.1 * x0_real8 + 0.9 * torch.randn_like(x0_real8)
    cont, _ = euler(m_full, xt09, t_start=0.9, steps=58)
    save_grid(torch.cat([x0_real8, cont]).cpu(), "fig8_real_t09_continue.png", nrow=8)

with open(common.OUT_DIR / "metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")
        print(f"{k}: {v:.4f}")
print("done")
