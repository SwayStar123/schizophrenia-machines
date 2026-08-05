"""Recompute every diversity/distance number in the article at n=256 with error bars.

Diversity = mean pairwise L2 in normalized latent space. Point estimate uses all
pairs of 256 samples; the CI half-width is 1.96 * SEM over 16 disjoint 16-sample
groups (independent estimates of the same quantity, matching the figures' n=16).
Distances-to-mean get plain 1.96*SD/sqrt(n) CIs.
"""
import common

import torch

from dit import TinyDiT

device = "cuda"
torch.manual_seed(1234)

stats = torch.load(common.LATENTS_DIR / "stats.pt")
latents_n = ((torch.load(common.LATENTS_DIR / "latents.pt").float() - stats["mean"]) / stats["std"])
C, H, W = latents_n.shape[1:]
N = 256
noise = torch.randn(N, C, H, W, device=device)
mean_lat = latents_n.mean(0).to(device)


def load(name):
    ck = torch.load(common.CKPT_DIR / f"dit_{name}.pt", map_location=device)
    m = TinyDiT(in_channels=C, grid=H).to(device)
    m.load_state_dict(ck["ema"])
    return m.eval().requires_grad_(False)


m_t1, m_t09, m_full, m_unc = load("t1"), load("t09"), load("full"), load("uncond")


@torch.no_grad()
def vel(m, x, t):
    return m(x, torch.full((len(x),), t, device=device))


@torch.no_grad()
def euler(m, x, t_start=1.0, steps=64):
    ts = torch.linspace(t_start, 0.0, steps + 1, device=device)
    for i in range(steps):
        x = x + (ts[i + 1] - ts[i]) * vel(m, x, ts[i].item())
    return x


def diversity_ci(x):
    f = x.flatten(1).float().cpu()
    dm = torch.cdist(f, f)
    n = len(f)
    point = (dm.sum() / (n * (n - 1))).item()
    groups = []
    for g in range(n // 16):
        sub = dm[g * 16:(g + 1) * 16, g * 16:(g + 1) * 16]
        groups.append((sub.sum() / (16 * 15)).item())
    gt = torch.tensor(groups)
    half = 1.96 * gt.std().item() / (len(groups) ** 0.5)
    return point, half


def dmean_ci(x):
    d = (x - mean_lat).flatten(1).norm(dim=1).cpu()
    return d.mean().item(), 1.96 * d.std().item() / (len(d) ** 0.5)


results = {}

# table 1
results["t1 one-step, pure noise"] = diversity_ci(vel(m_t1, noise, 1.0).mul(-1.0).add(noise))
results["t09 one-step, pure noise"] = diversity_ci(noise - 0.9 * vel(m_t09, noise, 0.9))
results["full, 64-step Euler"] = diversity_ci(euler(m_full, noise.clone()))
ridx = torch.randperm(len(latents_n))[:N]
real = latents_n[ridx].to(device)
results["real data"] = diversity_ci(real)

# table 2 (t09 on real-noised vs pure noise, + dist to mean)
xt_real = 0.1 * real + 0.9 * torch.randn_like(real)
t09_real = xt_real - 0.9 * vel(m_t09, xt_real, 0.9)
t09_pure = noise - 0.9 * vel(m_t09, noise, 0.9)
results["t09 on real-noised"] = diversity_ci(t09_real)
results["t09 on real-noised dist-to-mean"] = dmean_ci(t09_real)
results["t09 on pure noise dist-to-mean"] = dmean_ci(t09_pure)

# table 3 (two-step samplers)
x09_a = noise - 0.1 * vel(m_t1, noise, 1.0)
results["chained t1->t09"] = diversity_ci(x09_a - 0.9 * vel(m_t09, x09_a, 0.9))
x09_b = noise - 0.1 * vel(m_full, noise, 1.0)
results["full 2-step"] = diversity_ci(x09_b - 0.9 * vel(m_full, x09_b, 0.9))
results["t09 direct"] = diversity_ci(t09_pure)
x09_c = noise - 0.1 * vel(m_t09, noise, 0.9)
results["t09 both steps"] = diversity_ci(x09_c - 0.9 * vel(m_t09, x09_c, 0.9))

# table 4 (uncond probes)
u_pure = noise - 1.0 * vel(m_unc, noise, 0.0)
u_real = xt_real - 0.9 * vel(m_unc, xt_real, 0.0)
scaled = noise * (0.82 ** 0.5)
u_spoof = scaled - 0.9 * vel(m_unc, scaled, 0.0)
u09 = noise - 0.1 * vel(m_unc, noise, 0.0)
u_2step = u09 - 0.9 * vel(m_unc, u09, 0.0)
for k, x in [("uncond pure", u_pure), ("uncond real-noised", u_real),
             ("uncond spoof x0.906", u_spoof), ("uncond 2-step", u_2step)]:
    results[k] = diversity_ci(x)
    results[k + " dist-to-mean"] = dmean_ci(x)

# nearest-neighbor yardstick (for the empirical-Bayes section)
X = latents_n.flatten(1).to(device)
probe = X[ridx]
dref = torch.cdist(probe, X)
ref_nn = dref.topk(2, largest=False).values[:, 1]
results["real NN dist (yardstick)"] = (ref_nn.mean().item(),
                                       1.96 * ref_nn.std().item() / (N ** 0.5))

with open(common.OUT_DIR / "metrics_n256.txt", "w") as f:
    for k, (v, h) in results.items():
        line = f"{k}: {v:.1f} +/- {h:.1f}"
        print(line)
        f.write(line + "\n")
