"""What is "the average face given the 10% signal", exactly?

The best linear denoiser (exact posterior mean under a Gaussian fit of the data)
has a closed form from the dataset mean mu and covariance Sigma:

    E[x0 | xt] = mu + a*Sigma (a^2*Sigma + b^2*I)^-1 (xt - a*mu),  a=0.1, b=0.9

In the PCA basis each direction with variance lam is recovered with Wiener gain
SNR/(1+SNR), SNR = a^2*lam/b^2 = lam/81: only the few coarsest directions
survive, everything else shrinks to the mean.

The trained t09 model's *expected* recon is estimated by averaging its one-step
prediction over 256 noise draws of the same image (per-draw hallucinations
average out, the consistently-recovered signal stays).

Outputs:
  fig13_avg_given_signal.png  8 images (same as fig3), rows:
    original
    noised input (one draw, same as fig3)
    trained model, that single draw
    linear-optimal, that single draw
    trained model, averaged over 256 draws
    linear-optimal in expectation over draws  (= "the average given the 10% signal")
  avg_signal_stats.txt  spectrum + agreement numbers
"""
import common

import torch
from diffusers import AutoencoderKLFlux2
from torchvision.utils import make_grid, save_image

from dit import TinyDiT

device = "cuda"
torch.manual_seed(1234)

A, B = 0.1, 0.9

stats = torch.load(common.LATENTS_DIR / "stats.pt")
mean_s, std_s = stats["mean"].to(device), stats["std"].to(device)
latents_n = ((torch.load(common.LATENTS_DIR / "latents.pt").float() - stats["mean"]) / stats["std"])
C, H, W = latents_n.shape[1:]
D = C * H * W

# reproduce fig3's RNG stream: noise16 first, then eps for the real images
noise16 = torch.randn(16, C, H, W, device=device)
real_idx = torch.arange(8) * 1000 + 17
x0 = latents_n[real_idx].to(device)
eps = torch.randn_like(x0)
xt = A * x0 + B * eps

ck = torch.load(common.CKPT_DIR / "dit_t09.pt", map_location=device)
m = TinyDiT(in_channels=C, grid=H).to(device)
m.load_state_dict(ck["ema"])
m.eval().requires_grad_(False)

vae = AutoencoderKLFlux2.from_pretrained(common.VAE_REPO, torch_dtype=torch.bfloat16).to(device).eval()


@torch.no_grad()
def one_step(x):
    return x - 0.9 * m(x, torch.full((len(x),), 0.9, device=device))


@torch.no_grad()
def decode(lat_n):
    lat = lat_n.to(device) * std_s + mean_s
    imgs = vae.decode(lat.to(torch.bfloat16)).sample.float()
    return (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()


# ---------- exact linear-MMSE denoiser from data moments ----------
X = latents_n.flatten(1).to(device)
mu = X.mean(0)
Xc = X - mu
cov = (Xc.T @ Xc) / (len(X) - 1)
lam, V = torch.linalg.eigh(cov)          # ascending eigenvalues
snr = (A * A / (B * B)) * lam
gain_x = A * lam / (A * A * lam + B * B)     # applied to (xt - A*mu)
gain_e = snr / (1 + snr)                     # applied to (x0 - mu), expectation over draws


def to_img(flat):
    return flat.view(-1, C, H, W)


lin_single = mu + ((V.T @ (xt.flatten(1) - A * mu).T) * gain_x[:, None]).T @ V.T
lin_expect = mu + ((V.T @ (x0.flatten(1) - mu).T) * gain_e[:, None]).T @ V.T

# ---------- trained model: single draw + Monte-Carlo expectation ----------
model_single = one_step(xt)
N_MC = 256
acc = torch.zeros_like(x0)
g = torch.Generator(device=device).manual_seed(4321)
with torch.no_grad():
    for i in range(len(x0)):
        e = torch.randn(N_MC, C, H, W, device=device, generator=g)
        acc[i] = one_step(A * x0[i:i + 1] + B * e).mean(0)
model_avg = acc

# ---------- figure ----------
rows = torch.cat([x0, xt, model_single, to_img(lin_single), model_avg, to_img(lin_expect)])
imgs = torch.cat([decode(rows[i:i + 8]) for i in range(0, len(rows), 8)])
save_image(make_grid(imgs, nrow=8, padding=2), common.OUT_DIR / "fig13_avg_given_signal.png")

# ---------- stats ----------
lam_d = lam.flip(0)  # descending
lines = []
lines.append(f"latent dim {D}, trace(cov) = {lam.sum().item():.0f}")
lines.append("top eigenvalues: " + ", ".join(f"{v:.0f}" for v in lam_d[:10]))
lines.append(f"directions with SNR > 1 (lam > 81): {(lam > 81).sum().item()}")
lines.append(f"directions with SNR > 0.1 (lam > 8.1): {(lam > 8.1).sum().item()}")
rec_frac = (lam * gain_e).sum() / lam.sum()
lines.append(f"variance-weighted recovered fraction of the signal: {rec_frac.item():.3f}")

d_mean_orig = (x0.flatten(1) - mu).norm(dim=1)
mo = model_avg.flatten(1) - mu
lo = lin_expect - mu
cos = (mo * lo).sum(1) / (mo.norm(dim=1) * lo.norm(dim=1))
lines.append(f"|x0 - mu| per image: {d_mean_orig.mean().item():.1f}")
lines.append(f"|model_avg - mu|: {mo.norm(dim=1).mean().item():.1f}, "
             f"|linear_expect - mu|: {lo.norm(dim=1).mean().item():.1f}")
lines.append(f"cosine(model_avg - mu, linear_expect - mu) per image: "
             + ", ".join(f"{c:.2f}" for c in cos) + f"  (mean {cos.mean().item():.2f})")
d_hall = (model_single - model_avg).flatten(1).norm(dim=1)
lines.append(f"per-draw hallucination magnitude |single - avg|: {d_hall.mean().item():.1f}")

# ---------- same trick on PURE noise: what does the linear student see in static? ----------
# fig13b: same 16 noises as fig2. Column = one noise; row pairs = trained model / Wiener.
model_pure = one_step(noise16)
wien_pure16 = to_img(mu + ((V.T @ (noise16.flatten(1) - A * mu).T) * gain_x[:, None]).T @ V.T)
rows_b = torch.cat([model_pure[:8], wien_pure16[:8], model_pure[8:], wien_pure16[8:]])
imgs_b = torch.cat([decode(rows_b[i:i + 8]) for i in range(0, len(rows_b), 8)])
save_image(make_grid(imgs_b, nrow=8, padding=2), common.OUT_DIR / "fig13b_wiener_pure_noise.png")

# metrics on the same 256 noises as variance_check (default generator, seed 1234, first draw)
g2 = torch.Generator(device=device).manual_seed(1234)
noise256 = torch.randn(256, C, H, W, device=device, generator=g2)
wien_pure = mu + ((V.T @ (noise256.flatten(1) - A * mu).T) * gain_x[:, None]).T @ V.T
model_pure256 = one_step(noise256)


def diversity_ci(flat):
    f = flat.reshape(len(flat), -1).float().cpu()
    dm = torch.cdist(f, f)
    n = len(f)
    point = (dm.sum() / (n * (n - 1))).item()
    groups = []
    for gi in range(n // 16):
        sub = dm[gi * 16:(gi + 1) * 16, gi * 16:(gi + 1) * 16]
        groups.append((sub.sum() / (16 * 15)).item())
    gt = torch.tensor(groups)
    return point, 1.96 * gt.std().item() / (len(groups) ** 0.5)


def nn_dist(flat):
    d = torch.cdist(flat.reshape(len(flat), -1), X)
    return d.min(1).values.mean().item()


dw, hw = diversity_ci(wien_pure)
dm256, hm256 = diversity_ci(model_pure256)
lines.append("")
lines.append("--- pure noise ---")
lines.append(f"Wiener on pure noise: diversity {dw:.1f} +/- {hw:.1f}, "
             f"dist-to-mean {(wien_pure - mu).norm(dim=1).mean().item():.1f}, "
             f"nearest-train {nn_dist(wien_pure):.1f}")
lines.append(f"model on pure noise (same 256): diversity {dm256:.1f} +/- {hm256:.1f}, "
             f"nearest-train {nn_dist(model_pure256.flatten(1)):.1f}")

# ---------- full 64-step Euler sampling with the Wiener denoiser ----------
# At each t: a = 1-t, gains a*lam/(a^2*lam + t^2), x0hat -> v = (x - x0hat)/t.
@torch.no_grad()
def euler_wiener(x, steps=64):
    xf = x.reshape(len(x), -1).clone()
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].item()
        a = 1.0 - t
        g = a * lam / (a * a * lam + t * t)
        c = V.T @ (xf - a * mu).T
        x0hat = mu + (V @ (g[:, None] * c)).T
        xf = xf + (ts[i + 1].item() - t) * (xf - x0hat) / t
    return xf


@torch.no_grad()
def euler_model(mdl, x, steps=64):
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        v = mdl(x, torch.full((len(x),), ts[i].item(), device=device))
        x = x + (ts[i + 1] - ts[i]) * v
    return x


def nearest_train(flat):
    d = torch.cdist(flat.reshape(len(flat), -1), X)
    vals, idx = d.min(1)
    return vals, idx


ck_full = torch.load(common.CKPT_DIR / "dit_full.pt", map_location=device)
m_full = TinyDiT(in_channels=C, grid=H).to(device)
m_full.load_state_dict(ck_full["ema"])
m_full.eval().requires_grad_(False)

wien_samp8 = euler_wiener(noise16[:8])
full_samp8 = euler_model(m_full, noise16[:8].clone())
wv, wi = nearest_train(wien_samp8)
fv, fi = nearest_train(full_samp8)
rows_c = torch.cat([to_img(wien_samp8), latents_n[wi.cpu()].to(device),
                    full_samp8, latents_n[fi.cpu()].to(device)])
imgs_c = torch.cat([decode(rows_c[i:i + 8]) for i in range(0, len(rows_c), 8)])
save_image(make_grid(imgs_c, nrow=8, padding=2), common.OUT_DIR / "fig13c_wiener_full_sampling.png")

wien_samp256 = euler_wiener(noise256)
full_samp256 = euler_model(m_full, noise256.clone())
dws, hws = diversity_ci(wien_samp256)
wnn = nearest_train(wien_samp256)[0]
fnn = nearest_train(full_samp256)[0]
lines.append("")
lines.append("--- full 64-step sampling ---")
lines.append(f"Wiener sampler: diversity {dws:.1f} +/- {hws:.1f}, "
             f"nearest-train {wnn.mean().item():.1f} +/- {1.96 * wnn.std().item() / (len(wnn) ** 0.5):.1f}")
lines.append(f"trained full model: nearest-train {fnn.mean().item():.1f} "
             f"+/- {1.96 * fnn.std().item() / (len(fnn) ** 0.5):.1f}")
lines.append(f"(real NN yardstick: 88.5 +/- 1.4; real diversity: 111.6 +/- 0.9)")

with open(common.OUT_DIR / "avg_signal_stats.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote fig13_avg_given_signal.png, fig13b_wiener_pure_noise.png, fig13c_wiener_full_sampling.png")
