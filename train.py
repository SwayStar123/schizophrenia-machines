"""Flow matching training on precomputed FLUX.2 latents.

Convention: x_t = (1-t)*x0 + t*x1 with x1 ~ N(0,1), so t=1 is pure noise.
The model predicts the velocity v = x1 - x0. Sampling integrates from t=1 to t=0
via x_{t-dt} = x_t - dt * v(x_t, t).

--t-mode full   : t ~ U(0,1)          (a normal flow matching model)
--t-mode t1     : t = 1.0 always      (model only ever sees pure noise)
--t-mode t09    : t = 0.9 always      (model only ever sees 90%-noised data)
--t-mode uncond : t ∈ {1.0, 0.9} 50/50, but the model receives NO noise
                  conditioning (t input frozen to 0) — it must infer the noise
                  level from the input itself.
"""
import common

import argparse
import copy
import time

import torch

from dit import TinyDiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-mode", choices=["full", "t1", "t09", "uncond"], required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    latents = torch.load(common.LATENTS_DIR / "latents.pt").float()
    stats = torch.load(common.LATENTS_DIR / "stats.pt")
    latents = (latents - stats["mean"]) / stats["std"]
    latents = latents.to(device)  # ~30k tiny latents fit fine on GPU
    N, C, H, W = latents.shape
    print(f"latents on gpu: {latents.shape}, mem {latents.element_size()*latents.nelement()/1e9:.2f} GB")

    model = TinyDiT(in_channels=C, grid=H).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    warmup = 500

    def sample_t(bs):
        if args.t_mode == "full":
            return torch.rand(bs, device=device)
        if args.t_mode == "uncond":
            return torch.where(torch.rand(bs, device=device) < 0.5, 1.0, 0.9)
        return torch.full((bs,), 1.0 if args.t_mode == "t1" else 0.9, device=device)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, step / warmup)

        idx = torch.randint(0, N, (args.batch,), device=device)
        x0 = latents[idx]
        x1 = torch.randn_like(x0)
        t = sample_t(args.batch)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        target = x1 - x0
        # uncond: the model never learns what t was — conditioning input frozen to 0
        t_input = torch.zeros_like(t) if args.t_mode == "uncond" else t

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xt, t_input)
            loss = torch.nn.functional.mse_loss(pred.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            d = args.ema
            for pe, p in zip(ema_model.parameters(), model.parameters()):
                pe.lerp_(p, 1 - d)

        if step % 500 == 0 or step == 1:
            el = time.time() - t0
            eta = (args.steps - step) / max(step / max(el, 1e-9), 1e-9)
            print(f"[{args.t_mode}] step {step}/{args.steps} loss {loss.item():.4f} "
                  f"({el:.0f}s, {step/max(el,1e-9):.1f} it/s, eta {eta/60:.0f}m)", flush=True)

        if step % 500 == 0 or step == args.steps:
            out = common.CKPT_DIR / f"dit_{args.t_mode}.pt"
            torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(),
                        "step": step, "config": {"in_channels": C, "grid": H}}, out)
            print(f"[{args.t_mode}] checkpoint saved at step {step}", flush=True)


if __name__ == "__main__":
    main()
