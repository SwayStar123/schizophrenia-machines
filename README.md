# schizophrenia-machines

Experiments for the article *"Diffusion models are schizophrenia machines"*.

Everything is self-contained in this folder (venv, HF cache, dataset, checkpoints) —
delete the folder and it's all gone.

## Pipeline

```bash
.venv/Scripts/python download_assets.py     # FLUX.2 VAE + CelebA-HQ 256 into ./hf_cache
.venv/Scripts/python precompute_latents.py  # encode 28k images at 128x128 -> ./latents
.venv/Scripts/python train.py --t-mode full # normal flow matching model, t ~ U(0,1)
.venv/Scripts/python train.py --t-mode t1   # trained ONLY at t=1 (pure noise)
.venv/Scripts/python train.py --t-mode t09  # trained ONLY at t=0.9
.venv/Scripts/python experiments.py         # all article figures -> ./outputs
```

## The idea

Flow matching convention: `x_t = (1-t)·x0 + t·noise`, model predicts `v = noise - x0`.

- A model trained only at t=1 sees zero information about x0. Its optimal one-step
  output is the dataset mean — a blurry, front-facing, unisex face.
- A model trained only at t=0.9 learns to read faint structure out of 90% noise.
  Fed *pure* noise (which it never saw in training), it sees patterns that don't exist anyway.
- A full model does this at every step of sampling: each Euler step manufactures a
  slightly-off-manifold input for the next step, and the model's generalization
  fills the gap with things that were never there.
