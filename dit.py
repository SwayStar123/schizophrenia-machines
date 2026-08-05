"""Tiny DiT for flow matching on FLUX.2 latents.

adaLN-zero conditioning on t, fixed 2D sin-cos positional embeddings.
Latents are small (16x16 or 8x8 spatial), so this stays fast even at batch 512.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None] * 1000.0  # t in [0,1] -> scale like discrete steps
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sincos_pos_embed_2d(dim: int, grid: int) -> torch.Tensor:
    """[grid*grid, dim] fixed 2D sin-cos embedding."""
    assert dim % 4 == 0
    coords = torch.arange(grid).float()
    omega = torch.arange(dim // 4).float() / (dim // 4)
    omega = 1.0 / (10_000 ** omega)
    out = coords[:, None] * omega[None]  # [grid, dim/4]
    emb_1d = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [grid, dim/2]
    emb_y = emb_1d[:, None, :].expand(grid, grid, -1)
    emb_x = emb_1d[None, :, :].expand(grid, grid, -1)
    return torch.cat([emb_y, emb_x], dim=-1).reshape(grid * grid, dim)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c)[:, None].chunk(6, dim=-1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1)
        x = x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)
        return x


class TinyDiT(nn.Module):
    def __init__(self, in_channels: int = 32, grid: int = 16, patch: int = 1,
                 dim: int = 384, depth: int = 8, heads: int = 6):
        super().__init__()
        assert grid % patch == 0
        self.in_channels = in_channels
        self.grid = grid
        self.patch = patch
        self.tokens_per_side = grid // patch
        patch_dim = in_channels * patch * patch

        self.proj_in = nn.Linear(patch_dim, dim)
        self.register_buffer("pos", sincos_pos_embed_2d(dim, self.tokens_per_side)[None], persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj_out = nn.Linear(dim, patch_dim)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p, s = self.patch, self.tokens_per_side
        x = x.reshape(B, C, s, p, s, p).permute(0, 2, 4, 1, 3, 5).reshape(B, s * s, C * p * p)
        x = self.proj_in(x) + self.pos
        c = self.t_mlp(timestep_embedding(t, 256))
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.ada_out(c)[:, None].chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale) + shift
        x = self.proj_out(x)
        x = x.reshape(B, s, s, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x


if __name__ == "__main__":
    m = TinyDiT()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.2f}M")
    x = torch.randn(2, 32, 16, 16)
    t = torch.rand(2)
    print(m(x, t).shape)
