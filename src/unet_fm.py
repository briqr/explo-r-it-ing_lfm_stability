# unet_fm.py
# Minimal U-Net for Flow Matching velocity prediction.

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- helpers ---------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal time embedding (expects t in [0,1]).
    Returns [B, dim].
    """
    t = t.float()
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0), math.log(10000.0), half, device=device
        )
        * (-1.0)
    )
    
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb


class FiLM(nn.Module):
    """
    Produces scale, shift for feature-wise affine modulation from a context vector.
    """
    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_channels * 2)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.proj(h)
        scale, shift = s.chunk(2, dim=-1)
        return scale, shift


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(emb_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb)
        # reshape [B,C] -> [B,C,1,1]
        scale = scale[..., None, None]
        shift = shift[..., None, None]
        h = F.silu(self.norm2(h)) * (1 + scale) + shift
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """
    Lightweight channel attention (optional). Keeps it simple: self-attn on channels with spatial pooling.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = x
        x = self.norm(x)
        # pool to tokens (mean spatial)
        x_tok = x.mean(dim=(2,3))  # [B,C]
        q = self.q(x_tok)
        k = self.k(x_tok)
        v = self.v(x_tok)

        # multi-head
        def split_heads(t):
            return t.view(b, self.num_heads, c // self.num_heads)

        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
        attn = torch.einsum("bhd,bhd->bh", qh, kh) / math.sqrt(c // self.num_heads)
        attn = attn.softmax(dim=-1).unsqueeze(-1)  # [B,H,1]
        out = (attn * vh).reshape(b, c)
        out = self.proj(out).view(b, c, 1, 1)
        return h_in + out


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# --------- UNetFM ---------

class UNetFM(nn.Module):
    """
    U-Net for FM velocity prediction.
    - Predicts v_theta with same shape as input latent x.
    """
    def __init__(
        self,
        input_size: int,          # latent spatial size (e.g., 32 for 256/8)
        in_channels: int,         # 4 or 8 (latent channels)
        num_classes: int = 1,     # 1 for unconditional
        base_channels: int = 192, # width
        channel_mult: Tuple[int,...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_levels: Optional[List[int]] = None,  # which levels to add attention
        dropout: float = 0.0,
        learn_sigma: bool = False,                # ignored for FM
        class_dropout_prob: float = 0.0          # you control y upstream
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # FM velocity predicts same shape
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma   # not used for FM
        self.class_dropout_prob = class_dropout_prob
        if attn_levels is None:
            attn_levels = []

        # Embedding dims
        self.emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(128, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        if num_classes > 1:
            self.cls_emb = nn.Embedding(num_classes, self.emb_dim)
        else:
            self.cls_emb = None

        # Input
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        chs = [int(base_channels * m) for m in channel_mult]
        downs = []
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        self.skip_channels = []

        for level, ch in enumerate(chs):
            if in_ch != ch:
                proj = nn.Conv2d(in_ch, ch, 1)
            else:
                proj = nn.Identity()
            res = nn.ModuleList([ResBlock(ch, ch, self.emb_dim, dropout=dropout) for _ in range(num_res_blocks)])
            attn = AttnBlock(ch) if level in attn_levels else nn.Identity()
            self.down_blocks.append(nn.ModuleDict(dict(proj=proj, res=res, attn=attn)))
            self.skip_channels.append(ch)
            if level != len(chs) - 1:
                downs.append(Down(ch))
                in_ch = ch

        self.downs = nn.ModuleList(downs)

        # Middle
        mid_ch = chs[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, self.emb_dim, dropout=dropout)
        self.mid_attn  = AttnBlock(mid_ch) if (len(chs)-1) in attn_levels else nn.Identity()
        self.mid_block2 = ResBlock(mid_ch, mid_ch, self.emb_dim, dropout=dropout)

        # Ups
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_match = nn.ModuleList()   # NEW: match channels from prev level -> current level

        for li, level in enumerate(reversed(range(len(chs)))):  # order: L-1, L-2, ..., 0
            ch = chs[level]
            in_from_prev = chs[level] if level == (len(chs)-1) else chs[level+1]
            self.up_match.append(nn.Identity() if in_from_prev == ch else nn.Conv2d(in_from_prev, ch, 1))

            proj = nn.Conv2d(2 * ch, ch, 1)
            res = nn.ModuleList([ResBlock(ch, ch, self.emb_dim, dropout=dropout) for _ in range(num_res_blocks)])
            attn = AttnBlock(ch) if level in attn_levels else nn.Identity()
            self.up_blocks.append(nn.ModuleDict(dict(proj=proj, res=res, attn=attn)))

            # upsample to next spatial size for the next (lower) level, except last level
            if level != 0:
                self.ups.append(Up(ch))

        

        # Output head
        self.out_norm = nn.GroupNorm(32, chs[0])
        self.out_conv = nn.Conv2d(chs[0], self.out_channels, 3, padding=1)

    def _embed(self, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        te = timestep_embedding(t, 128)
        te = self.time_mlp(te)  # [B, emb_dim]
        if self.cls_emb is not None and y is not None:
            ye = self.cls_emb(y)
            te = te + ye
        return te

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, C, H, W] latent
        t: [B] or [B,1], in [0,1]
        y: [B] (optional class labels when num_classes > 1)
        returns v_theta(x,t): [B, C, H, W]
        """
        emb = self._embed(t, y)
        h = self.in_conv(x)
        skips = []

        # Down path
        for i, blk in enumerate(self.down_blocks):
            h = blk['proj'](h)
            for rb in blk['res']:
                h = rb(h, emb)
            h = blk['attn'](h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Up path
        for i, blk in enumerate(self.up_blocks):
            level = len(self.up_blocks) - 1 - i
            skip = skips[level]

            # ensure spatial size matches the skip
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")

            # NEW: match channels from previous level to current level 'ch'
            h = self.up_match[i](h)

            # concat with skip, then reduce back to ch
            h = torch.cat([h, skip], dim=1)
            h = blk['proj'](h)

            for rb in blk['res']:
                h = rb(h, emb)
            h = blk['attn'](h)

            if i < len(self.ups):
                h = self.ups[i](h)
        h = F.silu(self.out_norm(h))
        v = self.out_conv(h)
        return v


# ----- Registry to merge with DiT_models dict -----

def _make_unet(size_tag: str, **kw):
    """
    size_tag in {"S","M","L"} controls width/depth a bit.
    """
    if size_tag == "S":
        return UNetFM(base_channels=160, channel_mult=(1,2,2,3), **kw)
    if size_tag == "M":
        return UNetFM(base_channels=192, channel_mult=(1,2,3,4), **kw)
    if size_tag == "L":
        return UNetFM(base_channels=256, channel_mult=(1,2,3,4), **kw)
    raise ValueError(f"Unknown size tag: {size_tag}")

UNet_models = {
    "UNet-S/2": lambda input_size, in_channels, num_classes=1, class_dropout_prob=0.0, learn_sigma=False, **_: 
        _make_unet("S", input_size=input_size, in_channels=in_channels, num_classes=num_classes,
                   class_dropout_prob=class_dropout_prob, learn_sigma=learn_sigma),
    "UNet-M/2": lambda input_size, in_channels, num_classes=1, class_dropout_prob=0.0, learn_sigma=False, **_: 
        _make_unet("M", input_size=input_size, in_channels=in_channels, num_classes=num_classes,
                   class_dropout_prob=class_dropout_prob, learn_sigma=learn_sigma),
    "UNet-L/2": lambda input_size, in_channels, num_classes=1, class_dropout_prob=0.0, learn_sigma=False, **_: 
        _make_unet("L", input_size=input_size, in_channels=in_channels, num_classes=num_classes,
                   class_dropout_prob=class_dropout_prob, learn_sigma=learn_sigma),
}
