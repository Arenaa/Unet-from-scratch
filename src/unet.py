from functools import partial

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, Union, Optional

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def kernel_and_same_pad(*kernel_size):
    paddings = tuple(map(lambda k: k//2, kernel_size))
    return dict(kernel_size = kernel_size, paddings=paddings)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + eps).sqrt() * self.gamma

class Block(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 groups=8,
                 weight_standardize=False,
                 frame_kernel_size=1
            ):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)
        conv = nn.Conv3d if not weight_standardize else None

        self.proj = conv(dim, dim_out, **kernel_conv_kwargs(3, 3))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 groups=8,
                 frame_kernel_size=1,
                 nested_unet_depth=0,
                 nested_unet_dim=32,
                 weight_standardize=False):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups, weight_standardize=weight_standardize, frame_kernel_size=frame_kernel_size)
        self.block2 = Block(dim_out, dim_out, groups=groups, weight_standardize=weight_standardize, frame_kernel_size=frame_kernel_size)

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim!=dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1, 1))

    def forward(self, x):
        spatial_l2_norm = x.norm(p=2, dim=(2, 3, 4), keepdim=True)
        feat_norm = spatial_l2_norm / spatial_l2_norm.mean(dim=-1, keepdim=True).clamp(min = self.eps)
        return x * feat_norm * self.gamma + self.bias + x

class ConvNextBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 *,
                 mul=2,
                 frame_kernel_size=1,
                 nested_unet_depth=0,
                 nested_unet_dim=32):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)

        self.ds_conv = nn.Conv3d(dim, dim, **kernel_conv_kwargs(7, 7), groups=dim)

        inner_dim = dim_out * mul

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv3d(dim, inner_dim, **kernel_conv_kwargs(3, 3), groups=dim_out),
            nn.GELU(),
            GRN(inner_dim),
            nn.Conv3d(inner_dim, dim_out, **kernel_conv_kwargs(3, 3), groups=dim_out)
        )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim!=dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)

def FeedForward(dim, mult=4.):
    inner_dim = int(dim * mult)
    return Residual(nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, inner_dim, 1, bias=False),
        nn.GELU(),
        LayerNorm(inner_dim),
        nn.Conv3d(inner_dim, dim, 1, bias=False)
    ))

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=64
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self._norm = LayerNorm(dim)

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1, bias=False)

    def forward(self, x):
        f, h, w = x.shape[-3:]

        residual = x.clone()
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k ,v = map(lambda t:rearrange(t, 'b (h c) ... -> b h (...) c', h=self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (f x y) d -> b (h d) f x y', f = f, x = h, y = w)
        return self.to_out(out) + residual

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 *,
                 depth,
                 **kwargs
            ):
        super().__init__()
        self.attn = Attention(dim, **kwargs)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x