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

def cast_tuple(val, length=None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    return output
def Upsample(dim, dim_out):
    return nn.ConvTranspose3d(dim, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c f (h s1) (w s2) -> b (c s1 s2) f h w', s1=2, s2=2),
        nn.Conv3d(dim * 4, dim_out, 1)
    )

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

class WeightStandardizedConv3d(nn.Conv3d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight

        mean = reduce(weight, 'o ... -> o 1 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1 1', partial(torch.var, unbiased = False))
        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

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

class FeatureMapConsolidator(nn.Module):
    def __init__(self,
                 dim,
                 *,
                 dim_ins=tuple(),
                 dim_outs=tuple(),
                 resize_fmap_before=True,
                 conv_block_fn=None
            ):
        super().__init__()
        self.needs_consolidating = len(dim_ins) > 0

        block_fn = default(conv_block_fn, Block)

        self.fmap_convs = nn.ModuleList([block_fn(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.resize_fmap_before = resize_fmap_before

        self.final_dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def resize_fmaps(self, fmaps, height, width):
        return [F.interpolate(fmap, (fmap.shape[-3], height, width)) for fmap in fmaps]


    def forward(self, x, fmaps=None):
        target_height, target_width = x.shape[-2:]

        fmaps = default(fmaps, tuple())

        if not self.needs_consolidating:
            return x

        if self.resize_fmap_before:
            fmaps = self.resize_fmaps(fmaps, target_height, target_width)

        outs = []
        for fmap, conv in zip(fmaps, self.fmap_convs):
            outs.append(conv(fmap))

        if self.resize_fmap_before:
            outs = self.resize_fmaps(outs, target_height, target_width)

        return torch.cat((x, *outs), dim = 1)

    def kernel_and_same_pad(*kernel_size):
        paddings = tuple(map(lambda k: k//2, kernel_size))
        return dict(kernel_size = kernel_size, padding = paddings)


class PixelShuffleUpsample(nn.Module):
    def __init__(self,
                 dim,
                 dim_out=None,
                 scale_factor=2
            ):
        super().__init__()
        self.scale_squared = scale_factor ** 2
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * self.scale_squared, 1)

        self.net = nn.Sequential(conv,
                                 nn.SiLU(),
                                 Rearrange('b (c r s) f h w -> b c f (h r) (w s)', r=scale_factor, s=scale_factor)
                                )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i ,*rest_dims = conv.weight.shape
        conv_weight = torch.empty(o // self.scale_squared, i, *rest_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.scale_squared)

        conv_weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = self.net(x)
        return x



class xunet(nn.Module):
    def __init__(self,
                 dim,
                 init_dim=None,
                 out_dim=None,
                 frame_kernel_size=1,
                 dim_mults=(1, 2, 4, 8),
                 num_blocks_per_stage=(2, 2, 2, 2),
                 num_self_attn_per_stage=(0, 0, 0, 1),
                 nested_unet_depths=(0, 0, 0, 0),
                 nested_unet_dim=32,
                 channels=3,
                 use_convnext=False,
                 resnet_groups=8,
                 consolidate_upsample_fmaps = True,
                 skip_scale=2 ** -0.5,
                 weight_standardize=False,
                 attn_heads=8,
                 attn_dim_head=32
                 ):
        super().__init__()

        self.train_as_image = frame_kernel_size == 1

        self.skip_scale = skip_scale
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(channels, init_dim, **kernel_and_same_pad(frame_kernel_size, 7, 7))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)
        blocks = partial(ConvNextBlock, frame_kernel_size=frame_kernel_size) if use_convnext else partial(ResnetBlock, groups=resnet_groups,
                                                                                                          weight_standardize=weight_standardize,
                                                                                 frame_kernel_size=frame_kernel_size)

        nested_unet_depth = cast_tuple(nested_unet_depths, num_resolutions)

        num_blocks_per_stage = cast_tuple(num_blocks_per_stage, num_resolutions)

        num_self_attn_per_stage = cast_tuple(num_self_attn_per_stage, num_resolutions)

        attn_heads = cast_tuple(num_self_attn_per_stage, num_resolutions)

        attn_heads = cast_tuple(attn_heads, num_resolutions)
        attn_dim_head = cast_tuple(num_self_attn_per_stage, num_resolutions)

        skip_dims = []

        down_stage_parameters = [
            in_out,
            nested_unet_depth,
            num_blocks_per_stage,
            num_self_attn_per_stage,
            attn_heads,
            attn_dim_head
        ]

        up_stage_parameters = [reversed(params[:-1]) for params in down_stage_parameters]

        for ind, ((dim_in, dim_out), nested_unet_depth, num_blocks, self_attn_blocks, heads, dim_head) in enumerate(zip(*down_stage_parameters)):
            is_last = ind >= (num_resolutions - 1)
            skip_dims.append(dim_in)

            self.downs.append(nn.ModuleList([
                blocks(dim_in,
                       dim_in,
                       nested_unet_depth=nested_unet_depth,
                       nested_unet_dim=nested_unet_dim,
                       ),
                nn.ModuleList([blocks(dim_in, dim_in, nested_unet_depth = nested_unet_depth, nested_unet_dim = nested_unet_dim) for _ in range(num_blocks - 1)]),
                nn.ModuleList([TransformerBlock(dim_in, depth=self_attn_blocks, heads=heads, dim_head=dim_head) for _ in range(self_attn_blocks)]),
                Downsample(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        mid_neted_unet_depth = nested_unet_depths[-1]

        self.mid = blocks(mid_dim, mid_dim, nested_unet_depth=mid_neted_unet_depth, nested_unet_dim=nested_unet_dim)
        self.mid_attn = Attention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_after = blocks(mid_dim, mid_dim, nested_unet_depth = mid_neted_unet_depth, nested_unet_dim = nested_unet_dim)

        self.mid_upsample = Upsample(mid_dim, dims[-2])

        for ind, ((dim_in, dim_out), nested_unet_depth, num_blocks, self_attn_blocks, heads, dim_head) in enumerate(zip(*up_stage_parameters)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                blocks(dim_out + skip_dims.pop(), dim_out, nested_unet_depth=nested_unet_depth, nested_unet_dim=nested_unet_dim),
                nn.ModuleList([blocks(dim_out, dim_out, nested_unet_depth=nested_unet_depth, nested_unet_dim=nested_unet_dim) for _ in range(num_blocks - 1)]),
                nn.ModuleList([TransformerBlock(dim_out, depth=self_attn_blocks, heads=heads, dim_head=dim_head) for _ in range(self_attn_blocks)]),
                Upsample(dim_out, dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)

        if consolidate_upsample_fmaps:
            self.consolidator = FeatureMapConsolidator(
                dim,
                dim_ins=tuple(map(lambda m: dim * m, dim_mults)),
                dim_outs=(dim,) * len(dim_mults),
                conv_block_fn=blocks
            )
        else:
            self.consolidator = FeatureMapConsolidator(dim=dim)

        final_dim_in = self.consolidator.final_dim_out

        self.final_conv = nn.Sequential(
            blocks(final_dim_in + dim, dim),
            nn.Conv3d(dim, out_dim, **kernel_and_same_pad(frame_kernel_size, 3, 3))
        )

    def forward(self, x):
        is_image = x.ndim == 4

        if is_image:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        x = self.init_conv(x)

        r = x.clone()

        down_hiddens = []

        up_hiddens = []

        for init_block, blocks, attn_blocks, downsample in self.downs:
            x = init_block(x)

            for block in blocks:
                x = block(x)

            for attn_block in attn_blocks:
                x = attn_block(x)

            down_hiddens.append(x)
            x = downsample(x)

        x = self.mid(x)
        x = self.mid_attn(x) + x
        x = self.mid_after(x)

        up_hiddens.append(x)
        x = self.mid_upsample(x)

        for init_block, blocks, attn_blocks, upsample in self.ups:
            x = torch.cat((x, down_hiddens.pop() * self.skip_scale), dim=1)

            x = init_block(x)

            for block in blocks:
                x = block(x)

            for attn_block in attn_blocks:
                x = attn_block(x)

            up_hiddens.insert(0, x)
            x = upsample(x)

        x = self.consolidator(x, up_hiddens)

        x = torch.cat((x, r), dim=1)

        out = self.final_conv(x)

        if is_image:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

class NestedResidualUnet(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            M=32,
            frame_kernel_size=1,
            add_residual=False,
            groups=4,
            skip_scale=2**-0.5,
            weight_standardize=False
            ):
        super().__init__()

        self.depth = depth
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        conv = WeightStandardizedConv3d if weight_standardize else nn.Conv3d

        for ind in range(depth):
            is_first = ind = 0
            dim_in = dim if is_first else M

            down = nn.Sequential(
                conv(dim_in, M, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.GroupNorm(groups, M),
                nn.SiLU()
            )

            up = nn.Sequential(
                PixelShuffleUpsample(2 * M, dim_in),
                nn.GroupNorm(groups, dim_in),
                nn.SiLU()
            )

            self.downs.append(down)
            self.ups.append(up)

        self.mid = nn.Sequential(
            conv(M, M, **kernel_and_same_pad(frame_kernel_size, 3, 3)),
            nn.GroupNorm(groups, M),
            nn.SiLU()
        )

        self.skip_scale = skip_scale
        self.add_residual = add_residual

    def forward(self, x, residual = None):
        is_video = x.ndim == 5

        if self.add_residual:
            residual = default(residual, x.clone())

        *_, h, w = x.shape

        layers = len(self.ups)

        hiddens = []

        for down in self.downs:
            x = down(x)
            hiddens.append(x.clone().contiguous())

        x = self.mid(x)

        for up in reversed(self.ups):
            x = torch.cat((x, hiddens.pop() * self.skip_scale), dim = 1)
            x = up(x)

        if self.add_residual:
            x = x + residual
            x = F.silu(x)

        return x





