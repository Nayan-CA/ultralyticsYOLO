# Vision Mamba backbone — pure PyTorch + mamba_ssm, no timm dependency
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class DropPath(nn.Module):
    """Stochastic depth regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        )
        return x / keep_prob * random_tensor


def _trunc_normal(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0, b=2.0)


class PatchEmbed2D(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim."""

    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                          # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class PatchMerging2D(nn.Module):
    """2x spatial downsampling, doubles channels."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if H % 2 or W % 2:
            x = nn.functional.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = torch.cat([
            x[:, 0::2, 0::2, :],
            x[:, 1::2, 0::2, :],
            x[:, 0::2, 1::2, :],
            x[:, 1::2, 1::2, :],
        ], dim=-1)                                # (B, H/2, W/2, 4C)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class VimBlock(nn.Module):
    """
    Bidirectional Mamba block.
    Runs SSM in both forward and backward directions over the token sequence,
    then combines outputs — giving every token full global context.
    """

    def __init__(self, dim, drop_path=0.0, d_state=16, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=4, expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=4, expand=expand
        )
        self.out_proj = nn.Linear(dim * 2, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        if not x.is_cuda:
            return residual

        x = self.norm(x)
        y_fwd = self.mamba_fwd(x)
        y_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        y = self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        return residual + self.drop_path(y)


class HierVimStage(nn.Module):
    """Stack of VimBlocks at a fixed spatial resolution."""

    def __init__(self, dim, depth, drop_path_rates, d_state=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            VimBlock(dim=dim, drop_path=drop_path_rates[i], d_state=d_state)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class HierVimBackbone(nn.Module):
    """
    Hierarchical Vision Mamba backbone.

    Produces three feature maps for FPN-style detection heads:
        P3 — 1/8  resolution, dims[1] channels
        P4 — 1/16 resolution, dims[2] channels
        P5 — 1/32 resolution, dims[3] channels

    Default config matches Hier-Vim-T from the paper:
        depths=(2, 2, 5, 2), dims=(96, 192, 384, 768)

    Args:
        in_chans      : input image channels (default 3)
        depths        : number of VimBlocks per stage
        dims          : channel dims per stage
        d_state       : SSM state size (default 16)
        drop_path_rate: stochastic depth max rate
        patch_size    : initial patch size (default 4 → 1/4 downsample)
    """

    def __init__(
        self,
        in_chans=3,
        depths=(2, 2, 5, 2),
        dims=(96, 192, 384, 768),
        d_state=16,
        drop_path_rate=0.1,
        patch_size=4,
    ):
        super().__init__()

        total_depth = sum(depths)
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.patch_embed = PatchEmbed2D(in_chans, dims[0], patch_size)

        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        cur = 0
        for i in range(4):
            self.stages.append(
                HierVimStage(
                    dim=dims[i],
                    depth=depths[i],
                    drop_path_rates=dpr[cur: cur + depths[i]],
                    d_state=d_state,
                )
            )
            cur += depths[i]
            self.downsamplers.append(
                PatchMerging2D(dims[i]) if i < 3 else None
            )

        # Output norms for P3, P4, P5
        self.norm_p3 = nn.LayerNorm(dims[1])
        self.norm_p4 = nn.LayerNorm(dims[2])
        self.norm_p5 = nn.LayerNorm(dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stage 0 — P2/4
        x, H, W = self.patch_embed(x)
        x = self.stages[0](x)
        x, H, W = self.downsamplers[0](x, H, W)

        # Stage 1 — P3/8
        x = self.stages[1](x)
        p3 = self.norm_p3(x).transpose(1, 2).reshape(x.shape[0], -1, H, W)
        x, H, W = self.downsamplers[1](x, H, W)

        # Stage 2 — P4/16
        x = self.stages[2](x)
        p4 = self.norm_p4(x).transpose(1, 2).reshape(x.shape[0], -1, H, W)
        x, H, W = self.downsamplers[2](x, H, W)

        # Stage 3 — P5/32
        x = self.stages[3](x)
        p5 = self.norm_p5(x).transpose(1, 2).reshape(x.shape[0], -1, H, W)

        return [p3, p4, p5]
        # shapes: [(B,192,H/8,W/8), (B,384,H/16,W/16), (B,768,H/32,W/32)]
