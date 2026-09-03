import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Install mamba_ssm: pip install mamba_ssm")


class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding for hierarchical stages."""

    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)          # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class PatchMerging2D(nn.Module):
    """Downsample spatial resolution, double channels."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        # pad if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class VimBlock(nn.Module):
    """
    Bidirectional Mamba block for vision.
    Forward + Backward SSM combined.
    """

    def __init__(self, dim, drop_path=0.0, d_state=16, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=expand,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x: (B, L, C)
        residual = x
        x = self.norm(x)

        # Forward direction
        y_fwd = self.mamba_fwd(x)

        # Backward direction (flip sequence, process, flip back)
        y_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])

        # Combine
        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.out_proj(y)

        return residual + self.drop_path(y)


class HierVimStage(nn.Module):
    """One stage of Hier-Vim: N Vim blocks at a fixed resolution."""

    def __init__(self, dim, depth, drop_path_rates, d_state=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            VimBlock(
                dim=dim,
                drop_path=drop_path_rates[i],
                d_state=d_state,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class HierVimBackbone(nn.Module):
    """
    Hierarchical Vision Mamba backbone.
    Outputs feature maps at P3 (1/8), P4 (1/16), P5 (1/32) scales.
    Matches Hier-Vim-T config from the paper:
      depths=[2,2,5,2], dims=[96,192,384,768]
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

        # Stochastic depth decay
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Patch embedding (stage 0): 4× downsample → P2/4
        self.patch_embed = PatchEmbed2D(in_chans, dims[0], patch_size)

        # 4 stages with patch merging between them
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        cur = 0
        for i in range(4):
            stage = HierVimStage(
                dim=dims[i],
                depth=depths[i],
                drop_path_rates=dpr[cur: cur + depths[i]],
                d_state=d_state,
            )
            self.stages.append(stage)
            cur += depths[i]

            # Downsampler between stages (not after last stage)
            if i < 3:
                self.downsamplers.append(PatchMerging2D(dims[i]))
            else:
                self.downsamplers.append(None)

        # Output norms for P3, P4, P5 (stages 1, 2, 3)
        self.norm2 = nn.LayerNorm(dims[1])  # P3 output
        self.norm3 = nn.LayerNorm(dims[2])  # P4 output
        self.norm4 = nn.LayerNorm(dims[3])  # P5 output

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stage 0: P2/4
        x, H, W = self.patch_embed(x)
        x = self.stages[0](x)
        x, H, W = self.downsamplers[0](x, H, W)

        # Stage 1: P3/8
        x = self.stages[1](x)
        p3 = self.norm2(x).transpose(1, 2).view(x.shape[0], -1, H, W)
        x, H, W = self.downsamplers[1](x, H, W)

        # Stage 2: P4/16
        x = self.stages[2](x)
        p4 = self.norm3(x).transpose(1, 2).view(x.shape[0], -1, H, W)
        x, H, W = self.downsamplers[2](x, H, W)

        # Stage 3: P5/32
        x = self.stages[3](x)
        p5 = self.norm4(x).transpose(1, 2).view(x.shape[0], -1, H, W)

        return [p3, p4, p5]  # [(B,192,H/8,W/8), (B,384,H/16,W/16), (B,768,H/32,W/32)]
