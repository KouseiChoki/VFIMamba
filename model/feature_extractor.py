"""
mamba_extractor.py  —  升级版
================================
在原始 VFIMamba mamba_extractor 基础上做两项升级：

1. Mamba2（SSD）算子替换 Mamba1 selective_scan
   - 使用 mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined
   - 更高的硬件利用率，更低显存，训练更快

2. LC-Mamba 扫描策略替换原始 4 方向 SS2D
   - SW-H-SS2D：Hilbert 曲线扫描 + 移位局部窗口，保持窗口边界空间连续性
   - H-SS3D 思路：在双帧 merge_x 时沿时序维度融合，捕捉帧间时空特征

依赖:
    pip install mamba-ssm>=2.0.0 causal-conv1d einops timm
"""

import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from typing import Callable
from functools import partial
import torch.nn.functional as F

# ── Mamba2 SSD 算子 ──────────────────────────────────────────────────────────
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    USE_MAMBA2 = True
except ImportError:
    # 回退到 Mamba1 selective_scan
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    USE_MAMBA2 = False
    import warnings
    warnings.warn(
        "mamba_ssm >= 2.0.0 未找到，回退到 Mamba1 selective_scan。\n"
        "建议: pip install mamba-ssm>=2.0.0"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hilbert 曲线工具
# ─────────────────────────────────────────────────────────────────────────────

def _hilbert_d2xy(n: int, d: int):
    """将 Hilbert 曲线距离 d 转为 (x, y)，n 为阶数（边长）。"""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if (d & 1) ^ rx else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def build_hilbert_index(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    生成将 H×W 特征图展平成 Hilbert 曲线顺序的索引。
    返回 shape [H*W]，值为原始 HW 展平后的位置索引。
    """
    # 找到能覆盖 H×W 的最小 2 次幂边长
    n = 1
    while n < max(H, W):
        n <<= 1

    coords = []
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if x < H and y < W:
            coords.append(x * W + y)

    # 如果 H×W 不是 2^k × 2^k，coords 长度可能 < H*W，补齐剩余
    visited = set(coords)
    for i in range(H * W):
        if i not in visited:
            coords.append(i)

    return torch.tensor(coords[:H * W], dtype=torch.long, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助模块（与原始保持一致）
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


# ─────────────────────────────────────────────────────────────────────────────
# Mamba2 SSD 核心（替换原 SS2D.forward_core）
# ─────────────────────────────────────────────────────────────────────────────

class SSD2D(nn.Module):
    """
    Mamba2（SSD）版本的 2D 状态空间扫描模块。
    扫描策略：LC-Mamba 的 SW-H-SS2D
      - 将特征图按 Hilbert 曲线重排后再做双向扫描
      - 移位局部窗口：交替 shift，保持窗口间连续性
      - 双帧融合：沿时序维度 merge（H-SS3D 思路）
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,       # Mamba2 推荐 d_state=64~256
        d_conv: int = 4,
        expand: float = 2.0,
        headdim: int = 64,       # Mamba2 新增：每个 head 的维度
        chunk_size: int = 256,   # Mamba2 chunk scan 大小
        window_size: int = 8,    # 局部窗口大小（LC-Mamba）
        shift: bool = False,     # 是否移位（SW）
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model   = d_model
        self.d_state   = d_state
        self.d_inner   = int(expand * d_model)
        self.headdim   = headdim
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.shift     = shift
        self.nheads    = self.d_inner // headdim
        self.d_conv = d_conv

        # 投影层
        # Mamba2: x_proj 输出 z + x + B + C + dt
        d_in_proj = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj  = nn.Linear(d_model, d_in_proj, bias=bias)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
            groups=self.d_inner, bias=conv_bias,
        )
        self.act      = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else None

        # Mamba2 可学习参数
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32))
        )
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))

        # 缓存 Hilbert 索引（按 H,W 动态生成）
        self._hilbert_cache: dict = {}

    def _get_hilbert_idx(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W)
        if key not in self._hilbert_cache:
            self._hilbert_cache[key] = build_hilbert_index(H, W, device)
        return self._hilbert_cache[key].to(device)

    def _apply_shift(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """移位：将特征图循环平移 window_size//2，实现 SW（Shifted Window）。"""
        if not self.shift:
            return x
        shift_h = self.window_size // 2
        shift_w = self.window_size // 2
        return torch.roll(x, shifts=(-shift_h, -shift_w), dims=(2, 3))

    def _undo_shift(self, x: torch.Tensor) -> torch.Tensor:
        if not self.shift:
            return x
        shift_h = self.window_size // 2
        shift_w = self.window_size // 2
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))

    def merge_x_hilbert(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        双帧 Hilbert 融合（H-SS3D 思路）：
        img0/img1 特征在 batch 维度拼接 [2B, C, H, W]
        → 按 Hilbert 曲线重排空间维度
        → 沿时序维度交织合并，得到 [B, C, 2*H*W]
        """
        B2, C, H_, W_ = x.shape   # B2 = 2*B
        B = B2 // 2
        idx = self._get_hilbert_idx(H_, W_, x.device)  # [H*W]

        # Hilbert 重排：[2B, C, H*W] → 按 idx 排序
        x_flat = x.view(B2, C, H_ * W_)                # [2B, C, L]
        x_hil  = x_flat[:, :, idx]                      # Hilbert 顺序

        # 时序维度融合：img0/img1 交织 → [B, C, 2L]
        x0 = x_hil[:B]   # [B, C, L]
        x1 = x_hil[B:]   # [B, C, L]
        merged = torch.stack([x0, x1], dim=3)           # [B, C, L, 2]
        return merged.view(B, C, -1).contiguous()       # [B, C, 2L]

    def forward_core_mamba2(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B2, C, H_, W_ = x.shape
        B = B2 // 2
        L = H_ * W_       # 单帧序列长度
        L_total = 2 * L   # 双帧融合后总长度

        # ── Hilbert + 时序融合 → [B, L_total, C] ──
        seq = self.merge_x_hilbert(x, H_, W_)           # [B, C, L_total]
        seq = seq.permute(0, 2, 1).contiguous()         # [B, L_total, C]

        outs = []
        for direction in [seq, seq.flip(1)]:
            proj = self.in_proj(direction)              # [B, L_total, d_in_proj]
            z, x_ssm, B_ssm, C_ssm, dt = proj.split(
                [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads],
                dim=-1,
            )

            x_ssm_t = x_ssm.permute(0, 2, 1).contiguous()          # [B, d_inner, L_total]
            x_ssm_t = F.pad(x_ssm_t, (self.d_conv - 1, 0))         # 左 pad
            x_ssm_t = self.act(self.conv1d(x_ssm_t))               # 输出可能比 L_total 短或长

            # 强制对齐到 L_total
            if x_ssm_t.shape[-1] < L_total:
                # 输出比预期短，右侧补零
                x_ssm_t = F.pad(x_ssm_t, (0, L_total - x_ssm_t.shape[-1]))
            elif x_ssm_t.shape[-1] > L_total:
                # 输出比预期长，截断
                x_ssm_t = x_ssm_t[..., :L_total]

            x_ssm = x_ssm_t.permute(0, 2, 1).contiguous()          # [B, L_total, d_inner]
            # 验证维度整除性
            assert x_ssm.shape[-1] == self.nheads * self.headdim, \
                f"d_inner={x_ssm.shape[-1]} 不能被 nheads={self.nheads} × headdim={self.headdim} 整除"
            
            x_h = x_ssm.view(B, L_total, self.nheads, self.headdim)
            B_h = B_ssm.view(B, L_total, 1, self.d_state)
            C_h = C_ssm.view(B, L_total, 1, self.d_state)
            A   = -torch.exp(self.A_log.float())

            if USE_MAMBA2:
                y = mamba_chunk_scan_combined(
                    x_h, dt, A, B_h, C_h,
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                )                                              # [B, L_total, nheads, headdim]
            else:
                # Mamba1 fallback
                xs_1  = x_h.reshape(B, L_total, self.d_inner).permute(0, 2, 1)
                dts_1 = dt.permute(0, 2, 1).repeat_interleave(self.headdim, dim=1)
                Bs_1  = B_ssm.permute(0, 2, 1).unsqueeze(1)
                Cs_1  = C_ssm.permute(0, 2, 1).unsqueeze(1)
                As_1  = torch.diag_embed(
                    A.unsqueeze(1).expand(-1, self.headdim).reshape(-1)
                ).unsqueeze(-1).expand(-1, -1, self.d_state)
                y_1 = selective_scan_fn(
                    xs_1.float(), dts_1.float(), As_1.float(),
                    Bs_1.float(), Cs_1.float(),
                    self.D.repeat_interleave(self.headdim).float(),
                    z=None,
                    delta_bias=self.dt_bias.repeat_interleave(self.headdim).float(),
                    delta_softplus=True, return_last_state=False,
                )
                y = y_1.permute(0, 2, 1).view(B, L_total, self.nheads, self.headdim)

            y = y.reshape(B, L_total, self.d_inner)
            y = y * F.silu(z)
            outs.append(y)

        y_bi = outs[0] + outs[1].flip(1)                     # [B, L_total, d_inner]

        # ── 还原 ──
        y_bi = y_bi.view(B, L, 2, self.d_inner)
        y0   = y_bi[:, :, 0, :].permute(0, 2, 1)             # [B, d_inner, L]
        y1   = y_bi[:, :, 1, :].permute(0, 2, 1)

        idx     = self._get_hilbert_idx(H_, W_, x.device)
        inv_idx = torch.argsort(idx)
        y0 = y0[:, :, inv_idx].view(B, self.d_inner, H_, W_)
        y1 = y1[:, :, inv_idx].view(B, self.d_inner, H_, W_)
        return torch.cat([y0, y1], dim=0)                    # [2B, d_inner, H, W]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [2B, H, W, C]  (BHWC 格式，与原始 VSSBlock 接口一致)
        返回: [2B, H, W, C]
        """
        B2, H, W, C = x.shape

        # 移位（SW）
        x_nchw = x.permute(0, 3, 1, 2).contiguous()   # [2B, C, H, W]
        x_nchw = self._apply_shift(x_nchw, H, W)

        # 核心扫描
        y_nchw = self.forward_core_mamba2(x_nchw, H, W)  # [2B, d_inner, H, W]

        # 还原移位
        y_nchw = self._undo_shift(y_nchw)

        # out_norm + out_proj
        y = y_nchw.permute(0, 2, 3, 1).contiguous()   # [2B, H, W, d_inner]
        y = self.out_norm(y)
        y = self.out_proj(y)                           # [2B, H, W, C]

        if self.dropout is not None:
            y = self.dropout(y)
        return y


# ─────────────────────────────────────────────────────────────────────────────
# VSSBlock：用 SSD2D 替换原 SS2D，其余结构不变
# ─────────────────────────────────────────────────────────────────────────────

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        attn_drop_rate: float = 0,
        d_state: int = 64,
        mlp_ratio: float = 2.0,
        headdim: int = 64,
        window_size: int = 8,
        shift: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SSD2D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=mlp_ratio,
            dropout=attn_drop_rate,
            headdim=headdim,
            window_size=window_size,
            shift=shift,
        )
        self.skip_scale  = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk    = CAB(hidden_dim)
        self.ln_2        = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [2B, C, H, W]
        x = input.permute(0, 2, 3, 1).contiguous()    # [2B, H, W, C]
        x_ln = self.ln_1(x)
        x = x * self.skip_scale + self.self_attention(x_ln)
        x = x * self.skip_scale2 + self.conv_blk(
            self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1).contiguous()
        return x.permute(0, 3, 1, 2).contiguous()     # [2B, C, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# BiMambaBlock：交替 shift/non-shift（SW-H-SS2D 的核心循环）
# ─────────────────────────────────────────────────────────────────────────────

class BiMambaBlock(nn.Module):
    """
    LC-Mamba 的移位局部窗口策略：
    奇数层 shift=False，偶数层 shift=True，交替出现。
    保证相邻窗口之间的信息能够跨边界传播。
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        norm_layer=nn.LayerNorm,
        window_size: int = 8,
        headdim: int = 64,
        d_state: int = 64,
    ):
        super().__init__()
        self.dim   = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                norm_layer=nn.LayerNorm,
                d_state=d_state,
                headdim=min(headdim, dim),    # headdim 不能超过 d_inner
                window_size=window_size,
                shift=(i % 2 == 1),           # 奇数层 shift
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 其余模块与原始完全一致
# ─────────────────────────────────────────────────────────────────────────────

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(in_chans)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        return self.proj(x)


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2, act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            layers.append(nn.Conv2d(in_dim if i == 0 else out_dim, out_dim, 3, 1, 1))
            layers.append(act_layer(out_dim))
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# MambaFeature：顶层 backbone，接口与原始完全兼容
# ─────────────────────────────────────────────────────────────────────────────

class MambaFeature(nn.Module):
    """
    多尺度特征提取 backbone。
    前 conv_stages 层用 ConvBlock，之后用升级的 BiMambaBlock（SSD2D + Hilbert）。
    输入/输出接口与原始 VFIMamba mamba_extractor 完全一致。
    """
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dims=[16, 32, 64, 128, 256],
        depths=(2, 2, 2, 2, 2),
        conv_stages=2,
        norm_layer=nn.LayerNorm,
        window_size=8,
        headdim=32,
        d_state=64,
        **kwargs,
    ):
        super().__init__()
        self.num_stages = len(embed_dims)
        self.conv_stages = conv_stages

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans, embed_dims[i], depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3, 2, 1),
                        nn.PReLU(embed_dims[i]),
                    )
                    block = ConvBlock(embed_dims[i], embed_dims[i], depths[i])
                else:
                    patch_embed = OverlapPatchEmbed(
                        patch_size=3, stride=2,
                        in_chans=embed_dims[i-1], embed_dim=embed_dims[i],
                    )
                    block = BiMambaBlock(
                        dim=embed_dims[i],
                        depth=depths[i],
                        window_size=window_size,
                        headdim=min(headdim, embed_dims[i]),
                        d_state=d_state,
                    )
                setattr(self, f"patch_embed{i}", patch_embed)
            setattr(self, f"block{i}", block)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        x1, x2: [B, 3, H, W]
        返回: list of [2B, C_i, H_i, W_i]，与原始接口完全一致
        """
        x = torch.cat([x1, x2], dim=0)   # [2B, 3, H, W]
        features = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i}", None)
            block       = getattr(self, f"block{i}", None)
            if i > 0:
                x = patch_embed(x)
            x = block(x)
            features.append(x)

        return features


def feature_extractor(**kargs):
    model = MambaFeature(**kargs)
    return model
