"""
border_mask.py
==============
训练时动态检测 film border（纯黑/纯色边缘条带），
生成有效区域 mask，让 border 区域不参与 loss 计算。

检测原理：
  1. 对 gt 帧沿行/列求均值亮度
  2. 从四个边缘向内找到第一个"非暗"行/列（亮度超过阈值）
  3. 生成 [B, 1, H, W] 的 0/1 mask，border 区域为 0

使用示例：
    from border_mask import compute_border_mask, masked_loss

    mask = compute_border_mask(gt)                    # [B,1,H,W]
    loss = masked_loss(pred, gt, mask, loss_fn)
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional


def compute_border_mask(
    img: torch.Tensor,
    dark_thresh: float = 0.05,
    min_valid_ratio: float = 0.5,
    smooth_k: int = 5,
) -> torch.Tensor:
    """
    检测 film border 并返回有效区域 mask。

    Parameters
    ----------
    img : [B, C, H, W]  float 0~1
        用于检测的图像（通常传 gt）
    dark_thresh : float
        行/列均值亮度低于此值视为 border（默认 0.05，即 ~13/255）
    min_valid_ratio : float
        有效区域面积占全图的最低比例，低于此比例则认为检测失败，
        返回全 1 mask（避免误检导致无有效区域）
    smooth_k : int
        对亮度曲线做均值平滑的窗口大小，抑制单行噪声误检

    Returns
    -------
    mask : [B, 1, H, W]  float 0/1，border 区域为 0，有效区域为 1
    """
    B, C, H, W = img.shape
    device = img.device

    # 转灰度亮度：[B, H, W]
    if C == 3:
        luma = 0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]
    else:
        luma = img.mean(dim=1)

    masks = []
    for b in range(B):
        luma_b = luma[b]  # [H, W]

        # 行均值 [H]，列均值 [W]
        row_mean = luma_b.mean(dim=1)   # 沿 W 平均
        col_mean = luma_b.mean(dim=0)   # 沿 H 平均

        # 均值平滑，抑制噪声
        if smooth_k > 1:
            pad = smooth_k // 2
            row_mean = F.avg_pool1d(
                row_mean.view(1, 1, -1), smooth_k, stride=1, padding=pad
            ).view(-1)
            col_mean = F.avg_pool1d(
                col_mean.view(1, 1, -1), smooth_k, stride=1, padding=pad
            ).view(-1)

        # 从四边向内找有效起止索引
        row_valid = (row_mean > dark_thresh)
        col_valid = (col_mean > dark_thresh)

        # top / bottom
        top    = _first_true(row_valid,          H)
        bottom = _first_true(row_valid.flip(0),  H)
        bottom = H - bottom

        # left / right
        left  = _first_true(col_valid,          W)
        right = _first_true(col_valid.flip(0),  W)
        right = W - right

        # 检测失败保护：有效区域太小时退化为全 1
        valid_h = max(0, bottom - top)
        valid_w = max(0, right  - left)
        if valid_h * valid_w < min_valid_ratio * H * W:
            masks.append(torch.ones(1, H, W, device=device))
            continue

        mask_b = torch.zeros(1, H, W, device=device)
        mask_b[:, top:bottom, left:right] = 1.0
        masks.append(mask_b)

    return torch.stack(masks, dim=0)   # [B, 1, H, W]


def _first_true(bools: torch.Tensor, length: int) -> int:
    """返回第一个 True 的索引，全 False 时返回 0。"""
    idx = bools.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return 0
    return int(idx[0, 0].item())


# ─────────────────────────────────────────────────────────────────────────────
# masked loss
# ─────────────────────────────────────────────────────────────────────────────

def masked_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """mask 加权的 L1 loss，只计算 mask=1 的区域。"""
    diff = (pred - gt).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def masked_lap_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    lap_loss_fn,
) -> torch.Tensor:
    """
    对 LapLoss 的每个金字塔层应用 mask（下采样 mask 匹配尺寸）。
    """
    from model.loss import laplacian_pyramid
    kernel = lap_loss_fn.gauss_kernel

    pyr_pred   = laplacian_pyramid(pred,   kernel, lap_loss_fn.max_levels)
    pyr_target = laplacian_pyramid(gt,     kernel, lap_loss_fn.max_levels)

    total = torch.tensor(0.0, device=pred.device)
    mask_curr = mask.float()

    for a, b in zip(pyr_pred, pyr_target):
        # 下采样 mask 到当前金字塔层尺寸
        if mask_curr.shape[-2:] != a.shape[-2:]:
            mask_curr = F.interpolate(
                mask_curr, size=a.shape[-2:], mode='nearest'
            )
        diff  = (a - b).abs() * mask_curr
        denom = mask_curr.sum().clamp(min=1.0)
        total = total + diff.sum() / denom

    return total


def masked_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
    lap_loss_fn=None,
) -> torch.Tensor:
    """
    统一入口。

    Parameters
    ----------
    pred, gt  : [B, 3, H, W] float 0~1
    mask      : [B, 1, H, W] float 0/1（由 compute_border_mask 生成）
    loss_type : 'l1' | 'lap' | 'l1+lap'
    lap_loss_fn : LapLoss 实例（loss_type 含 'lap' 时必须传入）
    """
    if loss_type == 'l1':
        return masked_l1(pred, gt, mask)
    elif loss_type == 'lap':
        return masked_lap_loss(pred, gt, mask, lap_loss_fn)
    elif loss_type == 'l1+lap':
        return (
            0.5 * masked_l1(pred, gt, mask)
            + 0.5 * masked_lap_loss(pred, gt, mask, lap_loss_fn)
        )
    else:
        raise ValueError(f"未知 loss_type: {loss_type}")
