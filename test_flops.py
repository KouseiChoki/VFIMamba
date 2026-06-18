import cv2
import math
import sys, os
import torch
import numpy as np
import argparse
from fvcore.nn import FlopCountAnalysis, flop_count_table

sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VFIMamba', type=str)
parser.add_argument('--ckpt', default='/home/zhenying/qhong/repo/VFIMamba/ckpt/VFIMamba_0.pkl', type=str)
parser.add_argument('--width',  default=2560, type=int, help='测试宽度，默认 2K')
parser.add_argument('--height', default=1440, type=int, help='测试高度，默认 2K')
parser.add_argument('--old_version', action='store_true')
args = parser.parse_args()

assert args.model in ['VFIMamba_S', 'VFIMamba']

# ── 模型初始化 ────────────────────────────────────────────────────────
cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])
cfg.MODEL_CONFIG['MODEL_ARCH'][1]['version'] = 1 if args.old_version else 2

model = Model(-1)
model.load_model(args.ckpt)
model.eval()
model.device()
dev = model._dev

print(f"Device: {dev}")
print(f"Input resolution: {args.width} x {args.height}")

# ── 构造 2K dummy 输入 ────────────────────────────────────────────────
H, W = args.height, args.width
dummy_I0 = torch.zeros(1, 3, H, W, device=dev)
dummy_I2 = torch.zeros(1, 3, H, W, device=dev)

padder = InputPadder(dummy_I0.shape, divisor=32)
dummy_I0, dummy_I2 = padder.pad(dummy_I0, dummy_I2)
pH, pW = dummy_I0.shape[2], dummy_I0.shape[3]
print(f"Padded resolution: {pW} x {pH}\n")


# ── 各模块 FLOPs 测试函数 ─────────────────────────────────────────────
def measure(name, module, inputs, max_depth=4):
    print(f"{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(module, inputs)
            flops.unsupported_ops_settings(raise_on_error=False)
            total = flops.total()
            print(f"  Total: {total/1e9:.3f} GFLOPs  ({total/1e12:.4f} TFLOPs)")
            print(flop_count_table(flops, max_depth=max_depth))
    except Exception as e:
        print(f"  [FAILED] {e}")
    print()


# ── 1. feature_bone ───────────────────────────────────────────────────
measure(
    "feature_bone (MambaFeature)",
    model.net.feature_bone,
    (dummy_I0, dummy_I2),
    max_depth=4,
)

# ── 2. 整个 flow_estimation 网络 ──────────────────────────────────────
# flow_estimation.forward 签名通常是 (imgs, timestep, scale, ...)
# 根据你的实际接口调整
dummy_imgs = torch.cat([dummy_I0, dummy_I2], dim=1)  # [1, 6, H, W]
dummy_timestep = torch.tensor([0.5], device=dev)

measure(
    "flow_estimation (full net)",
    model.net,
    (dummy_imgs, dummy_timestep),
    max_depth=3,
)

# ── 3. 逐 stage 测 feature_bone ──────────────────────────────────────
print("="*60)
print("  feature_bone per-stage breakdown")
print("="*60)

fb = model.net.feature_bone
x = torch.cat([dummy_I0, dummy_I2], dim=0)  # [2, 3, H, W]

for i in range(fb.num_stages):
    patch_embed = getattr(fb, f"patch_embed{i}", None)
    block       = getattr(fb, f"block{i}", None)

    # 先跑一次得到 stage 输入
    with torch.no_grad():
        if i > 0 and patch_embed is not None:
            x_in = patch_embed(x)
        else:
            x_in = x

    stage_name = f"stage{i} - {'ConvBlock' if i < fb.conv_stages else 'BiMambaBlock'}"

    if i > 0 and patch_embed is not None:
        measure(
            f"  {stage_name} / patch_embed",
            patch_embed,
            (x,),
            max_depth=2,
        )

    measure(
        f"  {stage_name} / block",
        block,
        (x_in,),
        max_depth=3,
    )

    with torch.no_grad():
        x = block(x_in)

print("Done.")