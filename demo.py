import cv2
import math
import sys, os
import torch
import numpy as np
import argparse
from tqdm import tqdm

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from file_utils import read, write
from collections import defaultdict
import glob, re


parser = argparse.ArgumentParser()
parser.add_argument('--model',       default='VFIMamba', type=str)
parser.add_argument('--ckpt',        default='/home/zhenying/qhong/repo/VFIMamba/ckpt/0604/VFIMamba_0.pkl', type=str)
parser.add_argument('--root', '--path', required=True, type=str)
parser.add_argument('--output',      default='/home/zhenying/qhong/result/VfiMamba_result', type=str)
parser.add_argument('--scale',       default=0, type=float)
parser.add_argument('--output_mode', default='image', type=str,
                    choices=['image', 'video', 'both'],
                    help='输出模式: image=只存图, video=只存视频, both=都存')
parser.add_argument('--fps',         default=24, type=float,
                    help='输出视频帧率（仅 video/both 时有效）')
parser.add_argument('--video_ext',   default='mp4', type=str,
                    choices=['mp4', 'avi'],
                    help='视频容器格式（默认 mp4）')
parser.add_argument('--dump_data',action='store_true') 

args = parser.parse_args()
assert args.model in ['VFIMamba_S', 'VFIMamba'], 'Model not exists!'

'''==========Model setting=========='''
TTA = False
if args.model == 'VFIMamba':
    TTA = True
    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 3, 3]
    )
model = Model(-1)
model.load_model(args.ckpt)
model.eval()
model.device()

# ── resize 上限 ───────────────────────────────────────────────────────
MAX_W, MAX_H = 1920, 1080

def cap_size(w, h):
    scale = min(MAX_W / w, MAX_H / h)
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    return w, h

def resize_if_needed(img):
    h, w = img.shape[:2]
    cw, ch = cap_size(w, h)
    if (cw, ch) != (w, h):
        img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
    return img

def to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

def to_uint8(img_f32):
    return (img_f32 * 255).clip(0, 255).astype(np.uint8)


# ── VideoWriter 管理 ─────────────────────────────────────────────────
class SequenceVideoWriter:
    def __init__(self, fps, ext):
        self.fps    = fps
        self.ext    = ext
        self.fourcc = (cv2.VideoWriter_fourcc(*'mp4v') if ext == 'mp4'
                       else cv2.VideoWriter_fourcc(*'XVID'))
        self._frames = defaultdict(list)  # save_folder -> [(idx, bgr), ...]

    def add_frame(self, save_folder: str, frame_idx: int, img_f32: np.ndarray):
        bgr = to_uint8(img_f32)[..., ::-1].copy()
        self._frames[save_folder].append((frame_idx, bgr))

    def flush(self):
        for folder, frame_list in self._frames.items():
            frame_list.sort(key=lambda x: x[0])
            h, w = frame_list[0][1].shape[:2]
            video_path = os.path.join(folder, f'output.{self.ext}')
            writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, (w, h))
            if not writer.isOpened():
                print(f'[VideoWriter] 无法创建视频: {video_path}')
                continue
            for _, bgr in frame_list:
                writer.write(bgr)
            writer.release()
            print(f'[VideoWriter] 已保存: {video_path}  ({len(frame_list)} 帧)')
        self._frames.clear()


# ── 初始化 ────────────────────────────────────────────────────────────
need_image = args.output_mode in ('image', 'both')
need_video = args.output_mode in ('video', 'both')
video_writer = SequenceVideoWriter(args.fps, args.video_ext) if need_video else None

print('=========================Start Generating=========================')

def extract_number(x):
    nums = re.findall(r'(\d+)', os.path.basename(x))
    return int(nums[-1]) if nums else -1

root     = args.root
out_root = args.output

# ── 收集所有序列和 pair ───────────────────────────────────────────────
exts     = ["*.png", "*.exr", "*.tif"]
seq_dict = defaultdict(list)

for ext in exts:
    for f in glob.glob(os.path.join(root, "**", ext), recursive=True):
        folder = os.path.dirname(f)
        seq_dict[folder].append(f)

# 按 save_folder 分组构建 pairs，同时记录每组最后一个 pair 的全局 index
pairs          = []
last_index_of  = {}   # save_folder -> 该序列最后一个 pair 在 pairs 中的 index

for folder, files in seq_dict.items():
    files = sorted(files, key=extract_number)
    rel_folder  = os.path.relpath(folder, root)
    save_folder = os.path.join(out_root, rel_folder)
    for i in range(len(files) - 1):
        pairs.append((files[i], files[i + 1], save_folder))
        last_index_of[save_folder] = len(pairs) - 1

# ── 主循环 ───────────────────────────────────────────────────────────
for index in tqdm(range(len(pairs))):
    src0, src1, save_folder = pairs[index]
    ext    = os.path.splitext(src0)[1]
    is_exr = ext.lower() in ('.exr', '.tif')
    is_last = (index == last_index_of[save_folder])

    os.makedirs(save_folder, exist_ok=True)

    # ── 读取 & 预处理 ────────────────────────────────────────────────
    I0 = to_float(resize_if_needed(read(src0, type='image')))
    I2 = to_float(resize_if_needed(read(src1, type='image')))

    I0_ = torch.tensor(I0.transpose(2, 0, 1)).cuda().unsqueeze(0)
    I2_ = torch.tensor(I2.transpose(2, 0, 1)).cuda().unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)

    # ── 推理 ─────────────────────────────────────────────────────────
    with torch.no_grad():
        # if not args.dump_data:
        mid,flow,mask,merged,res,warp0,warp1= model.inference(I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=args.scale)
        # else:
            # mid, warp0, warp1, flow, mask = model.hr_inference(
                # I0_, I2_, True, TTA=False, fast_TTA=False)
    mid = padder.unpad(mid)[0].detach().cpu().numpy().transpose(1, 2, 0)

    # ── 命名 ─────────────────────────────────────────────────────────
    idx0    = extract_number(src0) * 2
    idx1    = extract_number(src1) * 2
    mid_idx = (idx0 + idx1) // 2

    # ── 每个 pair 只负责写 img0 + mid；最后一个 pair 额外写 img1 ────
    frames_to_write = [(idx0, I0), (mid_idx, mid)]
    if is_last:
        frames_to_write.append((idx1, I2))

    for frame_idx, img_f32 in frames_to_write:
        if need_image:
            out_path = os.path.join(save_folder, f"{frame_idx:06d}{ext}")
            if is_exr:
                write(out_path, img_f32)
            else:
                write(out_path, to_uint8(img_f32))

        if need_video:
            video_writer.add_frame(save_folder, frame_idx, img_f32)

    # ── args.dump_data 调试输出 ─────────────────────────────────────────────
    if args.dump_data:
        warp0 = padder.unpad(warp0)[0].detach().cpu().numpy().transpose(1, 2, 0)
        warp1 = padder.unpad(warp1[0]).detach().cpu().numpy().transpose(1, 2, 0)
        flow  = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
        mv0, mv1 = flow[..., :2], flow[..., 2:]
        mask  = padder.unpad(mask)[0].detach().cpu().numpy().transpose(1, 2, 0)
        merged  = padder.unpad(merged)[0].detach().cpu().numpy().transpose(1, 2, 0)
        res  = padder.unpad(res)[0].detach().cpu().numpy().transpose(1, 2, 0)
        write(os.path.join(save_folder, 'warp0', f"{mid_idx:06d}{ext}"), warp0)
        write(os.path.join(save_folder, 'warp1', f"{mid_idx:06d}{ext}"), warp1)
        write(os.path.join(save_folder, 'mv0',   f"{mid_idx:06d}.exr"),  mv0)
        write(os.path.join(save_folder, 'mv1',   f"{mid_idx:06d}.exr"),  mv1)
        write(os.path.join(save_folder, 'mask',  f"{mid_idx:06d}.exr"),  mask)
        write(os.path.join(save_folder, 'merged',  f"{mid_idx:06d}{ext}"),  merged)
        write(os.path.join(save_folder, 'res',  f"{mid_idx:06d}.exr"),  res)

# ── 写出视频 ─────────────────────────────────────────────────────────
if need_video:
    video_writer.flush()

print('=========================Done=========================')