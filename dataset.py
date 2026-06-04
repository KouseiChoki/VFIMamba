"""
dataset.py  —  KouseiDataset for VFIMamba
==========================================
txt 每行三个路径（相对 scenes_dir，空格分隔）：
    video/scene_0000/000000.png  video/scene_0000/000001.png  video/scene_0000/000002.png
    ^^^^ img0 ^^^^               ^^^^^^^^^^^^ gt ^^^^^^^^^^^   ^^^^^^^^^^^^ img1 ^^^^^^^^^^^
"""

import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KouseiDataset(Dataset):
    """
    Parameters
    ----------
    dataset_name : 'train' | 'val' | 'test'
    scenes_dir   : scenes/ 根目录（txt 中路径基于此拼接）
    txt_dir      : tri_trainlist.txt / tri_testlist.txt 所在目录
    crop_h/crop_w: 训练随机裁剪尺寸（默认 1080×1920，即不裁剪全帧）
    train_ratio  : trainlist 中划出多少比例作为 train，其余为 val（默认 0.95）
    """

    def __init__(
        self,
        dataset_name: str,
        scenes_dir: str,
        txt_dir: str,
        crop_h: int = 1080,
        crop_w: int = 1920,
        train_ratio: float = 0.95,
    ):
        self.dataset_name = dataset_name
        self.scenes_dir   = scenes_dir
        self.crop_h       = crop_h
        self.crop_w       = crop_w

        train_fn = os.path.join(txt_dir, "tri_trainlist.txt")
        test_fn  = os.path.join(txt_dir, "tri_testlist.txt")

        with open(train_fn, "r") as f:
            trainlist = f.read().splitlines()
        with open(test_fn, "r") as f:
            testlist = f.read().splitlines()

        cnt = int(len(trainlist) * train_ratio)
        if dataset_name == "train":
            self.meta_data = trainlist[:cnt]
        elif dataset_name == "test":
            self.meta_data = testlist
        else:  # val
            self.meta_data = trainlist[cnt:]

    def __len__(self) -> int:
        return len(self.meta_data)

    # ── 读取 ──────────────────────────────────────────────────────────────────

    def getimg(self, index: int):
        """直接从 txt 行解析三帧路径并读取。"""
        parts = self.meta_data[index].split()   # [img0_rel, gt_rel, img1_rel]
        assert len(parts) == 3, f"格式错误（需3列）: {self.meta_data[index]}"

        path_img0, path_gt, path_img1 = [
            os.path.join(self.scenes_dir, p) for p in parts
        ]

        img0 = cv2.imread(path_img0)[...,::-1]
        gt   = cv2.imread(path_gt)[...,::-1]
        img1 = cv2.imread(path_img1)[...,::-1]

        if img0 is None or gt is None or img1 is None:
            raise FileNotFoundError(
                f"无法读取:\n  {path_img0}\n  {path_gt}\n  {path_img1}"
            )

        return img0, gt, img1, 0.5

    # ── 数据增强 ──────────────────────────────────────────────────────────────

    def crop(self, img0, gt, img1):
        ih, iw = img0.shape[:2]
        h, w   = self.crop_h, self.crop_w
        if ih <= h or iw <= w:
            return img0, gt, img1          # 已是目标尺寸，无需裁剪
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        return (
            img0[x:x+h, y:y+w],
            gt  [x:x+h, y:y+w],
            img1[x:x+h, y:y+w],
        )

    def augment(self, img0, gt, img1, timestep):
        # 随机裁剪
        img0, gt, img1 = self.crop(img0, gt, img1)

        # 旋转 180°（50% 概率）
        if random.random() < 0.5:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt   = cv2.rotate(gt,   cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)

        # 时序翻转：交换 img0/img1（50% 概率）
        if random.random() < 0.5:
            img0, img1 = img1, img0
            timestep   = 1.0 - timestep

        return img0, gt, img1, timestep

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, index: int):
        img0, gt, img1, timestep = self.getimg(index)

        if self.dataset_name == "train":
            img0, gt, img1, timestep = self.augment(img0, gt, img1, timestep)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt   = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)

        return torch.cat((img0, img1, gt), dim=0), timestep


# ─────────────────────────────────────────────────────────────────────────────
# 快速自检
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scenes_dir", required=True)
    p.add_argument("--txt_dir",    required=True)
    p.add_argument("--crop_h",     type=int, default=1080)
    p.add_argument("--crop_w",     type=int, default=1920)
    args = p.parse_args()

    for split in ("train", "val", "test"):
        ds = KouseiDataset(
            dataset_name=split,
            scenes_dir=args.scenes_dir,
            txt_dir=args.txt_dir,
            crop_h=args.crop_h,
            crop_w=args.crop_w,
        )
        print(f"{split:5s}: {len(ds)} 条")
        if len(ds) > 0:
            frames, ts = ds[0]
            print(f"  frames shape : {frames.shape}")   # [9, H, W]
            print(f"  timestep     : {ts.item():.2f}")
