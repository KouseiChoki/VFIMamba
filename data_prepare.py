"""
generate_txt.py
===============
扫描 scenes/ 目录，为每个场景生成所有三元组，
每行记录三帧的完整相对路径（相对于 scenes_dir），空格分隔：

    video_name/scene_0000/000000.png video_name/scene_0000/000001.png video_name/scene_0000/000002.png

用法:
    python generate_txt.py \
        --scenes_dir /data/vfi_dataset/scenes \
        --output_dir /data/vfi_dataset \
        --test_ratio 0.02 \
        --min_scene_frames 3 \
        --seed 42
"""

import argparse
import random
import logging
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def collect_triplets(scenes_dir: Path, min_scene_frames: int) -> List[str]:
    """
    遍历 scenes_dir/<video>/<scene>/，
    对每个场景做滑动窗口（步长1，窗口3），生成所有三元组行。
    每行：  img0_rel img1_rel gt_rel  （均相对 scenes_dir，含后缀）
    """
    triplets: List[str] = []
    video_dirs = sorted(p for p in scenes_dir.iterdir() if p.is_dir())
    logger.info(f"发现 {len(video_dirs)} 个视频目录")

    for vdir in video_dirs:
        scene_dirs = sorted(p for p in vdir.iterdir() if p.is_dir())
        for sdir in scene_dirs:
            frames = sorted(
                p for p in sdir.iterdir()
                if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
            )
            if len(frames) < max(3, min_scene_frames):
                continue

            for i in range(1, len(frames) - 1):
                f0 = frames[i - 1].relative_to(scenes_dir)
                gt = frames[i    ].relative_to(scenes_dir)
                f1 = frames[i + 1].relative_to(scenes_dir)
                triplets.append(f"{f0} {gt} {f1}")

    return triplets


def run(args: argparse.Namespace) -> None:
    scenes_dir = Path(args.scenes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scenes_dir.exists():
        logger.error(f"scenes_dir 不存在: {scenes_dir}")
        return

    logger.info(f"扫描: {scenes_dir}")
    triplets = collect_triplets(scenes_dir, args.min_scene_frames)

    if not triplets:
        logger.error("未找到任何三元组，请检查目录结构")
        return

    logger.info(f"共找到 {len(triplets)} 个三元组")

    random.seed(args.seed)
    random.shuffle(triplets)

    n_test    = max(1, int(len(triplets) * args.test_ratio))
    testlist  = triplets[:n_test]
    trainlist = triplets[n_test:]

    train_fn = output_dir / "tri_trainlist.txt"
    test_fn  = output_dir / "tri_testlist.txt"
    train_fn.write_text("\n".join(trainlist))
    test_fn.write_text("\n".join(testlist))

    logger.info(
        f"✓ 已写入:\n"
        f"  train: {len(trainlist)} 条 → {train_fn}\n"
        f"  test : {len(testlist)}  条 → {test_fn}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 VFIMamba 训练/测试三元组列表 txt")
    p.add_argument("--scenes_dir",        type=str, required=True,
                   help="scenes 根目录")
    p.add_argument("--output_dir",        type=str, required=True,
                   help="txt 文件输出目录")
    p.add_argument("--test_ratio",        type=float, default=0.02,
                   help="测试集比例（默认 0.02）")
    p.add_argument("--min_scene_frames",  type=int,   default=3,
                   help="场景最少帧数（默认 3）")
    p.add_argument("--seed",              type=int,   default=42,
                   help="随机种子（默认 42）")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())


# python data_prepare.py --scenes_dir /home/zhenying/qhong/data/ssd/vfi_train_data/scenes --output_dir /home/zhenying/qhong/repo/VFIMamba/data_rec
