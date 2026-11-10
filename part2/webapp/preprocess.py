"""preprocess.py — 把任意图片整理成 MNIST 风格的 28×28 输入

训练脚本里用的是"直接缩放到 28×28 + 按边缘亮度判断是否反色"，对规整的截图够用，
但对手机拍照、随手画的图就不稳了：笔画的位置、大小、粗细都可能和 MNIST 差很远。
这里在缩放之前多做两步，让输入更接近 MNIST 的分布：

    1. 裁到笔画的紧贴边框（去掉大片空白）
    2. 按长边补成正方形，缩放到 20×20，再按重心平移进 28×28 的中心

这正是 LeCun 当年整理 MNIST 的做法（20×20 的框内按重心居中），所以对潦草的手写
和拍照都更鲁棒。
"""

import numpy as np
from PIL import Image

MEAN = 0.1307
STD = 0.3081

_MARGIN = 4          # 20×20 的数字放进 28×28 时四周留的空白
_INK = 0.35          # 归一化后判定为"笔画"的相对阈值


class EmptyImageError(ValueError):
    """图里没找到笔画，或者对比度太低"""


def _to_gray(img: Image.Image) -> Image.Image:
    """统一转灰度：带透明通道的先合到白底上，避免透明区域被当成黑色"""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.alpha_composite(img)
        img = background
    return img.convert("L")


def _is_light_background(arr: np.ndarray) -> bool:
    """取四条边的平均亮度判断底色：亮底（白纸、白画板）需要反色"""
    edges = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(edges.mean()) > 0.5


def _bounding_box(mask: np.ndarray):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _pad_square(arr: np.ndarray) -> np.ndarray:
    """把矩形裁片补成正方形，数字居中"""
    h, w = arr.shape
    side = max(h, w)
    out = np.zeros((side, side), dtype=np.float32)
    top = (side - h) // 2
    left = (side - w) // 2
    out[top:top + h, left:left + w] = arr
    return out


def _center_of_mass(arr: np.ndarray):
    total = float(arr.sum())
    if total <= 1e-6:
        return None
    ys, xs = np.indices(arr.shape)
    return float((arr * ys).sum() / total), float((arr * xs).sum() / total)


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """整体平移，空出来的部分补 0，不用 roll 以免环绕"""
    out = np.zeros_like(arr)
    h, w = arr.shape
    ys_src = slice(max(0, -dy), h - max(0, dy))
    ys_dst = slice(max(0, dy), h - max(0, -dy))
    xs_src = slice(max(0, -dx), w - max(0, dx))
    xs_dst = slice(max(0, dx), w - max(0, -dx))
    out[ys_dst, xs_dst] = arr[ys_src, xs_src]
    return out


def prepare(img: Image.Image) -> dict:
    """返回 dict: view(28×28 uint8)、tensor(float32 1×1×28×28)、bbox、inverted

    bbox 是在原图坐标里找到的笔画边框，前端可以拿它画提示。
    """
    gray = _to_gray(img)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    inverted = _is_light_background(arr)
    if inverted:
        arr = 1.0 - arr

    low, high = float(arr.min()), float(arr.max())
    if high - low < 0.08:
        raise EmptyImageError("图片对比度太低，没找到可以识别的笔画")

    # 拉伸到 0..1，抵消光照和曝光的影响
    arr = (arr - low) / (high - low)

    mask = arr > _INK
    if mask.mean() < 0.004:
        raise EmptyImageError("没有检测到笔画，换一张图或写粗一点再试")

    box = _bounding_box(mask)
    if box is None:
        raise EmptyImageError("没有检测到笔画")

    top, bottom, left, right = box
    if bottom - top < 6 or right - left < 6:
        raise EmptyImageError("笔画太小了，换一张更清晰的图")

    crop = arr[top:bottom + 1, left:right + 1]
    square = _pad_square(crop)

    # 缩到 20×20（MNIST 的内容框就是 20×20）
    side = 20
    small = np.asarray(
        Image.fromarray((square * 255).astype(np.uint8)).resize(
            (side, side), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    ) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[_MARGIN:_MARGIN + side, _MARGIN:_MARGIN + side] = small

    # 按重心把数字推到画布中心
    com = _center_of_mass(canvas)
    if com is not None:
        cy, cx = com
        dy = int(round(14.0 - cy))
        dx = int(round(14.0 - cx))
        if dy or dx:
            canvas = _shift(canvas, dy, dx)

    view = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    return {
        "view": view,
        "bbox": (left, top, right, bottom),
        "size": gray.size,
        "inverted": inverted,
    }


def to_tensor(view: np.ndarray):
    """28×28 的 uint8 灰度图 → 归一化后的 (1,1,28,28) 张量"""
    import torch

    arr = view.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
