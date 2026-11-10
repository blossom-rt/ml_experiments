"""app.py — 手写数字识别模型的网页演示

启动（在 part2/webapp 目录下，先激活装了 PyTorch 的环境）：

    python app.py

然后用浏览器打开 http://127.0.0.1:5000

页面提供两种输入方式：上传本地图片、在画板上用鼠标/手指写一个数字。
"""

import base64
import io
import os
import random
import time

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from model import load_model
from preprocess import EmptyImageError, prepare, to_tensor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL, DEVICE = load_model()

# 先跑一次空输入，把 CUDA 的首次初始化开销留在启动阶段，用户点第一次不必等
with torch.no_grad():
    MODEL(torch.zeros(1, 1, 28, 28, device=DEVICE))

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAW = os.path.join(os.path.dirname(_HERE), "data", "MNIST", "raw")


def _png_data_url(view, scale=4):
    """把 28×28 放大成整齐的像素块，方便前端直接显示"""
    img = Image.fromarray(view, mode="L").resize(
        (view.shape[1] * scale, view.shape[0] * scale), Image.Resampling.NEAREST
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_test_set():
    """内存映射 MNIST 测试集，用于"试一个样本"，不占常驻内存"""
    images = os.path.join(_RAW, "t10k-images-idx3-ubyte")
    labels = os.path.join(_RAW, "t10k-labels-idx1-ubyte")
    if not (os.path.exists(images) and os.path.exists(labels)):
        return None
    x = np.memmap(images, dtype=np.uint8, mode="r", offset=16).reshape(-1, 28, 28)
    y = np.memmap(labels, dtype=np.uint8, mode="r", offset=8)
    return x, y


TEST_X, TEST_Y = None, None
try:
    TEST_X, TEST_Y = _load_test_set()
except Exception as exc:
    print(f"测试集未加载（跳过示例功能）：{exc}")


def _read_image():
    """支持两种提交方式：multipart 文件，或表单里的 dataURL"""
    upload = request.files.get("file")
    if upload is not None and upload.filename:
        return Image.open(io.BytesIO(upload.read())), upload.filename

    raw = request.form.get("image", "")
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    if raw:
        return Image.open(io.BytesIO(base64.b64decode(raw))), "canvas"

    raise ValueError("没有收到图片")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        ok=True,
        device=str(DEVICE),
        gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "",
        torch=torch.__version__,
    )


@app.get("/api/example")
def api_example():
    """返回一张测试集样本，方便没有现成图片时试一下"""
    if TEST_X is None:
        return jsonify(ok=False, error="本地没有 MNIST 测试集文件"), 404

    want = request.args.get("label", type=int)
    if want is None:
        idx = random.randrange(len(TEST_X))
    else:
        candidates = np.flatnonzero(TEST_Y == want)
        if candidates.size == 0:
            return jsonify(ok=False, error=f"没有标签为 {want} 的样本"), 404
        idx = int(random.choice(candidates))

    view = np.asarray(TEST_X[idx], dtype=np.uint8)
    return jsonify(
        ok=True,
        image=_png_data_url(view, scale=8),
        label=int(TEST_Y[idx]),
        index=int(idx),
    )


@app.post("/api/predict")
def api_predict():
    started = time.perf_counter()

    try:
        img, name = _read_image()
    except Exception as exc:
        return jsonify(ok=False, error=f"图片读取失败：{exc}"), 400

    try:
        prepared = prepare(img)
    except EmptyImageError as exc:
        return jsonify(ok=False, error=str(exc)), 422
    except Exception as exc:
        return jsonify(ok=False, error=f"预处理失败：{exc}"), 400

    tensor = to_tensor(prepared["view"]).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    digit = int(probs.argmax())
    elapsed = (time.perf_counter() - started) * 1000

    return jsonify(
        ok=True,
        digit=digit,
        confidence=float(probs[digit]),
        probs=[float(p) for p in probs],
        preview=_png_data_url(prepared["view"]),
        inverted=prepared["inverted"],
        bbox=list(prepared["bbox"]),
        source={"name": name, "width": img.width, "height": img.height},
        elapsed_ms=round(elapsed, 1),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"模型已加载：{DEVICE}")
    print(f"打开 http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
