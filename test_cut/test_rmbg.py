import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import time

MODEL_PATH = "rmbg-1.4.onnx"
IMG_PATH = "person22.jpg"
OUT_PATH = "output_person.png"
INPUT_SIZE = 1024  # RMBG-1.4 固定输入分辨率

# 如若 CoreML provider 在你机器上不稳定，可强制 CPU：
# session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
session = ort.InferenceSession(MODEL_PATH, providers=ort.get_available_providers())
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def letterbox_resize(img: Image.Image, size: int = 1024):
    """等比缩放到不超过 size，然后在中心处填充到 size×size。
    返回：填充后的 PIL、有效内容的 (nw, nh)、偏移 (dx, dy) 以及原图 (w, h)
    """
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    img_resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    dx = (size - nw) // 2
    dy = (size - nh) // 2
    canvas.paste(img_resized, (dx, dy))
    return canvas, (w, h), (nw, nh), (dx, dy)

def preprocess(img: Image.Image, size: int = 1024):
    canvas, orig_wh, new_wh, offset = letterbox_resize(img, size)
    arr = np.array(canvas)[:, :, ::-1].astype(np.float32) / 255.0  # RGB→BGR, 0~1
    inp = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return inp, orig_wh, new_wh, offset

def postprocess_mask(mask_1024: np.ndarray, orig_wh, new_wh, offset):
    """从 1024×1024 的 mask 中裁掉填充区域，再缩放回原图尺寸并轻微羽化。"""
    ow, oh = orig_wh
    nw, nh = new_wh
    dx, dy = offset

    # 安全裁剪
    y1, y2 = max(dy, 0), min(dy + nh, INPUT_SIZE)
    x1, x2 = max(dx, 0), min(dx + nw, INPUT_SIZE)
    valid = mask_1024[y1:y2, x1:x2]

    # 还原到原图大小
    valid = (valid * 255.0).astype(np.uint8)
    mask = cv2.resize(valid, (ow, oh), interpolation=cv2.INTER_CUBIC)

    # 羽化边缘（更自然）
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return mask

def compose_rgba(rgb_img: Image.Image, mask_u8: np.ndarray) -> Image.Image:
    r, g, b = rgb_img.convert("RGB").split()
    a = Image.fromarray(mask_u8, "L")
    return Image.merge("RGBA", (r, g, b, a))

# 1) 读图
t_start = time.perf_counter()
img = Image.open(IMG_PATH).convert("RGB")
t_load = time.perf_counter()

# 2) 预处理（letterbox 到 1024×1024）
inp, orig_wh, new_wh, offset = preprocess(img, INPUT_SIZE)
t_preprocess = time.perf_counter()

# 3) 推理
pred = session.run([output_name], {input_name: inp})[0]  # 形状可能是 (1,1,1024,1024) 或 (1,1024,1024)
pred = np.squeeze(pred).astype(np.float32)
# 归一化到 0~1，防止极端输出
pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
t_inference = time.perf_counter()

# 4) 后处理：去 padding，缩放回原图，羽化
mask = postprocess_mask(pred, orig_wh, new_wh, offset)
t_postprocess = time.perf_counter()

# 5) 合成透明前景并保存
rgba = compose_rgba(img, mask)
rgba.save(OUT_PATH)
t_save = time.perf_counter()

# 打印性能统计
print(f"\n{'='*50}")
print(f"性能统计 - test_rmbg.py")
print(f"{'='*50}")
print(f"模型: {MODEL_PATH}")
print(f"图片加载:     {(t_load - t_start)*1000:>8.1f} ms")
print(f"预处理:       {(t_preprocess - t_load)*1000:>8.1f} ms")
print(f"推理时间:     {(t_inference - t_preprocess)*1000:>8.1f} ms")
print(f"后处理:       {(t_postprocess - t_inference)*1000:>8.1f} ms")
print(f"保存文件:     {(t_save - t_postprocess)*1000:>8.1f} ms")
print(f"{'='*50}")
print(f"总耗时:       {(t_save - t_start)*1000:>8.1f} ms ({(t_save - t_start):.3f}s)")
print(f"{'='*50}")
print(f"✅ Done! Saved to {OUT_PATH}")
