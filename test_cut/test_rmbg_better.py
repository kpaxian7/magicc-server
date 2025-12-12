"""
RMBG-1.4 最小测试脚本（修复版）
- 直接把图片缩放到 1024x1024（不 letterbox），与许多开源实现一致
- 人物右侧护栏/细线残留问题：通过更严格阈值 + 形态学处理清理
- 可一键切换是否只保留最大连通区域（通常就是人物主体）

用法：
  python test_rmbg_14.py
需要：
  - 同目录下 rmbg-1.4.onnx
  - 同目录下 person22.jpg（或改 INPUT_IMAGE）
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import os
import time

# =========================
# 可调参数（按需修改）
# =========================

# 模型与输入
MODEL_PATH = "rmbg-1.4.onnx"
INPUT_IMAGE = "person22.jpg"
OUTPUT_PNG = "output_person_14_raw.png"  # 透明 PNG（推荐发布用）
OUTPUT_WHITE = "output_person_white.jpg"  # 白底 JPG（可选预览）

INPUT_SIZE = 1024  # RMBG-1.4 常见输入
# INPUT_SIZE = 640  # RMBG-1.4 常见输入
FORCE_CPU = False  # 某些 macOS 上 CoreML 不稳定时可设 True 强制 CPU

# 归一化（大多数 1.4 实现用 [0,1]；如果效果异常可试 -1~1）
NORM_MODE = "zero_one"  # "zero_one" or "minus_one_one"
USE_SIGMOID = False  # 若导出的 onnx 输出是 logits，可设 True

# 后处理：羽化与二值化
BLUR_SIGMA = 1.2  # 边缘柔化（0 关掉）
THRESH_VAL = 251  # 二值阈值：越高越“严格”，能清更多细碎/护栏
ONLY_LARGEST = True  # 只保留最大连通区域（通常=人物）
MORPH_E_KSZ = (7, 7)  # 形态学核大小（大一些更能清长条噪声）
ERODE_FIRST = True  # 先腐蚀再膨胀，切断与人物的细线连通
OPEN_ON_BIN = True  # 对二值图做开运算，清小碎片/细线

# 额外输出
SAVE_WHITE_JPG = False  # 同时输出一张白底 JPG，方便在不支持透明的场景预览


# =========================
# 工具函数
# =========================

def get_session(model_path: str):
    """创建 ONNX Runtime Session；必要时可强制 CPU。"""
    # if FORCE_CPU:
    #     providers = ["CPUExecutionProvider"]
    # else:
    #     providers = ort.get_available_providers()
    #     print("providers = ", providers)
    #     if not providers:
    #         providers = ["CPUExecutionProvider"]
    # print(f"[INFO] Using providers: {providers}")

    providers_available = ort.get_available_providers()
    print("providers = ", providers_available)
    # providers = ["CoreMLExecutionProvider"]
    # providers = ["AzureExecutionProvider"]
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def preprocess_square_rgb(img: Image.Image, size: int, norm_mode: str):
    """
    直接缩放到 size×size，RGB 顺序。
    返回：NCHW float32
    """
    img_sq = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img_sq).astype(np.float32)  # HWC, RGB
    if norm_mode == "minus_one_one":
        arr = arr / 127.5 - 1.0  # [-1,1]
    else:
        arr = arr / 255.0  # [0,1]
    # NCHW
    inp = np.transpose(arr, (2, 0, 1))[None, ...]
    return inp


def postprocess_soft_to_u8(mask_sq: np.ndarray, orig_wh, blur_sigma: float):
    """
    mask_sq: 0~1 的 1024x1024（或 size×size）软 mask
    -> 缩回原图尺寸并羽化，返回 0~255 的 uint8 掩码
    """
    ow, oh = orig_wh
    m = np.clip(mask_sq, 0.0, 1.0)
    m = (m * 255.0).astype(np.uint8)
    m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_CUBIC)
    if blur_sigma and blur_sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return m


def keep_largest_component(binary_u8: np.ndarray) -> np.ndarray:
    """
    只保留最大连通区域；binary_u8 为 0/255。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num_labels <= 1:
        return binary_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_id = 1 + int(np.argmax(areas))
    return np.where(labels == largest_id, 255, 0).astype(np.uint8)


def compose_rgba(rgb_img: Image.Image, alpha_u8: np.ndarray) -> Image.Image:
    """
    用 alpha_u8（0~255）作为 A 通道，合成 RGBA。
    """
    r, g, b = rgb_img.convert("RGB").split()
    a = Image.fromarray(alpha_u8, "L")
    return Image.merge("RGBA", (r, g, b, a))


# =========================
# 主流程
# =========================

def main():
    assert os.path.exists(MODEL_PATH), f"模型不存在: {MODEL_PATH}"
    assert os.path.exists(INPUT_IMAGE), f"找不到输入图: {INPUT_IMAGE}"

    # 1) 加载模型
    t_start = time.perf_counter()
    session = get_session(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] input_name={input_name}, output_name={output_name}")
    t_load_model = time.perf_counter()

    # 2) 读图（原始尺寸仅用于最后缩回）
    img = Image.open(INPUT_IMAGE).convert("RGB")
    orig_wh = img.size  # (w, h)
    t_load_img = time.perf_counter()

    # 3) 预处理（与 1.4 常见实现保持一致：直接缩放为正方形、RGB、简单归一化）
    inp = preprocess_square_rgb(img, INPUT_SIZE, NORM_MODE)
    t_preprocess = time.perf_counter()

    # 4) 推理
    pred = session.run([output_name], {input_name: inp})[0]  # (1,1,H,W) 或 (1,H,W)
    pred = np.squeeze(pred).astype(np.float32)
    if USE_SIGMOID:
        pred = 1.0 / (1.0 + np.exp(-pred))  # logits -> prob

    # 归一化到 0~1 保险
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
    t_inference = time.perf_counter()

    # 5) 后处理：缩回原尺寸、羽化
    soft_u8 = postprocess_soft_to_u8(pred, orig_wh, BLUR_SIGMA)

    # 6) 护栏/细线清理：更严格阈值 + 形态学
    #   - 阈值越高 => 越多区域被当作背景（0）
    _, mid_bin = cv2.threshold(soft_u8, THRESH_VAL, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_E_KSZ)

    if ERODE_FIRST:
        # 先腐蚀再膨胀：切断与人物连接的细线（如护栏）
        mid_bin = cv2.erode(mid_bin, kernel, iterations=1)
        mid_bin = cv2.dilate(mid_bin, kernel, iterations=1)

    if ONLY_LARGEST:
        mid_bin = keep_largest_component(mid_bin)

    if OPEN_ON_BIN:
        # 再做一次开运算，清理小碎片/细线
        mid_bin = cv2.morphologyEx(mid_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    final_alpha = mid_bin  # 直接作为最终 alpha（也可与 soft_u8 融合，这里追求干净）
    t_postprocess = time.perf_counter()

    # 7) 合成与保存
    rgba = compose_rgba(img, final_alpha)
    rgba.save(OUTPUT_PNG)

    if SAVE_WHITE_JPG:
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])  # 用 alpha 粘贴
        bg.save(OUTPUT_WHITE, quality=95)
    t_save = time.perf_counter()

    # 打印性能统计
    print(f"\n{'='*50}")
    print(f"性能统计 - test_rmbg_better.py")
    print(f"{'='*50}")
    print(f"模型: {MODEL_PATH}")
    print(f"模型加载:     {(t_load_model - t_start)*1000:>8.1f} ms")
    print(f"图片加载:     {(t_load_img - t_load_model)*1000:>8.1f} ms")
    print(f"预处理:       {(t_preprocess - t_load_img)*1000:>8.1f} ms")
    print(f"推理时间:     {(t_inference - t_preprocess)*1000:>8.1f} ms")
    print(f"后处理:       {(t_postprocess - t_inference)*1000:>8.1f} ms")
    print(f"保存文件:     {(t_save - t_postprocess)*1000:>8.1f} ms")
    print(f"{'='*50}")
    print(f"总耗时:       {(t_save - t_start)*1000:>8.1f} ms ({(t_save - t_start):.3f}s)")
    print(f"{'='*50}")
    print(f"[OK] 保存透明 PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
