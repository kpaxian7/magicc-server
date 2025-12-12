"""
RMBG-2.0 最小测试脚本（带护栏/细线清理的修复后处理）
- 输入按 1024x1024 直接缩放（不 letterbox）
- 预处理：RGB + ImageNet Mean/Std 归一化（RMBG-2.0 常见配置）
- 后处理：阈值 + 形态学 + 只保留最大连通区域，尽量清除护栏/细线/小碎块
- 输出：透明 PNG（可选同时导出白底 JPG）

用法：
  python test_rmbg_20.py
需要：
  - 同目录下 rmbg-2.0.onnx  (把你的 2.0 onnx 命名为这个或改 MODEL_PATH)
  - 同目录下 person22.jpg     (或改 INPUT_IMAGE)
"""

import numpy as np
from PIL import Image
import cv2
import os
import time

os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  # 0-VERBOSE,1-INFO,2-WARNING,3-ERROR,4-FATAL

import onnxruntime as ort

# =========================
# 可调参数（按需修改）
# =========================

# 模型与输入
MODEL_PATH = "rmbg-2.0.onnx"
INPUT_IMAGE = "person22.jpg"
OUTPUT_PNG = "output_person.png"  # 透明 PNG（推荐发布用）
OUTPUT_WHITE = "output_person_white.jpg"  # 白底 JPG（可选预览）

INPUT_SIZE = 1024  # RMBG-2.0 常见为 1024；若你的模型是其他尺寸，请改这里
FORCE_CPU = False  # 某些 macOS 上 CoreML 不稳可设 True 强制 CPU

# 预处理（RMBG-2.0：RGB + ImageNet 均值/方差）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
USE_SIGMOID = False  # 若你的 onnx 输出为 logits 则设 True；若已是概率/掩码则 False

# 后处理：羽化与二值化
BLUR_SIGMA = 1.2  # 边缘柔化（0 关掉）
THRESH_VAL = 160  # 二值阈值：越高越“严格”，能清更多细碎/护栏；范围建议 140~180
ONLY_LARGEST = True  # 只保留最大连通区域（通常=人物）
MORPH_E_KSZ = (7, 7)  # 形态学核（大一些更能清长条噪声/护栏）
ERODE_FIRST = True  # 先腐蚀后膨胀，尝试切断与人物连通的细线
OPEN_ON_BIN = True  # 对二值图做开运算，清理小碎片/细线

# 额外输出
SAVE_WHITE_JPG = True  # 同时输出白底 JPG，便于在不支持透明的场景预览


# =========================
# 工具函数
# =========================

def get_session(model_path: str,
                prefer_gpu: bool = False,  # Ubuntu+NVIDIA 可 True
                force_cpu: bool = True):  # mac 上建议先 True，稳定
    so = ort.SessionOptions()
    # 日志等级（双保险）
    so.log_severity_level = 3
    # 稳定优先：不开激进优化，避免奇怪子图/形状推断告警
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # 关闭内存复用/内存池的激进策略，消除“re-use buffer”类告警
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = True  # 保留 CPU 内存池
    # 单线程或少量线程更稳定（你也可改成 os.cpu_count() 提升吞吐）
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if prefer_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif force_cpu:
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]  # 明确不用 CoreML/Azure

    print(f"[INFO] Using providers (forced): {providers}")
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def preprocess_rgb_imagenet(img: Image.Image, size: int,
                            mean: np.ndarray, std: np.ndarray):
    """
    直接缩放到 size×size（不 letterbox），RGB + ImageNet Mean/Std 归一化。
    返回：NCHW float32
    """
    img_sq = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img_sq).astype(np.float32) / 255.0  # HWC, 0~1
    arr = (arr - mean) / std  # 标准化
    inp = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return inp


def postprocess_soft_to_u8(mask_sq: np.ndarray, orig_wh, blur_sigma: float):
    """
    mask_sq: 模型输出的单通道 0~1（或近似范围）的 1024x1024（或 size×size）
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
    """只保留最大连通区域；binary_u8 为 0/255。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num_labels <= 1:
        return binary_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_id = 1 + int(np.argmax(areas))
    return np.where(labels == largest_id, 255, 0).astype(np.uint8)


def compose_rgba(rgb_img: Image.Image, alpha_u8: np.ndarray) -> Image.Image:
    """用 alpha_u8（0~255）作为 A 通道，合成 RGBA。"""
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
    session = get_session(MODEL_PATH, prefer_gpu=False, force_cpu=True)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] input_name={input_name}, output_name={output_name}")
    t_load_model = time.perf_counter()

    # 2) 读图（原始尺寸仅用于最后缩回）
    img = Image.open(INPUT_IMAGE).convert("RGB")
    orig_wh = img.size  # (w, h)
    t_load_img = time.perf_counter()

    # 3) 预处理（RMBG-2.0：RGB + ImageNet 归一化）
    inp = preprocess_rgb_imagenet(img, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD)
    t_preprocess = time.perf_counter()

    # 4) 推理
    pred = session.run([output_name], {input_name: inp})[0]  # (1,1,H,W) 或 (1,H,W)
    pred = np.squeeze(pred).astype(np.float32)

    # 若模型输出为 logits，则做 sigmoid；若已是概率/掩码，则可跳过
    if USE_SIGMOID:
        pred = 1.0 / (1.0 + np.exp(-pred))

    # 保障到 0~1
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
    t_inference = time.perf_counter()

    # 5) 后处理：缩回原尺寸 + 羽化
    soft_u8 = postprocess_soft_to_u8(pred, orig_wh, BLUR_SIGMA)

    # 6) 护栏/细线清理：更严格阈值 + 形态学 + 最大连通区域
    _, mid_bin = cv2.threshold(soft_u8, THRESH_VAL, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_E_KSZ)

    if ERODE_FIRST:
        # 先腐蚀后膨胀：尝试切断与人物相连的细线/护栏
        mid_bin = cv2.erode(mid_bin, kernel, iterations=1)
        mid_bin = cv2.dilate(mid_bin, kernel, iterations=1)

    if ONLY_LARGEST:
        mid_bin = keep_largest_component(mid_bin)

    if OPEN_ON_BIN:
        # 再做一次开运算，清理小碎片/细线
        mid_bin = cv2.morphologyEx(mid_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    final_alpha = mid_bin  # 直接作为最终 alpha；若想更柔和，可与 soft_u8 融合
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
    print(f"性能统计 - test_rmbg_20.py")
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
