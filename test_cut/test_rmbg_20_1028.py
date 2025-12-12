import os
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

# # ========== 可调参数 ==========
# MODEL_PATH = "rmbg-2.0.onnx"
# INPUT_IMAGE = "person22.jpg"
# OUT_PNG = "output_person_20_raw.png"
# OUT_JPG = "output_person_20_raw_white.jpg"

# ========== 可调参数 ==========

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型选择：测试不同模型的性能
# FP16 在这台 Mac 上的 CoreML 表现不佳，尝试量化版本
MODEL_PATH = os.path.join(BASE_DIR, "rmbg-2.0.onnx")
INPUT_IMAGE = os.path.join(BASE_DIR, "person.jpg")
OUT_PNG = os.path.join(BASE_DIR, "output_person_20_quantized.png")
OUT_JPG = os.path.join(BASE_DIR, "output_person_20_quantized_white.jpg")

IN_H, IN_W = 1024, 1024

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PROB_THRESHOLD = 0.7
MORPH_KERNEL = 5
MORPH_ITERS = 1
KEEP_LARGEST_COMPONENT = True


# ===========================

def pick_providers() -> Tuple[list, str]:
    ava = ort.get_available_providers()
    print(f"[INFO] 可用的 providers: {ava}")
    
    # 注意：RMBG-2.0 模型不兼容 CoreML（张量维度超过 16384 限制）
    # 强制使用 CPU，反而更快更稳定
    return ["CPUExecutionProvider"], "CPU"


def load_session(model_path: str) -> Tuple[ort.InferenceSession, str]:
    providers, label = pick_providers()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # CPU 线程优化（仅在使用 CPU 时生效）
    so.intra_op_num_threads = 4   # 根据你的 Mac CPU 核心数调整
    so.inter_op_num_threads = 1   # 防止过度上下文切换
    
    # 启用并行执行模式
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    
    # 内存优化
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    
    # 显示实际使用的 provider
    actual_provider = sess.get_providers()[0]
    print(f"[INFO] 实际使用的 provider: {actual_provider}")
    
    return sess, label


def preprocess(img_pil: Image.Image):
    """返回：模型输入(np.ndarray) 以及 原始尺寸"""
    orig_size = img_pil.size  # (w, h)
    img = img_pil.convert("RGB").resize((IN_W, IN_H), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return arr, orig_size


def postprocess_mask(raw_mask: np.ndarray) -> np.ndarray:
    """将模型输出变为 0~255 的二值掩码（仍是1024方形，后面再还原尺寸）"""
    mask = raw_mask.astype(np.float32)
    # 容错：若未归一化，做sigmoid
    if mask.max() > 1.5 or mask.min() < -0.5:
        mask = 1 / (1 + np.exp(-mask))
    mask = np.clip(mask, 0.0, 1.0)

    bw = (mask >= PROB_THRESHOLD).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    if MORPH_ITERS > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=MORPH_ITERS)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=MORPH_ITERS)

    if KEEP_LARGEST_COMPONENT:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = 1 + np.argmax(areas)
            bw = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    return bw  # 这里仍是 1024×1024


def alpha_blend_rgba(src_orig: Image.Image, alpha_u8_resized: Image.Image) -> Image.Image:
    """在原图尺寸上合成 RGBA（不再缩放原图）"""
    rgba = src_orig.convert("RGBA")
    rgba.putalpha(alpha_u8_resized)  # alpha 与原图同尺寸
    return rgba


def save_white_bg_jpg(rgba: Image.Image, path: str, quality: int = 95):
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[-1])
    bg.save(path, "JPEG", quality=quality)


def main():
    assert os.path.exists(MODEL_PATH), f"模型不存在：{MODEL_PATH}"
    assert os.path.exists(INPUT_IMAGE), f"找不到输入图：{INPUT_IMAGE}"

    t0 = time.perf_counter()
    sess, ep = load_session(MODEL_PATH)

    img_pil = Image.open(INPUT_IMAGE)
    inp, orig_size = preprocess(img_pil)  # orig_size = (w, h)

    input_name = sess.get_inputs()[0].name
    t1 = time.perf_counter()
    outputs = sess.run(None, {input_name: inp})
    t2 = time.perf_counter()

    out = outputs[0]
    if out.ndim == 4:
        out = out[0, 0]
    elif out.ndim == 3:
        out = out[0]
    elif out.ndim != 2:
        raise ValueError(f"不支持的输出维度：{out.shape}")

    # 先得到 1024方形 的mask，再还原到原始尺寸
    mask_u8_1024 = postprocess_mask(out)
    # 用 PIL 的 LANCZOS 把 mask 还原到 (orig_w, orig_h)
    alpha_img = Image.fromarray(mask_u8_1024, mode="L").resize(orig_size, Image.LANCZOS)

    # 在原图尺寸上合成
    rgba = alpha_blend_rgba(img_pil, alpha_img)

    rgba.save(OUT_PNG)
    save_white_bg_jpg(rgba, OUT_JPG)

    t3 = time.perf_counter()
    print("====== RMBG-2.0 结果 ======")
    print(f"执行提供者(EP): {ep}")
    print(f"预处理 + 加载耗时: {t1 - t0:.3f}s")
    print(f"推理耗时:         {t2 - t1:.3f}s")
    print(f"后处理 + 保存耗时: {t3 - t2:.3f}s")
    print(f"总耗时:           {t3 - t0:.3f}s")
    print(f"输出: {OUT_PNG}, {OUT_JPG}；尺寸: {rgba.size}")


def benchmark(num_runs=5):
    """运行多次推理，测试平均性能"""
    assert os.path.exists(MODEL_PATH), f"模型不存在：{MODEL_PATH}"
    assert os.path.exists(INPUT_IMAGE), f"找不到输入图：{INPUT_IMAGE}"
    
    print(f"[INFO] 加载模型: {os.path.basename(MODEL_PATH)}")
    sess, ep = load_session(MODEL_PATH)
    
    img_pil = Image.open(INPUT_IMAGE)
    print(f"[INFO] 图片尺寸: {img_pil.size}")
    
    inp, orig_size = preprocess(img_pil)
    input_name = sess.get_inputs()[0].name
    
    # 预热（第一次通常较慢）
    print(f"\n[INFO] 预热中...")
    sess.run(None, {input_name: inp})
    print("[INFO] 预热完成")
    
    # 正式测试
    print(f"\n[INFO] 开始 {num_runs} 次推理测试...\n")
    times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: inp})
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"  第 {i+1} 次: {elapsed*1000:.1f} ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n{'='*40}")
    print(f"性能统计 - {os.path.basename(MODEL_PATH)}")
    print(f"{'='*40}")
    print(f"执行提供者:     {ep}")
    print(f"输入尺寸:       {IN_W}x{IN_H}")
    print(f"测试次数:       {num_runs}")
    print(f"平均推理时间:   {avg_time*1000:.1f} ms ({avg_time:.3f}s)")
    print(f"最快:          {min_time*1000:.1f} ms")
    print(f"最慢:          {max_time*1000:.1f} ms")
    print(f"预估吞吐量:     {1/avg_time:.2f} 张/秒")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark(num_runs=10)
    else:
        main()
