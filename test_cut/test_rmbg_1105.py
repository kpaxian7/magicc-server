import os
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

# ========== 配置参数 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "rmbg-2.0-quantized.onnx")
INPUT_IMAGE = os.path.join(BASE_DIR, "person.jpg")
OUTPUT_PNG = os.path.join(BASE_DIR, "output_person_1105.png")

IN_H, IN_W = 1024, 1024

# ImageNet 归一化参数
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 后处理参数
PROB_THRESHOLD = 0.7
MORPH_KERNEL = 5
MORPH_ITERS = 1
KEEP_LARGEST_COMPONENT = True


def load_session(model_path: str) -> ort.InferenceSession:
    """加载 ONNX 模型，使用 CPU EP"""
    # 强制使用 CPU EP
    providers = ["CPUExecutionProvider"]
    
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # CPU 线程优化
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    
    # 内存优化
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    
    # 显示实际使用的 provider
    actual_provider = sess.get_providers()[0]
    print(f"[INFO] 实际使用的 provider: {actual_provider}")
    
    return sess


def preprocess(img_pil: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
    """预处理图像，返回模型输入和原始尺寸"""
    orig_size = img_pil.size  # (w, h)
    img = img_pil.convert("RGB").resize((IN_W, IN_H), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return arr, orig_size


def postprocess_mask(raw_mask: np.ndarray) -> np.ndarray:
    """将模型输出转换为二值掩码"""
    mask = raw_mask.astype(np.float32)
    
    # 如果未归一化，应用 sigmoid
    if mask.max() > 1.5 or mask.min() < -0.5:
        mask = 1 / (1 + np.exp(-mask))
    mask = np.clip(mask, 0.0, 1.0)

    # 二值化
    bw = (mask >= PROB_THRESHOLD).astype(np.uint8) * 255

    # 形态学操作（去噪）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    if MORPH_ITERS > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=MORPH_ITERS)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=MORPH_ITERS)

    # 保留最大连通组件
    if KEEP_LARGEST_COMPONENT:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = 1 + np.argmax(areas)
            bw = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    return bw


def main():
    print(f"\n{'='*60}")
    print(f"RMBG-2.0 图像抠图 (CPU EP)")
    print(f"{'='*60}")
    
    # 检查文件
    assert os.path.exists(MODEL_PATH), f"模型不存在：{MODEL_PATH}"
    assert os.path.exists(INPUT_IMAGE), f"找不到输入图：{INPUT_IMAGE}"

    # 总计时开始
    t_start = time.perf_counter()

    # 1. 加载模型
    t0 = time.perf_counter()
    sess = load_session(MODEL_PATH)
    t1 = time.perf_counter()
    print(f"[1/5] 模型加载耗时: {(t1 - t0)*1000:.1f} ms")

    # 2. 加载和预处理图像
    t0 = time.perf_counter()
    img_pil = Image.open(INPUT_IMAGE)
    inp, orig_size = preprocess(img_pil)
    t1 = time.perf_counter()
    print(f"[2/5] 图像加载+预处理耗时: {(t1 - t0)*1000:.1f} ms")
    print(f"      原始尺寸: {orig_size[0]}x{orig_size[1]}")

    # 3. 模型推理
    input_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    outputs = sess.run(None, {input_name: inp})
    t1 = time.perf_counter()
    print(f"[3/5] 模型推理耗时: {(t1 - t0)*1000:.1f} ms")

    # 4. 后处理
    t0 = time.perf_counter()
    out = outputs[0]
    if out.ndim == 4:
        out = out[0, 0]
    elif out.ndim == 3:
        out = out[0]
    elif out.ndim != 2:
        raise ValueError(f"不支持的输出维度：{out.shape}")

    # 生成掩码并还原到原始尺寸
    mask_u8_1024 = postprocess_mask(out)
    alpha_img = Image.fromarray(mask_u8_1024, mode="L").resize(orig_size, Image.LANCZOS)
    
    # 合成 RGBA
    rgba = img_pil.convert("RGBA")
    rgba.putalpha(alpha_img)
    t1 = time.perf_counter()
    print(f"[4/5] 后处理耗时: {(t1 - t0)*1000:.1f} ms")

    # 5. 保存结果
    t0 = time.perf_counter()
    rgba.save(OUTPUT_PNG)
    t1 = time.perf_counter()
    print(f"[5/5] 保存图像耗时: {(t1 - t0)*1000:.1f} ms")

    t_end = time.perf_counter()
    
    # 输出汇总
    print(f"\n{'='*60}")
    print(f"✓ 完成！")
    print(f"{'='*60}")
    print(f"总耗时:        {(t_end - t_start)*1000:.1f} ms ({(t_end - t_start):.3f}s)")
    print(f"输出文件:      {os.path.basename(OUTPUT_PNG)}")
    print(f"输出尺寸:      {rgba.size[0]}x{rgba.size[1]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


