# file: run_rmbg14_onnx.py
import onnxruntime as ort
from PIL import Image
import numpy as np
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE_DIR, "rmbg-1.4.onnx")
INPUT_IMAGE = os.path.join(BASE_DIR, "person.jpg")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "output_person_14_new.png")
MODEL_INPUT_SIZE = (1024, 1024)  # (W, H)

def preprocess(img: Image.Image, size=(1024, 1024)) -> np.ndarray:
    """
    Replicate RMBG-1.4 preprocess:
    - RGB
    - resize to 1024x1024 (bilinear)
    - to float32 in [0,1], then normalize by subtracting 0.5 (std=1.0)
    - NCHW
    """
    img = img.convert("RGB")
    w, h = img.size
    img_resized = img.resize(size, Image.BILINEAR)

    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    arr = arr - 0.5  # mean=[0.5,0.5,0.5], std=[1,1,1]
    # HWC -> CHW -> NCHW
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr, (w, h)

def postprocess(mask: np.ndarray, orig_size: tuple[int, int]) -> Image.Image:
    """
    - Expect mask shape [1, 1, H, W] or [1, H, W]
    - Resize back to original (W,H)
    - Min-max normalize to [0,255] uint8
    - Return alpha (PIL Image, mode 'L')
    """
    m = mask.squeeze()
    if m.ndim == 3:
        m = m[0]  # take channel 0

    # min-max normalize
    m_min = float(np.min(m))
    m_max = float(np.max(m))
    if m_max > m_min:
        m = (m - m_min) / (m_max - m_min)
    else:
        m = np.zeros_like(m, dtype=np.float32)

    m = (m * 255.0).astype(np.uint8)
    alpha = Image.fromarray(m).resize(orig_size, Image.BILINEAR)
    return alpha

def make_session(model_path: str) -> ort.InferenceSession:
    # 优先用 CoreML（在 Apple Silicon 上通常更快），否则退回 CPU
    avail = ort.get_available_providers()
    # providers = []
    # if "CoreMLExecutionProvider" in avail:
    #     providers.append("CoreMLExecutionProvider")
    # providers.append("CPUExecutionProvider")

    providers = ["CPUExecutionProvider"]
    # providers = ["CoreMLExecutionProvider"]
    print("result providers: ", providers)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # CPU 线程优化（如果最终使用了 CPU）
    so.intra_op_num_threads = 4  # 根据你的 CPU 核心数调整
    so.inter_op_num_threads = 1
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)

def main():
    if not os.path.exists(MODEL):
        raise FileNotFoundError(f"Model not found: {MODEL}")
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    # load image & preprocess
    t_start = time.time()
    img = Image.open(INPUT_IMAGE)
    t_load_img = time.time()
    
    inp, orig_size = preprocess(img, MODEL_INPUT_SIZE)
    t_preprocess = time.time()

    # run onnx
    sess = make_session(MODEL)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    t_load_model = time.time()

    pred = sess.run([out_name], {in_name: inp})[0]  # -> [1,1,1024,1024] or similar
    t_inference = time.time()

    # postprocess to alpha
    alpha = postprocess(pred, orig_size)

    # compose RGBA
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    t_postprocess = time.time()
    
    rgba.save(OUTPUT_IMAGE)
    t_save = time.time()

    # 打印性能统计
    print(f"\n{'='*50}")
    print(f"性能统计 - test_rmbg_14_1029.py")
    print(f"{'='*50}")
    print(f"模型: {os.path.basename(MODEL)}")
    print(f"图片加载:     {(t_load_img - t_start)*1000:>8.1f} ms")
    print(f"预处理:       {(t_preprocess - t_load_img)*1000:>8.1f} ms")
    print(f"模型加载:     {(t_load_model - t_preprocess)*1000:>8.1f} ms")
    print(f"推理时间:     {(t_inference - t_load_model)*1000:>8.1f} ms")
    print(f"后处理:       {(t_postprocess - t_inference)*1000:>8.1f} ms")
    print(f"保存文件:     {(t_save - t_postprocess)*1000:>8.1f} ms")
    print(f"{'='*50}")
    print(f"总耗时:       {(t_save - t_start)*1000:>8.1f} ms ({(t_save - t_start):.3f}s)")
    print(f"{'='*50}")
    print(f"Saved: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
