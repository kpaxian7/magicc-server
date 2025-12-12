# file: api/processor.py
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO

# 获取项目根目录和模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "rmbg-1.4.onnx")
MODEL_PATH = "/home/models/rmbg/rmbg-1.4.onnx"
# MODEL_PATH = "/Users/zelong/Desktop/mine/proj/models/rmbg-1.4.onnx"
MODEL_INPUT_SIZE = (1024, 1024)  # (W, H)

# 全局会话对象，避免重复加载模型
_session = None


def get_session() -> ort.InferenceSession:
    """
    获取或创建 ONNX 推理会话（单例模式）
    """
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        providers = ["CPUExecutionProvider"]
        print(f"Loading model from: {MODEL_PATH}")
        print(f"Using providers: {providers}")
        
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        
        _session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
        print("Model loaded successfully!")
    
    return _session


def preprocess(img: Image.Image, size=(1024, 1024)) -> tuple[np.ndarray, tuple[int, int]]:
    """
    预处理图片
    - RGB
    - resize to 1024x1024 (bilinear)
    - to float32 in [0,1], then normalize by subtracting 0.5 (std=1.0)
    - NCHW
    
    返回：(处理后的数组, 原始尺寸(W,H))
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
    后处理 mask
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


def remove_background(img: Image.Image) -> Image.Image:
    """
    抠图主函数
    
    参数：
        img: PIL Image 对象
    
    返回：
        RGBA 格式的 PIL Image 对象（透明背景）
    """
    # 预处理
    inp, orig_size = preprocess(img, MODEL_INPUT_SIZE)
    
    # 获取会话并推理
    sess = get_session()
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    pred = sess.run([out_name], {in_name: inp})[0]
    
    # 后处理得到 alpha 通道
    alpha = postprocess(pred, orig_size)
    
    # 组合 RGBA
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    
    return rgba


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """
    将 PIL Image 转换为 base64 字符串
    
    参数：
        img: PIL Image 对象
        format: 图片格式，默认 PNG
    
    返回：
        base64 编码的字符串
    """
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str


def process_image_to_base64(img: Image.Image) -> str:
    """
    完整处理流程：输入图片 -> 抠图 -> 返回 base64
    
    参数：
        img: PIL Image 对象
    
    返回：
        base64 编码的字符串
    """
    rgba_img = remove_background(img)
    base64_str = image_to_base64(rgba_img)
    return base64_str
