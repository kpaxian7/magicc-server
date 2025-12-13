# file: api/processor.py
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 获取项目根目录和模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = "/home/models/rmbg/rmbg-1.4.onnx"
MODEL_PATH = "/Users/zelong/Desktop/mine/proj/models/rmbg-1.4.onnx"
MODEL_INPUT_SIZE = (1024, 1024)  # (W, H)

# 全局会话对象，避免重复加载模型
_session = None


def get_session() -> ort.InferenceSession:
    """
    获取或创建 ONNX 推理会话（单例模式）
    """
    global _session
    if _session is None:
        logger.info(f"[get_session] 开始加载模型")
        start_time = time.time()
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"[get_session] 模型文件不存在: {MODEL_PATH}")
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        providers = ["CPUExecutionProvider"]
        logger.info(f"[get_session] 模型路径: {MODEL_PATH}")
        logger.info(f"[get_session] 使用提供者: {providers}")
        
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        
        _session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
        elapsed = time.time() - start_time
        logger.info(f"[get_session] 模型加载成功，耗时: {elapsed:.3f}秒")
    else:
        logger.debug(f"[get_session] 使用已加载的模型会话")
    
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
    start_time = time.time()
    logger.info(f"[preprocess] 开始预处理图片")
    
    img = img.convert("RGB")
    w, h = img.size
    logger.info(f"[preprocess] 原始图片尺寸: {w}x{h}")
    
    img_resized = img.resize(size, Image.BILINEAR)
    logger.debug(f"[preprocess] 图片已调整为: {size[0]}x{size[1]}")

    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    arr = arr - 0.5  # mean=[0.5,0.5,0.5], std=[1,1,1]
    # HWC -> CHW -> NCHW
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    
    elapsed = time.time() - start_time
    logger.info(f"[preprocess] 预处理完成，耗时: {elapsed:.3f}秒")
    return arr, (w, h)


def postprocess(mask: np.ndarray, orig_size: tuple[int, int]) -> Image.Image:
    """
    后处理 mask
    - Expect mask shape [1, 1, H, W] or [1, H, W]
    - Resize back to original (W,H)
    - Min-max normalize to [0,255] uint8
    - Return alpha (PIL Image, mode 'L')
    """
    start_time = time.time()
    logger.info(f"[postprocess] 开始后处理mask")
    logger.debug(f"[postprocess] mask形状: {mask.shape}")
    
    m = mask.squeeze()
    if m.ndim == 3:
        m = m[0]  # take channel 0

    # min-max normalize
    m_min = float(np.min(m))
    m_max = float(np.max(m))
    logger.debug(f"[postprocess] mask值范围: [{m_min:.4f}, {m_max:.4f}]")
    
    if m_max > m_min:
        m = (m - m_min) / (m_max - m_min)
    else:
        m = np.zeros_like(m, dtype=np.float32)

    m = (m * 255.0).astype(np.uint8)
    alpha = Image.fromarray(m).resize(orig_size, Image.BILINEAR)
    
    elapsed = time.time() - start_time
    logger.info(f"[postprocess] 后处理完成，目标尺寸: {orig_size[0]}x{orig_size[1]}，耗时: {elapsed:.3f}秒")
    return alpha


def remove_background(img: Image.Image) -> Image.Image:
    """
    抠图主函数
    
    参数：
        img: PIL Image 对象
    
    返回：
        RGBA 格式的 PIL Image 对象（透明背景）
    """
    start_time = time.time()
    logger.info(f"[remove_background] 开始抠图处理")
    
    # 预处理
    preprocess_start = time.time()
    inp, orig_size = preprocess(img, MODEL_INPUT_SIZE)
    logger.info(f"[remove_background] 预处理完成，耗时: {time.time() - preprocess_start:.3f}秒")
    
    # 获取会话并推理
    inference_start = time.time()
    sess = get_session()
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logger.info(f"[remove_background] 开始模型推理")
    pred = sess.run([out_name], {in_name: inp})[0]
    logger.info(f"[remove_background] 模型推理完成，耗时: {time.time() - inference_start:.3f}秒")
    
    # 后处理得到 alpha 通道
    postprocess_start = time.time()
    alpha = postprocess(pred, orig_size)
    logger.info(f"[remove_background] 后处理完成，耗时: {time.time() - postprocess_start:.3f}秒")
    
    # 组合 RGBA
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    
    total_elapsed = time.time() - start_time
    logger.info(f"[remove_background] 抠图处理全部完成，总耗时: {total_elapsed:.3f}秒")
    
    return rgba


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """
    将 PIL Image 转换为 base64 字符串
    
    参数：a
        img: PIL Image 对象
        format: 图片格式，默认 PNG
    
    返回：
        base64 编码的字符串
    """
    start_time = time.time()
    logger.info(f"[image_to_base64] 开始转换为base64，格式: {format}")
    
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    elapsed = time.time() - start_time
    logger.info(f"[image_to_base64] base64转换完成，大小: {len(base64_str)}字符，耗时: {elapsed:.3f}秒")
    return base64_str


def process_image_to_base64(img: Image.Image) -> str:
    """
    完整处理流程：输入图片 -> 抠图 -> 返回 base64
    
    参数：
        img: PIL Image 对象
    
    返回：
        base64 编码的字符串
    """
    start_time = time.time()
    logger.info(f"[process_image_to_base64] 开始完整处理流程")
    
    rgba_img = remove_background(img)
    base64_str = image_to_base64(rgba_img)
    
    elapsed = time.time() - start_time
    logger.info(f"[process_image_to_base64] 完整流程结束，总耗时: {elapsed:.3f}秒")
    return base64_str

