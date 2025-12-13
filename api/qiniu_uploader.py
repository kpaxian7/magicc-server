# file: api/qiniu_uploader.py
"""
七牛云上传模块
"""
import logging
import time
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# ==================== 配置项 ====================
# 请填写你的七牛云配置信息

# 访问密钥
QINIU_ACCESS_KEY = "muqTcYlJA-pTO7Hp3HYAWzUkSHHrjpaUsnTYqx_-"  # 填写你的 Access Key
QINIU_SECRET_KEY = "dKZs3DGxuueAxPMOhGDhlNd8BYqL8hapUuqzdPjJ"  # 填写你的 Secret Key

# 存储空间配置
QINIU_BUCKET_NAME = "magicc"  # 填写你的 Bucket 名称
QINIU_DOMAIN = "t77aps4nc.hd-bkt.clouddn.com"

# 上传配置
UPLOAD_PREFIX = "rmbg/"  # 上传文件的前缀路径，例如：rmbg/xxx.png
USE_HTTPS = True  # 是否使用 HTTPS
EXPIRES = 3600  # 上传凭证有效期（秒）

# ================================================


def check_qiniu_config() -> Tuple[bool, str]:
    """
    检查七牛云配置是否完整
    
    返回：
        (是否配置完整, 错误信息)
    """
    if not QINIU_ACCESS_KEY:
        return False, "QINIU_ACCESS_KEY 未配置"
    if not QINIU_SECRET_KEY:
        return False, "QINIU_SECRET_KEY 未配置"
    if not QINIU_BUCKET_NAME:
        return False, "QINIU_BUCKET_NAME 未配置"
    if not QINIU_DOMAIN:
        return False, "QINIU_DOMAIN 未配置"
    return True, ""


def init_qiniu():
    """
    初始化七牛云 SDK
    
    返回：
        (Auth对象, 错误信息)
    """
    try:
        from qiniu import Auth
        
        # 检查配置
        is_valid, error_msg = check_qiniu_config()
        if not is_valid:
            logger.error(f"[qiniu] 配置检查失败: {error_msg}")
            return None, error_msg
        
        auth = Auth(QINIU_ACCESS_KEY, QINIU_SECRET_KEY)
        logger.info("[qiniu] SDK 初始化成功")
        return auth, None
        
    except ImportError:
        error_msg = "qiniu SDK 未安装，请运行: pip install qiniu"
        logger.error(f"[qiniu] {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = f"初始化失败: {str(e)}"
        logger.error(f"[qiniu] {error_msg}")
        return None, error_msg


def generate_filename(prefix: str = "", extension: str = "png") -> str:
    """
    生成唯一的文件名
    
    参数：
        prefix: 文件名前缀
        extension: 文件扩展名
    
    返回：
        文件名
    """
    import uuid
    from datetime import datetime
    
    # 使用时间戳 + UUID 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{prefix}{timestamp}_{unique_id}.{extension}"
    return filename


def upload_image_to_qiniu(
    img: Image.Image,
    filename: Optional[str] = None,
    format: str = "PNG"
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    上传图片到七牛云
    
    参数：
        img: PIL Image 对象
        filename: 文件名（可选，如果不提供则自动生成）
        format: 图片格式（PNG, JPEG, WEBP 等）
    
    返回：
        (是否成功, 图片URL, 错误信息)
    """
    start_time = time.time()
    
    try:
        from qiniu import put_data
        
        logger.info(f"[qiniu] 开始上传图片，格式: {format}")
        
        # 初始化七牛云
        auth, error_msg = init_qiniu()
        if auth is None:
            return False, None, error_msg
        
        # 生成文件名
        if filename is None:
            extension = format.lower()
            filename = generate_filename(prefix=UPLOAD_PREFIX, extension=extension)
        else:
            filename = UPLOAD_PREFIX + filename
        
        logger.info(f"[qiniu] 目标文件名: {filename}")
        
        # 将图片转换为字节流
        buffer = BytesIO()
        img.save(buffer, format=format)
        image_data = buffer.getvalue()
        logger.info(f"[qiniu] 图片大小: {len(image_data)} 字节")
        
        # 生成上传凭证
        token = auth.upload_token(QINIU_BUCKET_NAME, filename, EXPIRES)
        
        # 上传到七牛云
        upload_start = time.time()
        ret, info = put_data(token, filename, image_data)
        upload_elapsed = time.time() - upload_start
        
        # 检查上传结果
        if info.status_code == 200:
            # 构建完整的 URL
            protocol = "https" if USE_HTTPS else "http"
            domain = QINIU_DOMAIN.rstrip('/')
            if domain.startswith('http://') or domain.startswith('https://'):
                image_url = f"{domain}/{ret['key']}"
            else:
                image_url = f"{protocol}://{domain}/{ret['key']}"
            
            total_elapsed = time.time() - start_time
            logger.info(f"[qiniu] 上传成功，耗时: {upload_elapsed:.3f}秒，总耗时: {total_elapsed:.3f}秒")
            logger.info(f"[qiniu] 图片URL: {image_url}")
            
            return True, image_url, None
        else:
            error_msg = f"上传失败，状态码: {info.status_code}, 错误: {info.error}"
            logger.error(f"[qiniu] {error_msg}")
            return False, None, error_msg
            
    except ImportError:
        error_msg = "qiniu SDK 未安装，请运行: pip install qiniu"
        logger.error(f"[qiniu] {error_msg}")
        return False, None, error_msg
    except Exception as e:
        error_msg = f"上传异常: {str(e)}"
        logger.error(f"[qiniu] {error_msg}")
        import traceback
        logger.error(f"[qiniu] 错误堆栈:\n{traceback.format_exc()}")
        return False, None, error_msg


def upload_bytes_to_qiniu(
    data: bytes,
    filename: str,
    content_type: str = "image/png"
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    直接上传字节数据到七牛云
    
    参数：
        data: 字节数据
        filename: 文件名
        content_type: 内容类型
    
    返回：
        (是否成功, 图片URL, 错误信息)
    """
    start_time = time.time()
    
    try:
        from qiniu import put_data
        
        logger.info(f"[qiniu] 开始上传数据，大小: {len(data)} 字节")
        
        # 初始化七牛云
        auth, error_msg = init_qiniu()
        if auth is None:
            return False, None, error_msg
        
        # 添加前缀
        full_filename = UPLOAD_PREFIX + filename
        logger.info(f"[qiniu] 目标文件名: {full_filename}")
        
        # 生成上传凭证
        token = auth.upload_token(QINIU_BUCKET_NAME, full_filename, EXPIRES)
        
        # 上传到七牛云
        upload_start = time.time()
        ret, info = put_data(token, full_filename, data)
        upload_elapsed = time.time() - upload_start
        
        # 检查上传结果
        if info.status_code == 200:
            # 构建完整的 URL
            protocol = "https" if USE_HTTPS else "http"
            domain = QINIU_DOMAIN.rstrip('/')
            if domain.startswith('http://') or domain.startswith('https://'):
                image_url = f"{domain}/{ret['key']}"
            else:
                image_url = f"{protocol}://{domain}/{ret['key']}"
            
            total_elapsed = time.time() - start_time
            logger.info(f"[qiniu] 上传成功，耗时: {upload_elapsed:.3f}秒，总耗时: {total_elapsed:.3f}秒")
            logger.info(f"[qiniu] 文件URL: {image_url}")
            
            return True, image_url, None
        else:
            error_msg = f"上传失败，状态码: {info.status_code}, 错误: {info.error}"
            logger.error(f"[qiniu] {error_msg}")
            return False, None, error_msg
            
    except ImportError:
        error_msg = "qiniu SDK 未安装，请运行: pip install qiniu"
        logger.error(f"[qiniu] {error_msg}")
        return False, None, error_msg
    except Exception as e:
        error_msg = f"上传异常: {str(e)}"
        logger.error(f"[qiniu] {error_msg}")
        import traceback
        logger.error(f"[qiniu] 错误堆栈:\n{traceback.format_exc()}")
        return False, None, error_msg

