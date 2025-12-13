# file: api/routes.py
from sanic import Blueprint, response
from sanic.request import Request
from PIL import Image
from io import BytesIO
import traceback
import logging
import time

from .processor import process_image_to_base64, remove_background

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 创建蓝图
bp = Blueprint('rmbg', url_prefix='/api')


@bp.route('/remove-background-full', methods=['POST'])
async def remove_background_full(request: Request):
    """
    抠图接口
    
    请求：
        - method: POST
        - content-type: multipart/form-data
        - 参数: image (文件类型)
    
    返回：
        - JSON 格式
        - 成功: {"success": true, "data": {"image_base64": "..."}}
        - 失败: {"success": false, "error": "错误信息"}
    """
    start_time = time.time()
    request_id = id(request)
    logger.info(f"[remove-background-full #{request_id}] 收到请求")
    
    try:
        # 检查是否有上传的文件
        logger.debug(f"[remove-background-full #{request_id}] 检查上传文件")
        if 'image' not in request.files:
            logger.warning(f"[remove-background-full #{request_id}] 缺少image参数")
            return response.json({
                'success': False,
                'error': '缺少 image 参数，请上传图片文件'
            }, status=400)

        # 获取上传的文件
        image_file = request.files.get('image')

        if not image_file:
            logger.warning(f"[remove-background-full #{request_id}] 图片文件为空")
            return response.json({
                'success': False,
                'error': '图片文件为空'
            }, status=400)

        # 读取图片数据
        logger.info(f"[remove-background-full #{request_id}] 读取图片数据")
        image_bytes = image_file.body
        logger.info(f"[remove-background-full #{request_id}] 图片大小: {len(image_bytes)}字节")

        # 将字节流转换为 PIL Image
        img_load_start = time.time()
        img = Image.open(BytesIO(image_bytes))
        logger.info(f"[remove-background-full #{request_id}] PIL Image加载完成，耗时: {time.time() - img_load_start:.3f}秒")

        # 处理图片：抠图并转换为 base64
        process_start = time.time()
        logger.info(f"[remove-background-full #{request_id}] 开始处理图片")
        base64_str = process_image_to_base64(img)
        logger.info(f"[remove-background-full #{request_id}] 图片处理完成，耗时: {time.time() - process_start:.3f}秒")

        # 返回成功响应
        total_elapsed = time.time() - start_time
        logger.info(f"[remove-background-full #{request_id}] 请求处理完成，总耗时: {total_elapsed:.3f}秒")
        return response.json({
            'success': True,
            'data': {
                'image_base64': base64_str
            }
        })

    except Exception as e:
        # 捕获所有异常并返回错误信息
        error_msg = str(e)
        error_trace = traceback.format_exc()
        elapsed = time.time() - start_time
        logger.error(f"[remove-background-full #{request_id}] 处理失败，耗时: {elapsed:.3f}秒")
        logger.error(f"[remove-background-full #{request_id}] 错误信息: {error_msg}")
        logger.error(f"[remove-background-full #{request_id}] 错误堆栈:\n{error_trace}")

        return response.json({
            'success': False,
            'error': f'处理图片时发生错误: {error_msg}'
        }, status=500)


@bp.route('/remove-background-binary', methods=['POST'])
async def remove_background_binary(request: Request):
    start_time = time.time()
    request_id = id(request)
    logger.info(f"[remove-background-binary #{request_id}] 收到请求")
    
    try:
        # 检查是否有上传的文件
        logger.debug(f"[remove-background-binary #{request_id}] 检查上传文件")
        if 'image' not in request.files:
            logger.warning(f"[remove-background-binary #{request_id}] 缺少image参数")
            return response.json({
                'success': False,
                'error': '缺少 image 参数，请上传图片文件'
            }, status=400)

        # 获取上传的文件
        image_file = request.files.get('image')

        if not image_file:
            logger.warning(f"[remove-background-binary #{request_id}] 图片文件为空")
            return response.json({
                'success': False,
                'error': '图片文件为空'
            }, status=400)

        # 读取图片数据
        logger.info(f"[remove-background-binary #{request_id}] 读取图片数据")
        image_bytes = image_file.body
        logger.info(f"[remove-background-binary #{request_id}] 图片大小: {len(image_bytes)}字节")

        # 将字节流转换为 PIL Image
        img_load_start = time.time()
        img = Image.open(BytesIO(image_bytes))
        logger.info(f"[remove-background-binary #{request_id}] PIL Image加载完成，耗时: {time.time() - img_load_start:.3f}秒")

        # 处理图片：抠图
        process_start = time.time()
        logger.info(f"[remove-background-binary #{request_id}] 开始处理图片")
        rgba_img = remove_background(img)
        logger.info(f"[remove-background-binary #{request_id}] 图片处理完成，耗时: {time.time() - process_start:.3f}秒")

        # 保存为PNG二进制
        buffer_start = time.time()
        buffer = BytesIO()
        rgba_img.save(buffer, format="PNG")
        # rgba_img.save(buffer, format="WEBP", lossless=False, quality=90)
        buffer.seek(0)
        logger.info(f"[remove-background-binary #{request_id}] PNG编码完成，大小: {len(buffer.getvalue())}字节，耗时: {time.time() - buffer_start:.3f}秒")

        total_elapsed = time.time() - start_time
        logger.info(f"[remove-background-binary #{request_id}] 请求处理完成，总耗时: {total_elapsed:.3f}秒")
        
        return response.raw(buffer.getvalue(),content_type="image/png")
        # return response.raw(buffer.getvalue(), content_type="image/webp")
    except Exception as e:
        # 捕获所有异常并返回错误信息
        error_msg = str(e)
        error_trace = traceback.format_exc()
        elapsed = time.time() - start_time
        logger.error(f"[remove-background-binary #{request_id}] 处理失败，耗时: {elapsed:.3f}秒")
        logger.error(f"[remove-background-binary #{request_id}] 错误信息: {error_msg}")
        logger.error(f"[remove-background-binary #{request_id}] 错误堆栈:\n{error_trace}")

        return response.json({
            'success': False,
            'error': f'处理图片时发生错误: {error_msg}'
        }, status=500)


@bp.route('/health', methods=['GET'])
async def health_check(request: Request):
    """
    健康检查接口
    """
    logger.info(f"[health] 健康检查请求")
    return response.json({
        'success': True,
        'message': 'API is running',
        'service': 'background-removal'
    })

