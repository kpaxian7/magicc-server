# file: api/routes.py
from sanic import Blueprint, response
from sanic.request import Request
from PIL import Image
from io import BytesIO
import traceback

from .processor import process_image_to_base64, remove_background

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
    try:
        # 检查是否有上传的文件
        if 'image' not in request.files:
            return response.json({
                'success': False,
                'error': '缺少 image 参数，请上传图片文件'
            }, status=400)

        # 获取上传的文件
        image_file = request.files.get('image')

        if not image_file:
            return response.json({
                'success': False,
                'error': '图片文件为空'
            }, status=400)

        # 读取图片数据
        image_bytes = image_file.body

        # 将字节流转换为 PIL Image
        img = Image.open(BytesIO(image_bytes))

        # 处理图片：抠图并转换为 base64
        base64_str = process_image_to_base64(img)

        # 返回成功响应
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
        print(f"Error processing image: {error_msg}")
        print(error_trace)

        return response.json({
            'success': False,
            'error': f'处理图片时发生错误: {error_msg}'
        }, status=500)


@bp.route('/remove-background-binary', methods=['POST'])
async def remove_background_binary(request: Request):
    try:
        # 检查是否有上传的文件
        if 'image' not in request.files:
            return response.json({
                'success': False,
                'error': '缺少 image 参数，请上传图片文件'
            }, status=400)

        # 获取上传的文件
        image_file = request.files.get('image')

        if not image_file:
            return response.json({
                'success': False,
                'error': '图片文件为空'
            }, status=400)

        # 读取图片数据
        image_bytes = image_file.body

        # 将字节流转换为 PIL Image
        img = Image.open(BytesIO(image_bytes))

        rgba_img = remove_background(img)

        buffer = BytesIO()
        rgba_img.save(buffer, format="PNG")
        # rgba_img.save(buffer, format="WEBP", lossless=False, quality=90)
        buffer.seek(0)

        return response.raw(buffer.getvalue(),content_type="image/png")
        # return response.raw(buffer.getvalue(), content_type="image/webp")
    except Exception as e:
        # 捕获所有异常并返回错误信息
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error processing image: {error_msg}")
        print(error_trace)

        return response.json({
            'success': False,
            'error': f'处理图片时发生错误: {error_msg}'
        }, status=500)


@bp.route('/health', methods=['GET'])
async def health_check(request: Request):
    """
    健康检查接口
    """
    return response.json({
        'success': True,
        'message': 'API is running',
        'service': 'background-removal'
    })
