# MAGICC 抠图 API

基于 Sanic 框架的图片背景去除 API 服务。

## 项目结构

```
api/
├── __init__.py         # 包初始化文件
├── app.py              # Sanic 应用创建和配置
├── routes.py           # API 路由定义
├── processor.py        # 抠图逻辑处理
├── qiniu_uploader.py   # 七牛云上传模块
└── README.md           # 本文档
```

## 功能特性

- ✅ 使用 RMBG-1.4 模型进行高质量背景去除
- ✅ 支持多种图片格式（JPEG、PNG、WEBP、HEIC 等）
- ✅ 三种返回方式：base64、二进制、七牛云链接
- ✅ 模型预加载，提升响应速度
- ✅ 完善的日志记录和错误处理
- ✅ 性能监控（每个步骤记录耗时）

## API 接口

### 1. 抠图接口 - Base64 返回

**请求：**
```
POST /api/remove-background-full
Content-Type: multipart/form-data
```

**参数：**
- `image`: 图片文件（必须）

**返回：**
```json
{
    "success": true,
    "data": {
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
}
```

### 2. 抠图接口 - 二进制返回

**请求：**
```
POST /api/remove-background-binary
Content-Type: multipart/form-data
```

**参数：**
- `image`: 图片文件（必须）

**返回：**
- Content-Type: `image/png`
- 直接返回 PNG 图片二进制数据

### 3. 抠图接口 - 七牛云链接 ⭐️ 新增

**请求：**
```
POST /api/remove-background-link
Content-Type: multipart/form-data
```

**参数：**
- `image`: 图片文件（必须）

**返回：**
```json
{
    "success": true,
    "data": {
        "image_url": "https://cdn.example.com/rmbg/20251213_103045_a1b2c3d4.png"
    }
}
```

**配置要求：**
需要先在 `api/qiniu_uploader.py` 中配置七牛云参数。详见 [QINIU_CONFIG.md](../QINIU_CONFIG.md)

### 4. 健康检查

**请求：**
```
GET /api/health
```

**返回：**
```json
{
    "success": true,
    "message": "API is running",
    "service": "background-removal"
}
```

## 错误响应格式

```json
{
    "success": false,
    "error": "错误信息"
}
```

## 使用示例

### 示例 1: Base64 返回（Python）

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# 上传图片进行抠图
with open('input.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/api/remove-background-full', files=files)

if response.json()['success']:
    # 获取 base64 数据并保存
    base64_str = response.json()['data']['image_base64']
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save('output.png')
    print('抠图成功！')
else:
    print(f'错误: {response.json()["error"]}')
```

### 示例 2: 二进制返回（cURL）

```bash
curl -X POST http://localhost:8000/api/remove-background-binary \
  -F "image=@input.jpg" \
  --output output.png
```

### 示例 3: 七牛云链接返回（cURL）

```bash
curl -X POST http://localhost:8000/api/remove-background-link \
  -F "image=@input.jpg" \
  | jq -r '.data.image_url'
```

### 示例 4: 测试 HEIC 格式

```bash
# iPhone 拍摄的 HEIC 格式照片
curl -X POST http://localhost:8000/api/remove-background-binary \
  -F "image=@photo.heic" \
  --output result.png
```

## 启动服务

从项目根目录运行：

```bash
python main.py
```

服务将在 `http://0.0.0.0:8000` 启动。

## 技术栈

- **框架**: Sanic (异步 Web 框架)
- **AI 模型**: RMBG-1.4 (ONNX 格式)
- **图像处理**: PIL/Pillow, pillow-heif
- **推理引擎**: ONNX Runtime
- **云存储**: 七牛云 SDK

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- `sanic>=25.3.0` - Web 框架
- `pillow>=12.0.0` - 图像处理
- `pillow-heif==0.22.0` - HEIC 格式支持
- `onnxruntime>=1.19.2` - AI 推理
- `qiniu==7.14.0` - 七牛云 SDK

## 配置说明

### 应用配置 (`api/app.py`)

- `REQUEST_MAX_SIZE`: 最大请求大小（默认 50MB）
- `REQUEST_TIMEOUT`: 请求超时时间（默认 60秒）
- `RESPONSE_TIMEOUT`: 响应超时时间（默认 60秒）

### 模型配置 (`api/processor.py`)

- `MODEL_PATH`: ONNX 模型路径
- `MODEL_INPUT_SIZE`: 模型输入尺寸（默认 1024x1024）
- `intra_op_num_threads`: CPU 线程数（默认 4）

### 七牛云配置 (`api/qiniu_uploader.py`)

详见 [QINIU_CONFIG.md](../QINIU_CONFIG.md)

必填项：
- `QINIU_ACCESS_KEY` - Access Key
- `QINIU_SECRET_KEY` - Secret Key
- `QINIU_BUCKET_NAME` - Bucket 名称
- `QINIU_DOMAIN` - CDN 域名

## 日志查看

使用 journalctl 查看日志：

```bash
# 实时查看
journalctl -u your-service-name -f

# 查看最近 100 条
journalctl -u your-service-name -n 100

# 只看错误
journalctl -u your-service-name -p err
```

## 性能优化

- ✅ 模型在服务启动时预加载，避免首次请求延迟
- ✅ 使用单例模式管理 ONNX 会话，避免重复加载
- ✅ 支持异步处理，提高并发性能
- ✅ 详细的性能日志，便于分析瓶颈

## 支持的图片格式

- JPEG / JPG
- PNG
- WEBP
- GIF
- HEIC / HEIF (需要 pillow-heif)
- AVIF (需要 pillow-heif)

## 相关文档

- [QINIU_CONFIG.md](../QINIU_CONFIG.md) - 七牛云配置指南
- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - 部署指南
- [INSTALL_HEIC_SUPPORT.md](../INSTALL_HEIC_SUPPORT.md) - HEIC 支持安装

