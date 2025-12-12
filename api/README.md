# MAGICC 抠图 API

基于 Sanic 框架的图片背景去除 API 服务。

## 项目结构

```
api/
├── __init__.py      # 包初始化文件
├── app.py           # Sanic 应用创建和配置
├── routes.py        # API 路由定义
├── processor.py     # 抠图逻辑处理
└── README.md        # 本文档
```

## 功能特性

- ✅ 使用 RMBG-1.4 模型进行高质量背景去除
- ✅ 支持多种图片格式（JPEG、PNG 等）
- ✅ 返回 base64 编码的透明背景图片
- ✅ 模型预加载，提升响应速度
- ✅ 完善的错误处理

## API 接口

### 1. 抠图接口

**请求：**
```
POST /api/remove-background
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

**错误响应：**
```json
{
    "success": false,
    "error": "错误信息"
}
```

### 2. 健康检查

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

## 使用示例

### Python 示例

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# 上传图片进行抠图
with open('input.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/api/remove-background', files=files)

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

### cURL 示例

```bash
curl -X POST http://localhost:8000/api/remove-background \
  -F "image=@input.jpg" \
  | jq -r '.data.image_base64' \
  | base64 -d > output.png
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
- **图像处理**: PIL/Pillow
- **推理引擎**: ONNX Runtime

## 配置说明

在 `api/app.py` 中可以修改以下配置：

- `REQUEST_MAX_SIZE`: 最大请求大小（默认 50MB）
- `REQUEST_TIMEOUT`: 请求超时时间（默认 60秒）
- `RESPONSE_TIMEOUT`: 响应超时时间（默认 60秒）

在 `api/processor.py` 中可以修改：

- `MODEL_INPUT_SIZE`: 模型输入尺寸（默认 1024x1024）
- `intra_op_num_threads`: CPU 线程数（默认 4）

## 性能优化

- 模型在服务启动时预加载，避免首次请求延迟
- 使用单例模式管理 ONNX 会话，避免重复加载
- 支持异步处理，提高并发性能
