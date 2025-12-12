# file: api/app.py
from sanic import Sanic
from sanic.response import json

from .routes import bp


def create_app() -> Sanic:
    """
    创建并配置 Sanic 应用
    
    返回：
        配置好的 Sanic 应用实例
    """
    # 创建 Sanic 应用
    app = Sanic("magicc-server")
    
    # 配置
    app.config.REQUEST_MAX_SIZE = 50 * 1024 * 1024  # 50MB 最大请求大小
    app.config.REQUEST_TIMEOUT = 60  # 60秒超时
    app.config.RESPONSE_TIMEOUT = 60  # 60秒响应超时
    
    # 注册蓝图
    app.blueprint(bp)
    
    # 添加根路由
    @app.route('/')
    async def index(request):
        return json({
            'message': 'Welcome to MAGICC Background Removal API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/api/health',
                'remove_background': '/api/remove-background (POST)'
            }
        })
    
    # 启动时的回调
    @app.before_server_start
    async def setup_model(app, loop):
        """服务启动前预加载模型"""
        print("=" * 50)
        print("Starting MAGICC Background Removal API")
        print("=" * 50)
        # 预加载模型
        from .processor import get_session
        get_session()
        print("Model pre-loaded successfully!")
        print("=" * 50)
    
    return app
