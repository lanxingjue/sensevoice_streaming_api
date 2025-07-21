"""FastAPI应用主入口"""
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import settings
from .api.routes.upload import router as upload_router
from .inference.sensevoice_service import sensevoice_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("正在初始化SenseVoice服务...")
    
    # 在后台线程中初始化模型，避免阻塞启动
    def init_model():
        success = sensevoice_service.initialize()
        if success:
            logger.info("SenseVoice服务初始化完成")
        else:
            logger.error("SenseVoice服务初始化失败")
    
    # 启动后台初始化任务
    asyncio.create_task(asyncio.to_thread(init_model))
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭服务...")


# 创建FastAPI应用
app = FastAPI(
    title="SenseVoice实时转写API",
    description="基于SenseVoice的高性能音频转写服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "SenseVoice实时转写API服务",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_debug
    )
