"""简化版主应用"""
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
    """应用生命周期"""
    logger.info("启动SenseVoice简化版服务...")
    
    # 创建目录
    import os
    os.makedirs("temp", exist_ok=True)
    
    # 初始化模型
    def init_model():
        success = sensevoice_service.initialize()
        logger.info(f"模型初始化: {'成功' if success else '失败'}")
    
    asyncio.create_task(asyncio.to_thread(init_model))
    
    yield
    
    logger.info("关闭服务...")


# 创建应用
app = FastAPI(
    title="SenseVoice简化版",
    description="支持大文件流式上传的音频转写服务",
    version="2.0-simplified",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload_router)


@app.get("/")
async def root():
    """根页面"""
    from .preprocessing.task_manager import task_manager
    stats = task_manager.get_statistics()
    
    return {
        "service": "SenseVoice简化版",
        "version": "2.0-simplified", 
        "features": [
            "流式大文件上传",
            "简化音频切片",
            "批处理准备"
        ],
        "stats": stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
