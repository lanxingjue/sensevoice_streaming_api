"""SenseVoice流式转写API - 阶段3主应用"""
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import settings
from .api.routes.upload import router as upload_router
from .api.routes.streaming import router as streaming_router
from .inference.sensevoice_service import sensevoice_service
from .streaming.batch_processor import batch_processor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动SenseVoice流式转写API - 阶段3")
    
    # 创建目录
    import os
    os.makedirs("temp", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 初始化SenseVoice模型
    def init_model():
        success = sensevoice_service.initialize()
        logger.info(f"SenseVoice模型初始化: {'成功' if success else '失败'}")
        return success
    
    # 异步初始化模型
    model_ready = await asyncio.to_thread(init_model)
    
    if model_ready:
        # 启动批处理系统
        logger.info("启动批处理系统...")
        await batch_processor.start()
        logger.info("批处理系统启动完成")
    else:
        logger.warning("模型初始化失败，批处理系统未启动")
    
    logger.info("=== 阶段3流式转写服务启动完成 ===")
    logger.info("特性: 双队列调度 + 批量推理 + 首片段优先")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭流式转写服务...")
    await batch_processor.stop()
    logger.info("服务关闭完成")


# 创建FastAPI应用
app = FastAPI(
    title="SenseVoice流式转写API",
    description="阶段3: 双队列批处理系统，支持首片段优先的实时转写",
    version="3.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload_router)
app.include_router(streaming_router)


@app.get("/")
async def root():
    """根页面"""
    from .preprocessing.task_manager import task_manager
    
    # 获取各组件状态
    task_stats = task_manager.get_statistics()
    batch_status = batch_processor.get_status() if batch_processor.is_running else None
    
    return {
        "service": "SenseVoice流式转写API",
        "version": "3.0.0",
        "stage": "阶段3 - 双队列批处理系统",
        "features": [
            "双队列智能调度",
            "GPU批量推理",
            "首片段优先处理",
            "200ms批处理窗口",
            "实时结果分发",
            "性能监控"
        ],
        "status": {
            "model_ready": sensevoice_service.is_ready(),
            "batch_processor_running": batch_processor.is_running,
            "task_stats": task_stats,
            "batch_stats": batch_status.get("queue_stats") if batch_status else None
        },
        "endpoints": {
            "upload": "/api/v1/upload",
            "streaming_control": "/api/v1/streaming/start",
            "streaming_status": "/api/v1/streaming/status",
            "first_segment": "/api/v1/streaming/first-segment/{audio_id}",
            "documentation": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
