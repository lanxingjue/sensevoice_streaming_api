"""FastAPI应用主入口 - 阶段2扩展版本"""
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import settings
from .api.routes.upload import router as upload_router
from .api.routes.segments import router as segments_router
from .inference.sensevoice_service import sensevoice_service
from .preprocessing.segment_manager import segment_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("正在初始化SenseVoice流式转写服务...")
    
    # 创建必要目录
    import os
    os.makedirs("logs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # 在后台线程中初始化模型
    def init_model():
        success = sensevoice_service.initialize()
        if success:
            logger.info("SenseVoice服务初始化完成")
        else:
            logger.error("SenseVoice服务初始化失败")
    
    # 启动后台任务
    asyncio.create_task(asyncio.to_thread(init_model))
    
    # 启动定期清理任务
    asyncio.create_task(periodic_cleanup())
    
    logger.info("服务启动完成，支持长音频智能切片处理")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭服务...")


async def periodic_cleanup():
    """定期清理过期任务"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时执行一次
            segment_manager.cleanup_completed_tasks(max_age_hours=24.0)
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="SenseVoice实时流式转写API",
    description="支持长音频智能切片的高性能音频转写服务",
    version="2.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload_router)
app.include_router(segments_router)


@app.get("/")
async def root():
    """根路径"""
    stats = segment_manager.get_statistics()
    
    return {
        "service": "SenseVoice实时流式转写API",
        "version": "2.0.0",
        "stage": "阶段2 - 智能切片处理",
        "status": "running",
        "features": [
            "长音频支持(最大200MB)",
            "智能切片(10s+2s重叠)",
            "音频质量增强",
            "切片质量分析",
            "批量处理准备"
        ],
        "current_load": {
            "active_audio_tasks": stats["audio_tasks"]["total"],
            "ready_segments": stats["segment_tasks"]["by_status"].get("created", 0),
            "processing_segments": stats["segment_tasks"]["by_status"].get("processing", 0)
        },
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
