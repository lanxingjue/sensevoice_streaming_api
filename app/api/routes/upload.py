"""简化的上传路由"""
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ...config.settings import settings
from ...models.audio_task import TaskStatus
from ...preprocessing.stream_uploader import stream_uploader
from ...preprocessing.task_manager import task_manager
from ..schemas.request_models import AudioUploadResponse, TranscriptionResult

router = APIRouter(prefix="/api/v1", tags=["音频处理"])


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="音频文件")
):
    """流式上传音频文件"""
    
    # 基本验证
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    file_ext = Path(file.filename).suffix.lower().lstrip('.')
    if file_ext not in settings.audio_supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持格式: {file_ext}"
        )
    
    # 生成任务和文件路径
    task_id = str(uuid.uuid4())
    temp_dir = Path(settings.audio_temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / f"{task_id}.{file_ext}"
    
    # 流式保存文件
    success, file_size_mb, error = await stream_uploader.save_uploaded_file(
        file, file_path, settings.audio_max_file_size_mb
    )
    
    if not success:
        raise HTTPException(status_code=413, detail=error)
    
    # 创建音频任务
    audio_task = task_manager.create_audio_task(
        task_id, file.filename, str(file_path), file_size_mb
    )
    
    # 后台处理
    background_tasks.add_task(
        task_manager.process_audio_task, task_id
    )
    
    return AudioUploadResponse(
        task_id=task_id,
        status="uploaded",
        message=f"文件已上传 ({file_size_mb:.1f}MB)，正在处理...",
        file_info={
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "format": file_ext
        }
    )


@router.get("/result/{task_id}", response_model=TranscriptionResult)
async def get_result(task_id: str):
    """获取转写结果"""
    
    task = task_manager.get_audio_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return TranscriptionResult(
        task_id=task_id,
        status=task.status.value,
        text="",  # 将在后续阶段实现
        confidence=0.0,
        processing_time=task.processing_time,
        error_message=task.error_message
    )


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """获取详细状态"""
    
    task = task_manager.get_audio_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "task_id": task_id,
        "status": task.status.value,
        "filename": task.filename,
        "file_size_mb": task.file_size_mb,
        "duration_seconds": task.duration_seconds,
        "segment_count": task.segment_count,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "error": task.error_message
    }


@router.get("/segments/ready")
async def get_ready_segments():
    """获取准备处理的切片"""
    
    segments = task_manager.get_ready_segments(limit=128)
    
    return {
        "count": len(segments),
        "segments": [
            {
                "segment_id": seg.segment_id,
                "parent_id": seg.parent_id,
                "index": seg.index,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "priority": seg.priority
            }
            for seg in segments
        ]
    }


@router.get("/stats")
async def get_stats():
    """获取系统状态"""
    
    stats = task_manager.get_statistics()
    
    return {
        "service": "SenseVoice简化版",
        "version": "2.0-simplified",
        "statistics": stats
    }
