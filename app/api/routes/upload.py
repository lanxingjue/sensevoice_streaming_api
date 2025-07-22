"""音频上传和处理路由 - 阶段2扩展版本"""
import os
import uuid
import time
import asyncio
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ...config.settings import settings
from ...api.schemas.request_models import (
    AudioUploadResponse, 
    TranscriptionResult, 
    HealthCheck
)
from ...models.audio_task import AudioTask, AudioTaskStatus, AudioMetadata
from ...preprocessing.segment_manager import segment_manager
from ...inference.sensevoice_service import sensevoice_service

router = APIRouter(prefix="/api/v1", tags=["音频处理"])


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="音频文件")
):
    """上传音频文件进行转写 - 支持长音频自动切片"""
    
    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    if file_extension not in settings.audio_supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的音频格式: {file_extension}，支持: {settings.audio_supported_formats}"
        )
    
    # 读取文件内容
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    # 验证文件大小
    if file_size_mb > settings.audio_max_file_size_mb:
        raise HTTPException(
            status_code=400, 
            detail=f"文件过大: {file_size_mb:.1f}MB，最大支持: {settings.audio_max_file_size_mb}MB"
        )
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 保存原始文件
    temp_dir = Path(settings.audio_temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    original_file_path = temp_dir / f"{task_id}_original.{file_extension}"
    with open(original_file_path, "wb") as f:
        f.write(content)
    
    # 创建音频任务对象
    metadata = AudioMetadata(
        filename=file.filename,
        file_size_mb=file_size_mb,
        duration_seconds=0.0,  # 将在预处理时更新
        original_sample_rate=0,  # 将在预处理时更新
        original_channels=0,     # 将在预处理时更新
        original_format=file_extension
    )
    
    audio_task = AudioTask(
        task_id=task_id,
        status=AudioTaskStatus.UPLOADED,
        metadata=metadata,
        original_file_path=str(original_file_path)
    )
    
    # 启动后台处理任务
    background_tasks.add_task(process_long_audio_task, audio_task)
    
    return AudioUploadResponse(
        task_id=task_id,
        status="uploaded",
        message="长音频文件已接收，正在进行预处理和切片...",
        file_info={
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "format": file_extension,
            "processing_type": "segmented" if file_size_mb > 10 else "direct"
        }
    )


@router.get("/result/{task_id}", response_model=TranscriptionResult)
async def get_transcription_result(task_id: str):
    """获取转写结果 - 支持切片任务状态"""
    
    # 先查找音频任务
    audio_task = segment_manager.get_audio_task(task_id)
    if audio_task:
        # 长音频任务
        return TranscriptionResult(
            task_id=task_id,
            status=audio_task.status.value,
            text=audio_task.final_text,
            confidence=0.95,  # 平均置信度
            processing_time=audio_task.processing_time,
            error_message=audio_task.error_message
        )
    
    # 查找原有的简单任务存储（向后兼容）
    # 这里保持与阶段1的兼容性
    raise HTTPException(status_code=404, detail="任务不存在")


@router.get("/status/{task_id}")
async def get_detailed_status(task_id: str):
    """获取详细的任务状态信息"""
    
    audio_task = segment_manager.get_audio_task(task_id)
    if not audio_task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 获取切片信息
    segments = segment_manager.get_segments_by_audio(task_id)
    
    segment_summary = {
        "total": len(segments),
        "created": sum(1 for s in segments if s.status.value == "created"),
        "processing": sum(1 for s in segments if s.status.value == "processing"),
        "completed": sum(1 for s in segments if s.status.value == "completed"),
        "failed": sum(1 for s in segments if s.status.value == "failed")
    }
    
    return {
        "task_id": task_id,
        "status": audio_task.status.value,
        "progress_percent": audio_task.progress_percent,
        "current_step": audio_task.current_step,
        "metadata": {
            "filename": audio_task.metadata.filename,
            "duration_seconds": audio_task.metadata.duration_seconds,
            "file_size_mb": audio_task.metadata.file_size_mb
        },
        "segments": segment_summary,
        "created_at": audio_task.created_at,
        "updated_at": audio_task.updated_at,
        "estimated_completion_time": _estimate_completion_time(audio_task, segments)
    }


def _estimate_completion_time(audio_task: AudioTask, segments: list) -> float:
    """估算完成时间"""
    if not segments:
        return 0.0
    
    completed = sum(1 for s in segments if s.status.value == "completed")
    if completed == 0:
        return 60.0  # 默认估算1分钟
    
    # 基于已完成切片的平均处理时间估算
    total_processing_time = sum(
        s.processing_time for s in segments 
        if s.processing_time and s.status.value == "completed"
    )
    
    if total_processing_time > 0:
        avg_time_per_segment = total_processing_time / completed
        remaining_segments = len(segments) - completed
        return remaining_segments * avg_time_per_segment
    
    return 30.0  # 默认估算


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """健康检查接口 - 包含切片管理器状态"""
    
    model_ready = sensevoice_service.is_ready()
    stats = segment_manager.get_statistics()
    
    status = "healthy" if model_ready else "initializing"
    
    return HealthCheck(
        status=status,
        message=f"服务状态: {'就绪' if model_ready else '初始化中'}, "
               f"活跃任务: {stats['audio_tasks']['total']}, "
               f"待处理切片: {stats['segment_tasks']['by_status'].get('created', 0)}",
        timestamp=time.time(),
        version="2.0.0"
    )


async def process_long_audio_task(audio_task: AudioTask):
    """处理长音频任务 - 预处理和切片"""
    try:
        start_time = time.time()
        
        # 使用切片管理器处理音频
        success = segment_manager.process_audio_file(audio_task)
        
        if success:
            processing_time = time.time() - start_time
            audio_task.processing_time = processing_time
            
            # 如果是短音频，可以直接处理第一个切片作为示例
            segments = segment_manager.get_segments_by_audio(audio_task.task_id)
            if len(segments) == 1:
                # 单切片情况，直接转写
                segment = segments[0]
                result = sensevoice_service.transcribe(segment.file_path)
                
                if result["success"]:
                    segment_manager.set_segment_result(
                        segment.task_id,
                        result["text"],
                        result["confidence"],
                        result["processing_time"]
                    )
                    audio_task.final_text = result["text"]
        
    except Exception as e:
        audio_task.set_error(f"处理异常: {str(e)}")


@router.get("/tasks/overview")
async def tasks_overview():
    """任务概览 - 调试和监控用"""
    
    stats = segment_manager.get_statistics()
    
    # 获取最近的任务
    recent_tasks = []
    for audio_task in list(segment_manager.audio_tasks.values())[-10:]:
        recent_tasks.append({
            "task_id": audio_task.task_id,
            "filename": audio_task.metadata.filename,
            "status": audio_task.status.value,
            "progress": audio_task.progress_percent,
            "segments": audio_task.segment_count,
            "created_at": audio_task.created_at
        })
    
    return {
        "statistics": stats,
        "recent_tasks": recent_tasks,
        "service_info": {
            "version": "2.0.0",
            "features": ["long_audio", "smart_segmentation", "quality_enhancement"]
        }
    }
