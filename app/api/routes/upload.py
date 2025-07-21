"""音频上传和处理路由"""
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
from ...utils.audio_utils import AudioProcessor
from ...inference.sensevoice_service import sensevoice_service

router = APIRouter(prefix="/api/v1", tags=["音频处理"])

# 全局状态存储 (阶段1使用简单字典，后续会替换为队列系统)
task_results: Dict[str, Dict[str, Any]] = {}
audio_processor = AudioProcessor()


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="音频文件")
):
    """上传音频文件进行转写"""
    
    # 验证文件格式
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    if file_extension not in settings.audio_supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的音频格式: {file_extension}，支持的格式: {settings.audio_supported_formats}"
        )
    
    # 验证文件大小
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.audio_max_file_size_mb:
        raise HTTPException(
            status_code=400, 
            detail=f"文件过大: {file_size_mb:.1f}MB，最大支持: {settings.audio_max_file_size_mb}MB"
        )
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 保存临时文件
    temp_dir = Path(settings.audio_temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    original_file_path = temp_dir / f"{task_id}_original.{file_extension}"
    with open(original_file_path, "wb") as f:
        f.write(content)
    
    # 初始化任务状态
    task_results[task_id] = {
        "status": "processing",
        "file_info": {
            "filename": file.filename,
            "size_mb": file_size_mb,
            "format": file_extension
        },
        "created_at": time.time()
    }
    
    # 启动后台处理任务
    background_tasks.add_task(process_audio_task, task_id, original_file_path)
    
    return AudioUploadResponse(
        task_id=task_id,
        status="processing",
        message="音频文件已接收，正在处理中...",
        file_info={
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "format": file_extension
        }
    )


@router.get("/result/{task_id}", response_model=TranscriptionResult)
async def get_transcription_result(task_id: str):
    """获取转写结果"""
    
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_data = task_results[task_id]
    
    return TranscriptionResult(
        task_id=task_id,
        status=task_data["status"],
        text=task_data.get("text"),
        confidence=task_data.get("confidence"),
        processing_time=task_data.get("processing_time"),
        error_message=task_data.get("error_message")
    )


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """健康检查接口"""
    
    model_ready = sensevoice_service.is_ready()
    status = "healthy" if model_ready else "initializing"
    
    return HealthCheck(
        status=status,
        message=f"SenseVoice服务状态: {'就绪' if model_ready else '初始化中'}",
        timestamp=time.time(),
        version="1.0.0"
    )


async def process_audio_task(task_id: str, audio_file_path: Path):
    """处理音频转写任务"""
    try:
        # 验证音频文件
        if not audio_processor.validate_audio_file(str(audio_file_path)):
            task_results[task_id].update({
                "status": "failed",
                "error_message": "音频文件格式无效或损坏"
            })
            return
        
        # 获取音频信息
        audio_info = audio_processor.get_audio_info(str(audio_file_path))
        if audio_info:
            task_results[task_id]["file_info"].update(audio_info)
        
        # 格式转换 (转换为16kHz WAV格式)
        converted_file_path = audio_file_path.parent / f"{task_id}_converted.wav"
        
        conversion_success = audio_processor.convert_audio_format(
            str(audio_file_path), 
            str(converted_file_path)
        )
        
        if not conversion_success:
            task_results[task_id].update({
                "status": "failed",
                "error_message": "音频格式转换失败"
            })
            return
        
        # 执行转写
        start_time = time.time()
        result = sensevoice_service.transcribe(str(converted_file_path))
        
        if result["success"]:
            task_results[task_id].update({
                "status": "completed",
                "text": result["text"],
                "confidence": result["confidence"],
                "processing_time": result["processing_time"],
                "total_time": time.time() - start_time
            })
        else:
            task_results[task_id].update({
                "status": "failed",
                "error_message": result.get("error", "转写失败"),
                "processing_time": result.get("processing_time", 0)
            })
    
    except Exception as e:
        task_results[task_id].update({
            "status": "failed",
            "error_message": f"处理异常: {str(e)}"
        })
    
    finally:
        # 清理临时文件
        audio_processor.clean_temp_file(str(audio_file_path))
        if 'converted_file_path' in locals():
            audio_processor.clean_temp_file(str(converted_file_path))


@router.get("/tasks")
async def list_tasks():
    """列出所有任务 (调试用)"""
    return {
        "total_tasks": len(task_results),
        "tasks": [
            {
                "task_id": task_id,
                "status": data["status"],
                "filename": data["file_info"].get("filename"),
                "created_at": data["created_at"]
            }
            for task_id, data in task_results.items()
        ]
    }
