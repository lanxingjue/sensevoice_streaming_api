"""切片管理API路由"""
from typing import List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ...preprocessing.segment_manager import segment_manager
from ...models.audio_task import AudioTaskStatus
from ...models.segment_task import SegmentStatus
from ..schemas.segment_models import (
    SegmentInfo, 
    AudioSegmentationResult,
    SegmentTranscriptionResult
)

router = APIRouter(prefix="/api/v1/segments", tags=["切片管理"])


@router.get("/audio/{audio_task_id}", response_model=AudioSegmentationResult)
async def get_audio_segments(audio_task_id: str):
    """获取音频的所有切片信息"""
    
    audio_task = segment_manager.get_audio_task(audio_task_id)
    if not audio_task:
        raise HTTPException(status_code=404, detail="音频任务不存在")
    
    segments = segment_manager.get_segments_by_audio(audio_task_id)
    
    # 转换为响应模型
    segment_infos = []
    for seg in segments:
        segment_infos.append(SegmentInfo(
            task_id=seg.task_id,
            segment_index=seg.segment_index,
            start_time=seg.start_time,
            end_time=seg.end_time,
            duration=seg.duration,
            file_size_mb=seg.file_size_mb,
            status=seg.status.value,
            priority=seg.priority,
            has_speech=seg.has_speech,
            quality_score=seg.audio_quality_score
        ))
    
    return AudioSegmentationResult(
        audio_task_id=audio_task_id,
        total_segments=len(segments),
        segments=segment_infos,
        processing_time=audio_task.processing_time or 0.0
    )


@router.get("/ready", response_model=List[SegmentInfo])
async def get_ready_segments(
    limit: int = Query(default=128, le=128, description="返回数量限制")
):
    """获取准备处理的切片任务"""
    
    segments = segment_manager.get_ready_segments(limit=limit)
    
    segment_infos = []
    for seg in segments:
        segment_infos.append(SegmentInfo(
            task_id=seg.task_id,
            segment_index=seg.segment_index,
            start_time=seg.start_time,
            end_time=seg.end_time,
            duration=seg.duration,
            file_size_mb=seg.file_size_mb,
            status=seg.status.value,
            priority=seg.priority,
            has_speech=seg.has_speech,
            quality_score=seg.audio_quality_score
        ))
    
    return segment_infos


@router.get("/{segment_id}", response_model=SegmentTranscriptionResult)
async def get_segment_result(segment_id: str):
    """获取切片转写结果"""
    
    segment = segment_manager.get_segment_task(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="切片任务不存在")
    
    if segment.status != SegmentStatus.COMPLETED:
        raise HTTPException(status_code=202, detail="切片尚未处理完成")
    
    return SegmentTranscriptionResult(
        segment_id=segment.task_id,
        parent_audio_id=segment.parent_audio_id,
        segment_index=segment.segment_index,
        text=segment.transcription_text or "",
        confidence=segment.confidence or 0.0,
        start_time=segment.start_time,
        end_time=segment.end_time,
        processing_time=segment.processing_time or 0.0
    )


@router.get("/statistics/overview")
async def get_statistics():
    """获取切片处理统计信息"""
    
    stats = segment_manager.get_statistics()
    return JSONResponse(stats)
