"""流式处理API接口"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional

from ...streaming.batch_processor import batch_processor
from ...streaming.result_dispatcher import result_dispatcher
from ...preprocessing.task_manager import task_manager
from ...models.segment_task import SegmentStatus
from ..schemas.streaming_models import (
    StreamingStatus,
    BatchProcessingStats,
    FirstSegmentResult,
    SegmentResultResponse
)

router = APIRouter(prefix="/api/v1/streaming", tags=["流式处理"])


@router.post("/start")
async def start_streaming():
    """启动流式处理系统"""
    try:
        await batch_processor.start()
        return {
            "message": "流式处理系统启动成功",
            "status": "running",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.post("/stop")
async def stop_streaming():
    """停止流式处理系统"""
    try:
        await batch_processor.stop()
        return {
            "message": "流式处理系统已停止",
            "status": "stopped",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")


@router.get("/status", response_model=StreamingStatus)
async def get_streaming_status():
    """获取流式处理系统状态"""
    status = batch_processor.get_status()
    
    return StreamingStatus(
        is_running=status["is_running"],
        uptime_seconds=status["uptime_seconds"],
        total_processed_batches=status["total_processed_batches"],
        avg_batches_per_minute=status["avg_batches_per_minute"],
        active_workers=status["active_workers"],
        queue_size=status["queue_stats"]["total_queued"],
        gpu_utilization=status["gpu_status"].get("memory_utilization", 0)
    )


@router.get("/stats", response_model=BatchProcessingStats)
async def get_processing_stats():
    """获取详细的处理统计"""
    status = batch_processor.get_status()
    
    return BatchProcessingStats(
        queue_stats=status["queue_stats"],
        inference_stats=status["inference_stats"],
        dispatch_stats=status["dispatch_stats"],
        gpu_status=status["gpu_status"]
    )


@router.post("/submit/{audio_task_id}")
async def submit_audio_for_streaming(audio_task_id: str):
    """提交音频任务到流式处理队列"""
    
    # 获取音频任务
    audio_task = task_manager.get_audio_task(audio_task_id)
    if not audio_task:
        raise HTTPException(status_code=404, detail="音频任务不存在")
    
    # 检查任务状态
    if audio_task.status.value != "ready":
        raise HTTPException(
            status_code=400, 
            detail=f"音频任务状态不正确: {audio_task.status.value}，需要为 ready"
        )
    
    # 获取所有切片任务
    segments = task_manager.get_segments_by_audio(audio_task_id)
    if not segments:
        raise HTTPException(status_code=400, detail="没有找到切片任务")
    
    # 提交切片到流式处理队列
    submitted_count = 0
    failed_count = 0
    
    for segment in segments:
        if segment.status == SegmentStatus.READY:
            success = await batch_processor.add_segment_task(segment)
            if success:
                submitted_count += 1
            else:
                failed_count += 1
    
    if submitted_count == 0:
        raise HTTPException(status_code=500, detail="没有切片成功提交到处理队列")
    
    # 更新音频任务状态
    audio_task.update_status(TaskStatus.PROCESSING, f"已提交 {submitted_count} 个切片到处理队列")
    
    return {
        "message": f"音频任务已提交到流式处理队列",
        "audio_task_id": audio_task_id,
        "submitted_segments": submitted_count,
        "failed_segments": failed_count,
        "total_segments": len(segments),
        "estimated_completion_time": submitted_count * 0.2  # 估算完成时间
    }


@router.get("/first-segment/{audio_id}", response_model=FirstSegmentResult)
async def get_first_segment_result(audio_id: str):
    """获取音频的首片段转写结果"""
    
    result = result_dispatcher.get_first_segment_result(audio_id)
    
    if not result:
        raise HTTPException(
            status_code=202, 
            detail="首片段结果尚未完成，请稍后查询"
        )
    
    return FirstSegmentResult(
        audio_id=audio_id,
        segment_id=result.segment_id,
        text=result.text,
        confidence=result.confidence,
        start_time=result.start_time,
        end_time=result.end_time,
        processing_time=result.processing_time,
        is_ready=True
    )


@router.get("/segments/{segment_id}", response_model=SegmentResultResponse)
async def get_segment_result(segment_id: str):
    """获取特定切片的转写结果"""
    
    result = result_dispatcher.get_segment_result(segment_id)
    
    if not result:
        raise HTTPException(
            status_code=404, 
            detail="切片结果不存在或尚未完成"
        )
    
    return SegmentResultResponse(
        segment_id=result.segment_id,
        parent_audio_id=result.parent_audio_id,
        segment_index=result.segment_index,
        text=result.text,
        confidence=result.confidence,
        start_time=result.start_time,
        end_time=result.end_time,
        duration=result.duration,
        processing_time=result.processing_time
    )


@router.get("/audio/{audio_id}/segments")
async def get_audio_segments_results(audio_id: str):
    """获取音频的所有已完成切片结果"""
    
    results = result_dispatcher.get_audio_segments_results(audio_id)
    
    return {
        "audio_id": audio_id,
        "completed_segments": len(results),
        "results": [
            {
                "segment_id": r.segment_id,
                "segment_index": r.segment_index,
                "text": r.text,
                "confidence": r.confidence,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "is_first_segment": r.is_first_segment
            }
            for r in results
        ]
    }


@router.get("/performance")
async def get_performance_metrics():
    """获取性能指标"""
    metrics = batch_processor.get_performance_metrics()
    
    return {
        "timestamp": time.time(),
        "metrics": metrics,
        "system_load": {
            "queue_pressure": metrics.get("scheduler_metrics", {}).get("avg_queue_wait_time", 0),
            "inference_efficiency": metrics.get("inference_metrics", {}).get("avg_batch_time", 0),
            "dispatch_speed": metrics.get("dispatch_metrics", {}).get("avg_dispatch_time_ms", 0)
        }
    }


@router.post("/cleanup")
async def cleanup_old_results(max_age_hours: float = Query(default=1.0, ge=0.1, le=24.0)):
    """清理过期结果"""
    
    max_age_seconds = max_age_hours * 3600
    
    try:
        result_dispatcher.cleanup_old_results(max_age_seconds)
        return {
            "message": f"清理完成，清理超过 {max_age_hours} 小时的结果",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")
