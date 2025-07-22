"""流式处理相关数据模型"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class StreamingStatus(BaseModel):
    """流式处理系统状态"""
    is_running: bool = Field(..., description="是否运行中")
    uptime_seconds: float = Field(..., description="运行时间(秒)")
    total_processed_batches: int = Field(..., description="已处理批次总数")
    avg_batches_per_minute: float = Field(..., description="平均每分钟处理批次数")
    active_workers: int = Field(..., description="活跃工作线程数")
    queue_size: int = Field(..., description="队列中待处理任务数")
    gpu_utilization: float = Field(..., description="GPU利用率(%)")


class BatchProcessingStats(BaseModel):
    """批处理统计信息"""
    queue_stats: Dict[str, Any] = Field(..., description="队列统计")
    inference_stats: Dict[str, Any] = Field(..., description="推理统计")
    dispatch_stats: Dict[str, Any] = Field(..., description="分发统计")
    gpu_status: Dict[str, Any] = Field(..., description="GPU状态")


class FirstSegmentResult(BaseModel):
    """首片段转写结果"""
    audio_id: str = Field(..., description="音频ID")
    segment_id: str = Field(..., description="切片ID")
    text: str = Field(..., description="转写文本")
    confidence: float = Field(..., description="置信度")
    start_time: float = Field(..., description="开始时间(秒)")
    end_time: float = Field(..., description="结束时间(秒)")
    processing_time: float = Field(..., description="处理时间(秒)")
    is_ready: bool = Field(..., description="是否已完成")


class SegmentResultResponse(BaseModel):
    """切片结果响应"""
    segment_id: str = Field(..., description="切片ID")
    parent_audio_id: str = Field(..., description="父音频ID")
    segment_index: int = Field(..., description="切片序号")
    text: str = Field(..., description="转写文本")
    confidence: float = Field(..., description="置信度")
    start_time: float = Field(..., description="开始时间(秒)")
    end_time: float = Field(..., description="结束时间(秒)")
    duration: float = Field(..., description="时长(秒)")
    processing_time: float = Field(..., description="处理时间(秒)")
