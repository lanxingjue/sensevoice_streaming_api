"""切片相关数据模型"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class SegmentStatusEnum(str, Enum):
    """切片状态枚举"""
    CREATED = "created"
    QUEUED = "queued" 
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SegmentInfo(BaseModel):
    """切片信息模型"""
    task_id: str = Field(..., description="切片任务ID")
    segment_index: int = Field(..., description="切片序号")
    start_time: float = Field(..., description="开始时间(秒)")
    end_time: float = Field(..., description="结束时间(秒)")
    duration: float = Field(..., description="时长(秒)")
    file_size_mb: float = Field(..., description="文件大小(MB)")
    status: SegmentStatusEnum = Field(..., description="处理状态")
    priority: int = Field(..., description="优先级")
    has_speech: bool = Field(..., description="是否包含语音")
    quality_score: float = Field(..., description="质量评分")


class AudioSegmentationResult(BaseModel):
    """音频切片结果模型"""
    audio_task_id: str = Field(..., description="音频任务ID")
    total_segments: int = Field(..., description="切片总数") 
    segments: List[SegmentInfo] = Field(..., description="切片列表")
    processing_time: float = Field(..., description="处理时间(秒)")


class SegmentTranscriptionResult(BaseModel):
    """切片转写结果模型"""
    segment_id: str = Field(..., description="切片ID")
    parent_audio_id: str = Field(..., description="父音频ID") 
    segment_index: int = Field(..., description="切片序号")
    text: str = Field(..., description="转写文本")
    confidence: float = Field(..., description="置信度")
    start_time: float = Field(..., description="开始时间(秒)")
    end_time: float = Field(..., description="结束时间(秒)")
    processing_time: float = Field(..., description="处理时间(秒)")
