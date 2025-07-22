"""切片任务数据模型"""
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time


class SegmentStatus(Enum):
    """切片任务状态"""
    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SegmentTask:
    """音频切片任务"""
    task_id: str
    parent_audio_id: str
    segment_index: int
    
    # 时间信息
    start_time: float           # 在原音频中的开始时间(秒)
    end_time: float            # 在原音频中的结束时间(秒)
    duration: float            # 切片实际时长
    overlap_start: float       # 重叠区域开始位置
    overlap_end: float         # 重叠区域结束位置
    
    # 文件信息
    file_path: str
    file_size_mb: float
    
    # 处理状态
    status: SegmentStatus = SegmentStatus.CREATED
    priority: int = 1          # 优先级，首片段为10
    created_at: float = time.time()
    updated_at: float = time.time()
    
    # 转写结果
    transcription_text: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    # 质量信息
    has_speech: bool = True    # 是否包含语音
    silence_ratio: float = 0.0 # 静音比例
    audio_quality_score: float = 1.0  # 音频质量评分
    
    @property
    def is_first_segment(self) -> bool:
        """是否为首片段"""
        return self.segment_index == 0
    
    @property
    def has_overlap_with_previous(self) -> bool:
        """是否与前一片段有重叠"""
        return self.segment_index > 0
    
    def update_status(self, status: SegmentStatus):
        """更新状态"""
        self.status = status
        self.updated_at = time.time()
    
    def set_transcription_result(self, text: str, confidence: float, processing_time: float):
        """设置转写结果"""
        self.transcription_text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.status = SegmentStatus.COMPLETED
        self.updated_at = time.time()
    
    def set_error(self, error_message: str):
        """设置错误状态"""
        self.error_message = error_message
        self.status = SegmentStatus.FAILED
        self.updated_at = time.time()
