"""批处理结果数据模型"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class BatchStatus(Enum):
    """批处理状态"""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SegmentResult:
    """单个切片的转写结果"""
    segment_id: str
    parent_audio_id: str
    segment_index: int
    
    # 转写结果
    text: str
    confidence: float
    processing_time: float
    
    # 时间信息
    start_time: float
    end_time: float
    duration: float
    
    # 元数据
    is_first_segment: bool
    priority: int
    file_path: str
    
    # 质量信息
    audio_quality_score: float = 1.0
    has_speech: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "segment_id": self.segment_id,
            "parent_audio_id": self.parent_audio_id,
            "segment_index": self.segment_index,
            "text": self.text,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "is_first_segment": self.is_first_segment,
            "priority": self.priority
        }


@dataclass
class BatchResult:
    """批处理结果"""
    batch_id: str
    status: BatchStatus
    
    # 批次信息
    batch_size: int
    first_segments_count: int
    normal_segments_count: int
    
    # 时间信息
    created_at: float = field(default_factory=time.time)
    processing_start: Optional[float] = None
    processing_end: Optional[float] = None
    
    # 结果
    segment_results: List[SegmentResult] = field(default_factory=list)
    failed_segments: List[str] = field(default_factory=list)
    
    # 性能指标
    gpu_inference_time: float = 0.0
    total_processing_time: float = 0.0
    gpu_memory_used: float = 0.0
    
    # 错误信息
    error_message: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.batch_size == 0:
            return 0.0
        return len(self.segment_results) / self.batch_size
    
    @property
    def throughput(self) -> float:
        """吞吐量 (segments/second)"""
        if self.total_processing_time <= 0:
            return 0.0
        return self.batch_size / self.total_processing_time
    
    def get_first_segments(self) -> List[SegmentResult]:
        """获取首片段结果"""
        return [result for result in self.segment_results if result.is_first_segment]
    
    def get_normal_segments(self) -> List[SegmentResult]:
        """获取普通片段结果"""
        return [result for result in self.segment_results if not result.is_first_segment]
