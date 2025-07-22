"""简化的切片任务模型"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time


class SegmentStatus(Enum):
    """切片状态"""
    CREATED = "created"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SegmentTask:
    """简化的切片任务"""
    segment_id: str
    parent_id: str
    index: int
    file_path: str
    
    # 时间信息
    start_time: float
    end_time: float
    duration: float
    
    # 处理信息
    status: SegmentStatus = SegmentStatus.CREATED
    priority: int = 1  # 首片段优先级为10
    created_at: float = field(default_factory=time.time)
    
    # 结果
    text: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def is_first_segment(self) -> bool:
        return self.index == 0
    
    def set_result(self, text: str, confidence: float, proc_time: float):
        """设置转写结果"""
        self.text = text
        self.confidence = confidence
        self.processing_time = proc_time
        self.status = SegmentStatus.COMPLETED
    