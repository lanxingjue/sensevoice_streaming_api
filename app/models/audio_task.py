"""简化的音频任务模型"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import time


class TaskStatus(Enum):
    """任务状态"""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    SLICING = "slicing"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AudioTask:
    """简化的音频任务"""
    task_id: str
    filename: str
    file_path: str
    file_size_mb: float
    status: TaskStatus = TaskStatus.UPLOADING
    
    # 切片信息
    duration_seconds: float = 0.0
    segment_count: int = 0
    segments: List[str] = field(default_factory=list)
    
    # 时间信息
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 结果信息
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    def update_status(self, status: TaskStatus, message: str = None):
        """更新状态"""
        self.status = status
        self.updated_at = time.time()
        if message:
            self.error_message = message
