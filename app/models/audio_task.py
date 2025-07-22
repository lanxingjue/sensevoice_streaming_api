"""音频任务数据模型"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class AudioTaskStatus(Enum):
    """音频任务状态枚举"""
    UPLOADED = "uploaded"           # 已上传
    PREPROCESSING = "preprocessing" # 预处理中
    SEGMENTED = "segmented"        # 已切片
    PROCESSING = "processing"       # 处理中
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"              # 处理失败


@dataclass
class AudioMetadata:
    """音频元数据"""
    filename: str
    file_size_mb: float
    duration_seconds: float
    original_sample_rate: int
    original_channels: int
    original_format: str
    target_sample_rate: int = 16000
    target_channels: int = 2
    target_format: str = "wav"


@dataclass  
class AudioTask:
    """音频处理任务"""
    task_id: str
    status: AudioTaskStatus
    metadata: AudioMetadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 文件路径信息
    original_file_path: Optional[str] = None
    processed_file_path: Optional[str] = None
    
    # 切片信息
    segment_count: int = 0
    segment_tasks: List[str] = field(default_factory=list)  # 切片任务ID列表
    
    # 处理结果
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    final_text: Optional[str] = None
    
    # 进度信息
    progress_percent: float = 0.0
    current_step: str = "等待处理"
    
    def update_status(self, status: AudioTaskStatus, message: str = None):
        """更新任务状态"""
        self.status = status
        self.updated_at = time.time()
        if message:
            self.current_step = message
    
    def set_error(self, error_message: str):
        """设置错误状态"""
        self.status = AudioTaskStatus.FAILED
        self.error_message = error_message
        self.updated_at = time.time()
    
    def get_estimated_segments(self) -> int:
        """估算切片数量"""
        if self.metadata.duration_seconds <= 0:
            return 0
        
        segment_length = 10.0  # 配置中的切片长度
        overlap = 2.0         # 重叠长度
        effective_length = segment_length - overlap
        
        return int((self.metadata.duration_seconds - overlap) / effective_length) + 1
