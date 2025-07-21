"""API请求响应模型"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AudioUploadResponse(BaseModel):
    """音频上传响应模型"""
    task_id: str = Field(..., description="任务唯一标识")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="响应消息")
    file_info: Dict[str, Any] = Field(..., description="文件信息")


class TranscriptionResult(BaseModel):
    """转写结果模型"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="处理状态")
    text: Optional[str] = Field(None, description="转写文本")
    confidence: Optional[float] = Field(None, description="置信度")
    processing_time: Optional[float] = Field(None, description="处理时间(秒)")
    error_message: Optional[str] = Field(None, description="错误信息")


class HealthCheck(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    message: str = Field(..., description="状态描述")
    timestamp: float = Field(..., description="检查时间戳")
    version: str = Field(..., description="服务版本")
