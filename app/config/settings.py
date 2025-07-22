"""完整配置管理模块 - 修正版"""
import yaml
from typing import List, Optional, Any, Dict
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioPreprocessingSettings(BaseModel):
    """音频预处理配置"""
    target_sample_rate: int = 16000
    target_channels: int = 1
    target_format: str = "wav"
    enable_noise_reduction: bool = True
    enable_normalization: bool = True
    silence_threshold_db: float = -40.0


class AudioSegmentationSettings(BaseModel):
    """音频切片配置"""
    segment_length_seconds: float = 10.0
    overlap_seconds: float = 2.0
    min_segment_length: float = 3.0
    max_silence_length: float = 1.0
    fade_duration: float = 0.1


class StreamingSettings(BaseModel):
    """流式处理配置"""
    batch_size: int = 128                    # 重要：修复字段名
    batch_timeout_ms: int = 200
    first_segment_priority: int = 10
    normal_segment_priority: int = 1
    max_queue_size: int = 1000
    queue_check_interval_ms: int = 50
    max_concurrent_batches: int = 2
    gpu_memory_threshold: float = 0.9
    enable_performance_monitoring: bool = True


class MonitoringSettings(BaseModel):
    """监控配置"""
    enable_metrics: bool = True
    metrics_interval_seconds: int = 30
    log_batch_performance: bool = True


class Settings(BaseSettings):
    """应用主配置类"""
    
    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_debug: bool = True

    # 模型配置
    model_name: str = "iic/SenseVoiceSmall"
    model_device: str = "cuda:0"
    model_trust_remote_code: bool = True

    # 音频配置
    audio_max_file_size_mb: int = 500
    audio_max_duration_minutes: int = 120
    audio_supported_formats: List[str] = ["wav", "mp3", "m4a", "flac", "aac"]
    audio_temp_dir: str = "./temp"
    audio_segment_length: float = 10.0
    audio_overlap_length: float = 2.0
    audio_target_sample_rate: int = 16000
    audio_target_channels: int = 1

    # 处理配置
    processing_timeout_seconds: int = 1800
    processing_max_concurrent_tasks: int = 10
    processing_chunk_size_mb: int = 1

    # 嵌套配置模型
    audio_preprocessing: AudioPreprocessingSettings = Field(
        default_factory=AudioPreprocessingSettings
    )
    audio_segmentation: AudioSegmentationSettings = Field(
        default_factory=AudioSegmentationSettings
    )
    streaming: StreamingSettings = Field(
        default_factory=StreamingSettings
    )
    monitoring: MonitoringSettings = Field(
        default_factory=MonitoringSettings
    )

    # Pydantic v2 配置
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file_encoding='utf-8'
    )


def load_config_from_yaml(config_path: str = "config.yaml") -> Settings:
    """从YAML文件加载配置并创建Settings实例"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return Settings()

        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

        if not yaml_config:
            print(f"配置文件 {config_path} 为空，使用默认配置")
            return Settings()

        # 处理扁平化配置
        processed_config = {}
        
        for section, values in yaml_config.items():
            if isinstance(values, dict):
                # 对于嵌套配置，保持其结构用于嵌套模型
                if section in ['audio_preprocessing', 'audio_segmentation', 'streaming', 'monitoring']:
                    processed_config[section] = values
                else:
                    # 对于其他配置，进行扁平化处理
                    for key, value in values.items():
                        processed_config[f"{section}_{key}"] = value
            else:
                processed_config[section] = values

        print(f"成功加载配置文件: {config_path}")
        return Settings(**processed_config)

    except yaml.YAMLError as e:
        print(f"YAML解析错误: {e}，使用默认配置")
        return Settings()
    except Exception as e:
        print(f"配置加载失败: {e}，使用默认配置")
        return Settings()


# 创建全局配置实例
settings = load_config_from_yaml()

# 提供向后兼容的属性访问
# 修复dual_queue_scheduler.py中的字段访问问题
settings.streaming_batch_size = settings.streaming.batch_size
settings.streaming_batch_timeout_ms = settings.streaming.batch_timeout_ms
settings.streaming_max_queue_size = settings.streaming.max_queue_size
settings.streaming_max_concurrent_batches = settings.streaming.max_concurrent_batches
