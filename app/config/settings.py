"""配置管理模块 - 修复版本"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AudioPreprocessingSettings(BaseModel):
    """音频预处理配置"""
    target_sample_rate: int = 16000
    target_channels: int = 2
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


class Settings(BaseSettings):
    """应用配置类 - 完整版本"""
    
    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 9999
    server_debug: bool = True
    
    # 模型配置
    model_name: str = "iic/SenseVoiceSmall"
    model_device: str = "cuda:0"
    model_trust_remote_code: bool = True
    
    # 音频配置
    audio_max_file_size_mb: int = 200
    audio_max_duration_minutes: int = 120  # 新增字段
    audio_supported_formats: List[str] = ["wav", "mp3", "m4a", "flac", "aac"]
    audio_temp_dir: str = "./temp"
    
    # 音频预处理配置 - 新增
    audio_preprocessing: AudioPreprocessingSettings = Field(
        default_factory=AudioPreprocessingSettings
    )
    
    # 音频切片配置 - 新增
    audio_segmentation: AudioSegmentationSettings = Field(
        default_factory=AudioSegmentationSettings
    )
    
    # 处理配置
    processing_timeout_seconds: int = 600
    processing_max_concurrent_tasks: int = 10  # 新增字段
    
    class Config:
        env_file = ".env"


def load_config_from_yaml(config_path: str = "config.yaml") -> Settings:
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 处理嵌套配置
        processed_config = {}
        
        for section, values in config_data.items():
            if isinstance(values, dict):
                if section in ['audio_preprocessing', 'audio_segmentation']:
                    # 保持嵌套结构
                    processed_config[section] = values
                else:
                    # 扁平化其他配置
                    for key, value in values.items():
                        processed_config[f"{section}_{key}"] = value
            else:
                processed_config[section] = values
                
        return Settings(**processed_config)
        
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return Settings()
    except Exception as e:
        print(f"加载配置文件失败: {e}，使用默认配置")
        return Settings()


# 全局配置实例
settings = load_config_from_yaml()
