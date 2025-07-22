"""简化配置管理"""
from typing import List
import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """简化的配置类"""
    
    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 9999
    server_debug: bool = True
    
    # 模型配置
    model_name: str = "iic/SenseVoiceSmall"
    model_device: str = "cuda:0"
    model_trust_remote_code: bool = True
    
    # 音频配置
    audio_max_file_size_mb: int = 500
    audio_supported_formats: List[str] = ["wav", "mp3", "m4a", "flac", "aac"]
    audio_temp_dir: str = "./temp"
    audio_segment_length: float = 10.0
    audio_overlap_length: float = 2.0
    audio_target_sample_rate: int = 16000
    audio_target_channels: int = 1
    
    # 处理配置
    processing_timeout_seconds: int = 1800
    processing_chunk_size_mb: int = 1
    
    class Config:
        env_file = ".env"


def load_config() -> Settings:
    """从YAML加载配置"""
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 扁平化配置
        flat_config = {}
        for section, values in config_data.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_config[f"{section}_{key}"] = value
            else:
                flat_config[section] = values
                
        return Settings(**flat_config)
        
    except Exception as e:
        print(f"配置加载失败: {e}，使用默认配置")
        return Settings()


# 全局配置
settings = load_config()
