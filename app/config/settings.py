"""配置管理模块"""
from pathlib import Path
from typing import List
import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 9999
    server_debug: bool = True
    
    # 模型配置
    model_name: str = "iic/SenseVoiceSmall"
    model_device: str = "cuda:0"
    model_trust_remote_code: bool = True
    
    # 音频配置
    audio_max_file_size_mb: int = 100
    audio_supported_formats: List[str] = ["wav", "mp3", "m4a", "flac"]
    audio_temp_dir: str = "./temp"
    
    # 处理配置
    processing_timeout_seconds: int = 300
    
    class Config:
        env_file = ".env"


def load_config_from_yaml(config_path: str = "config.yaml") -> Settings:
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 扁平化配置字典
        flat_config = {}
        for section, values in config_data.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_config[f"{section}_{key}"] = value
            else:
                flat_config[section] = values
                
        return Settings(**flat_config)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return Settings()
    except Exception as e:
        print(f"加载配置文件失败: {e}，使用默认配置")
        return Settings()


# 全局配置实例
settings = load_config_from_yaml()
