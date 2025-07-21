"""音频处理工具类"""
import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, target_sr: int = 16000, target_channels: int = 1):
        self.target_sr = target_sr
        self.target_channels = target_channels
    
    def validate_audio_file(self, file_path: str) -> bool:
        """验证音频文件是否有效"""
        try:
            # 尝试读取音频文件信息
            info = sf.info(file_path)
            if info.duration <= 0:
                logger.error(f"音频文件时长无效: {file_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"音频文件验证失败 {file_path}: {e}")
            return False
    
    def get_audio_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取音频文件信息"""
        try:
            info = sf.info(file_path)
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype
            }
        except Exception as e:
            logger.error(f"获取音频信息失败 {file_path}: {e}")
            return None
    
    def convert_audio_format(self, 
                           input_path: str, 
                           output_path: str) -> bool:
        """转换音频格式为标准格式"""
        try:
            # 读取音频文件
            audio_data, original_sr = librosa.load(
                input_path, 
                sr=None,  # 保持原始采样率
                mono=False  # 保持原始声道数
            )
            
            # 确保音频是2D数组 (channels, samples)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.ndim == 2 and audio_data.shape[0] > audio_data.shape[1]:
                audio_data = audio_data.T
            
            # 重采样到目标采样率
            if original_sr != self.target_sr:
                audio_resampled = []
                for channel in range(audio_data.shape[0]):
                    resampled = librosa.resample(
                        audio_data[channel], 
                        orig_sr=original_sr, 
                        target_sr=self.target_sr
                    )
                    audio_resampled.append(resampled)
                audio_data = np.array(audio_resampled)
            
            # 转换声道数
            if self.target_channels == 1 and audio_data.shape[0] > 1:
                # 转为单声道 (平均多声道)
                audio_data = np.mean(audio_data, axis=0)
            elif self.target_channels == 2 and audio_data.shape[0] == 1:
                # 转为双声道 (复制单声道)
                audio_data = np.repeat(audio_data, 2, axis=0)
            
            # 保存转换后的音频
            if audio_data.ndim == 1:
                sf.write(output_path, audio_data, self.target_sr)
            else:
                sf.write(output_path, audio_data.T, self.target_sr)
            
            logger.info(f"音频格式转换成功: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"音频格式转换失败 {input_path}: {e}")
            return False
    
    def clean_temp_file(self, file_path: str) -> None:
        """清理临时文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"临时文件已删除: {file_path}")
        except Exception as e:
            logger.warning(f"删除临时文件失败 {file_path}: {e}")
