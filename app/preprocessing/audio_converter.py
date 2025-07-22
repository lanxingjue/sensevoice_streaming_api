"""增强音频转换器"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedAudioConverter:
    """增强音频转换器"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 target_channels: int = 2,
                 enable_enhancement: bool = True):
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.enable_enhancement = enable_enhancement
    
    def analyze_audio(self, file_path: str) -> Optional[Dict[str, Any]]:
        """分析音频文件详细信息"""
        try:
            # 获取基本信息
            info = sf.info(file_path)
            
            # 读取音频数据进行深度分析
            audio_data, sr = librosa.load(file_path, sr=None, mono=False)
            
            # 确保是2D数组
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            
            # 计算音频质量指标
            analysis = {
                # 基本信息
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
                
                # 质量分析
                "rms_energy": float(np.sqrt(np.mean(audio_data**2))),
                "peak_amplitude": float(np.max(np.abs(audio_data))),
                "dynamic_range": self._calculate_dynamic_range(audio_data),
                "silence_ratio": self._calculate_silence_ratio(audio_data, sr),
                "frequency_analysis": self._analyze_frequency_content(audio_data, sr),
                
                # 质量评分
                "quality_score": self._calculate_quality_score(audio_data, sr)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"音频分析失败 {file_path}: {e}")
            return None
    
    def convert_with_enhancement(self, 
                               input_path: str, 
                               output_path: str) -> bool:
        """增强音频转换"""
        try:
            logger.info(f"开始增强转换: {input_path} -> {output_path}")
            
            # 读取原始音频
            audio_data, original_sr = librosa.load(
                input_path, 
                sr=None, 
                mono=False
            )
            
            # 确保音频是2D格式 [channels, samples]
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.ndim == 2 and audio_data.shape[0] > audio_data.shape[1]:
                audio_data = audio_data.T
            
            # 音质增强处理
            if self.enable_enhancement:
                audio_data = self._enhance_audio_quality(audio_data, original_sr)
            
            # 重采样
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
            
            # 声道转换
            audio_data = self._convert_channels(audio_data)
            
            # 最终质量优化
            audio_data = self._final_quality_optimization(audio_data)
            
            # 保存音频
            if audio_data.ndim == 1:
                sf.write(output_path, audio_data, self.target_sr)
            else:
                sf.write(output_path, audio_data.T, self.target_sr)
            
            logger.info(f"音频转换完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"增强音频转换失败 {input_path}: {e}")
            return False
    
    def _enhance_audio_quality(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """音质增强处理"""
        enhanced_audio = []
        
        for channel in range(audio_data.shape[0]):
            channel_data = audio_data[channel]
            
            # 降噪处理
            channel_data = self._reduce_noise(channel_data, sr)
            
            # 动态范围压缩
            channel_data = self._compress_dynamic_range(channel_data)
            
            # 音量标准化
            channel_data = self._normalize_volume(channel_data)
            
            enhanced_audio.append(channel_data)
        
        return np.array(enhanced_audio)
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """简单降噪处理"""
        try:
            # 使用谱减法进行降噪
            # 1. 短时傅里叶变换
            stft = librosa.stft(audio, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 2. 估算噪声谱（使用音频开头的静音部分）
            noise_frames = min(20, magnitude.shape[1] // 10)  # 前10%或前20帧
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # 3. 谱减法
            alpha = 2.0  # 过减因子
            beta = 0.01  # 剩余噪声因子
            
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # 4. 重构音频
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"降噪处理失败，使用原音频: {e}")
            return audio
    
    def _compress_dynamic_range(self, audio: np.ndarray, 
                              threshold: float = 0.7,
                              ratio: float = 4.0) -> np.ndarray:
        """动态范围压缩"""
        try:
            # 简单压缩器实现
            compressed = audio.copy()
            
            # 找到超过阈值的样本
            mask = np.abs(compressed) > threshold
            
            # 应用压缩比
            compressed[mask] = np.sign(compressed[mask]) * (
                threshold + (np.abs(compressed[mask]) - threshold) / ratio
            )
            
            return compressed
            
        except Exception as e:
            logger.warning(f"动态范围压缩失败: {e}")
            return audio
    
    def _normalize_volume(self, audio: np.ndarray, target_level: float = -12.0) -> np.ndarray:
        """音量标准化（到指定dB级别）"""
        try:
            # 计算当前RMS电平
            rms = np.sqrt(np.mean(audio**2))
            
            if rms > 0:
                # 计算当前dB电平
                current_db = 20 * np.log10(rms)
                
                # 计算增益
                gain_db = target_level - current_db
                gain_linear = 10**(gain_db / 20)
                
                # 应用增益，但避免削波
                normalized = audio * gain_linear
                max_val = np.max(np.abs(normalized))
                
                if max_val > 0.95:  # 防止削波
                    normalized = normalized * (0.95 / max_val)
                
                return normalized
            
            return audio
            
        except Exception as e:
            logger.warning(f"音量标准化失败: {e}")
            return audio
    
    def _convert_channels(self, audio_data: np.ndarray) -> np.ndarray:
        """声道转换"""
        current_channels = audio_data.shape[0]
        
        if current_channels == self.target_channels:
            return audio_data
        elif self.target_channels == 1 and current_channels > 1:
            # 转为单声道（平均多声道）
            return np.mean(audio_data, axis=0)
        elif self.target_channels == 2 and current_channels == 1:
            # 转为双声道（复制单声道）
            return np.repeat(audio_data, 2, axis=0)
        elif self.target_channels == 2 and current_channels > 2:
            # 多声道转双声道（取前两个声道）
            return audio_data[:2]
        else:
            logger.warning(f"不支持的声道转换: {current_channels} -> {self.target_channels}")
            return audio_data
    
    def _final_quality_optimization(self, audio_data: np.ndarray) -> np.ndarray:
        """最终质量优化"""
        # 应用轻微的淡入淡出，避免点击声
        fade_samples = int(0.01 * self.target_sr)  # 10ms淡入淡出
        
        if audio_data.ndim == 1:
            # 单声道
            if len(audio_data) > 2 * fade_samples:
                # 淡入
                fade_in = np.linspace(0, 1, fade_samples)
                audio_data[:fade_samples] *= fade_in
                
                # 淡出
                fade_out = np.linspace(1, 0, fade_samples)
                audio_data[-fade_samples:] *= fade_out
        else:
            # 多声道
            for channel in range(audio_data.shape[0]):
                if audio_data.shape[1] > 2 * fade_samples:
                    # 淡入
                    fade_in = np.linspace(0, 1, fade_samples)
                    audio_data[channel, :fade_samples] *= fade_in
                    
                    # 淡出  
                    fade_out = np.linspace(1, 0, fade_samples)
                    audio_data[channel, -fade_samples:] *= fade_out
        
        return audio_data
    
    def _calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """计算动态范围"""
        if audio_data.size == 0:
            return 0.0
        
        peak = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms > 0:
            return float(20 * np.log10(peak / rms))
        return 0.0
    
    def _calculate_silence_ratio(self, audio_data: np.ndarray, sr: int, 
                               threshold_db: float = -40.0) -> float:
        """计算静音比例"""
        try:
            # 转换阈值到线性值
            threshold_linear = 10**(threshold_db / 20)
            
            # 计算每个样本的RMS（使用滑动窗口）
            window_size = int(0.1 * sr)  # 100ms窗口
            rms_values = []
            
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            if len(rms_values) == 0:
                return 0.0
            
            # 计算静音帧比例
            silence_frames = sum(1 for rms in rms_values if rms < threshold_linear)
            return silence_frames / len(rms_values)
            
        except Exception as e:
            logger.warning(f"静音比例计算失败: {e}")
            return 0.0
    
    def _analyze_frequency_content(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """分析频率内容"""
        try:
            # 使用第一个声道进行分析
            if audio_data.ndim > 1:
                audio_for_analysis = audio_data[0]
            else:
                audio_for_analysis = audio_data
            
            # 计算频谱
            fft = np.fft.rfft(audio_for_analysis)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_for_analysis), 1/sr)
            
            # 计算各频段能量
            def energy_in_band(low_freq, high_freq):
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                return float(np.sum(magnitude[mask]**2))
            
            total_energy = np.sum(magnitude**2)
            
            if total_energy > 0:
                return {
                    "low_freq_energy": energy_in_band(0, 500) / total_energy,      # 低频
                    "mid_freq_energy": energy_in_band(500, 2000) / total_energy,   # 中频
                    "high_freq_energy": energy_in_band(2000, 8000) / total_energy, # 高频
                    "spectral_centroid": float(librosa.feature.spectral_centroid(
                        y=audio_for_analysis, sr=sr)[0].mean())
                }
            
            return {
                "low_freq_energy": 0.0,
                "mid_freq_energy": 0.0, 
                "high_freq_energy": 0.0,
                "spectral_centroid": 0.0
            }
            
        except Exception as e:
            logger.warning(f"频率分析失败: {e}")
            return {
                "low_freq_energy": 0.0,
                "mid_freq_energy": 0.0,
                "high_freq_energy": 0.0, 
                "spectral_centroid": 0.0
            }
    
    def _calculate_quality_score(self, audio_data: np.ndarray, sr: int) -> float:
        """计算音频质量评分 (0-1)"""
        try:
            score = 1.0
            
            # 基于动态范围的评分
            dynamic_range = self._calculate_dynamic_range(audio_data)
            if dynamic_range < 10:  # 动态范围太小
                score *= 0.8
            elif dynamic_range > 60:  # 动态范围太大
                score *= 0.9
            
            # 基于静音比例的评分
            silence_ratio = self._calculate_silence_ratio(audio_data, sr)
            if silence_ratio > 0.3:  # 静音过多
                score *= (1.0 - silence_ratio * 0.5)
            
            # 基于削波检测的评分
            peak = np.max(np.abs(audio_data))
            if peak > 0.98:  # 可能存在削波
                score *= 0.7
            
            # 基于采样率的评分
            if sr < 16000:  # 采样率过低
                score *= 0.8
            
            return float(np.clip(score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"质量评分计算失败: {e}")
            return 0.8  # 默认评分
