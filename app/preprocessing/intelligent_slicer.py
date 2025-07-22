"""智能音频切片器"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from ..models.segment_task import SegmentTask, SegmentStatus

logger = logging.getLogger(__name__)


class IntelligentAudioSlicer:
    """智能音频切片器"""
    
    def __init__(self,
                 segment_length: float = 10.0,
                 overlap_duration: float = 2.0,
                 min_segment_length: float = 3.0,
                 silence_threshold_db: float = -40.0,
                 max_silence_length: float = 1.0):
        self.segment_length = segment_length
        self.overlap_duration = overlap_duration
        self.min_segment_length = min_segment_length
        self.silence_threshold_db = silence_threshold_db
        self.max_silence_length = max_silence_length
        
    def slice_audio(self, 
                   audio_file_path: str,
                   parent_audio_id: str,
                   output_dir: str) -> List[SegmentTask]:
        """智能切片音频文件"""
        try:
            logger.info(f"开始切片音频: {audio_file_path}")
            
            # 读取音频数据
            audio_data, sr = librosa.load(audio_file_path, sr=None)
            total_duration = len(audio_data) / sr
            
            logger.info(f"音频总时长: {total_duration:.2f}秒, 采样率: {sr}Hz")
            
            # 计算切片位置
            slice_positions = self._calculate_slice_positions(audio_data, sr, total_duration)
            
            # 创建切片任务列表
            segment_tasks = []
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            for i, (start_time, end_time) in enumerate(slice_positions):
                # 生成切片文件
                segment_file_path = output_path / f"{parent_audio_id}_segment_{i:03d}.wav"
                
                success = self._create_segment_file(
                    audio_data, sr, start_time, end_time, str(segment_file_path)
                )
                
                if success:
                    # 创建切片任务
                    segment_task = self._create_segment_task(
                        parent_audio_id=parent_audio_id,
                        segment_index=i,
                        start_time=start_time,
                        end_time=end_time,
                        file_path=str(segment_file_path),
                        total_segments=len(slice_positions)
                    )
                    
                    segment_tasks.append(segment_task)
                    
                    logger.debug(f"创建切片 {i}: {start_time:.2f}s - {end_time:.2f}s")
                else:
                    logger.error(f"切片 {i} 创建失败")
            
            logger.info(f"音频切片完成，共生成 {len(segment_tasks)} 个切片")
            return segment_tasks
            
        except Exception as e:
            logger.error(f"音频切片失败 {audio_file_path}: {e}")
            return []
    
    def _calculate_slice_positions(self, 
                                  audio_data: np.ndarray, 
                                  sr: int, 
                                  total_duration: float) -> List[Tuple[float, float]]:
        """计算智能切片位置"""
        positions = []
        
        # 检测静音区间
        silence_intervals = self._detect_silence_intervals(audio_data, sr)
        
        current_start = 0.0
        
        while current_start < total_duration:
            # 计算理想的结束位置
            ideal_end = current_start + self.segment_length
            
            if ideal_end >= total_duration:
                # 最后一个切片
                if total_duration - current_start >= self.min_segment_length:
                    positions.append((current_start, total_duration))
                else:
                    # 如果最后一段太短，合并到前一个切片
                    if positions:
                        last_start, _ = positions[-1]
                        positions[-1] = (last_start, total_duration)
                break
            
            # 在理想位置附近寻找最佳切分点
            best_split_point = self._find_best_split_point(
                silence_intervals, ideal_end, current_start + self.min_segment_length, 
                min(ideal_end + 2.0, total_duration)
            )
            
            # 添加重叠区域
            actual_end = min(best_split_point + self.overlap_duration, total_duration)
            positions.append((current_start, actual_end))
            
            # 下一个切片的开始位置
            current_start = best_split_point
        
        return positions
    
    def _detect_silence_intervals(self, 
                                 audio_data: np.ndarray, 
                                 sr: int) -> List[Tuple[float, float]]:
        """检测静音区间"""
        try:
            # 转换阈值到线性值
            threshold_linear = 10**(self.silence_threshold_db / 20)
            
            # 计算短时能量
            hop_length = int(0.01 * sr)  # 10ms步长
            frame_length = int(0.025 * sr)  # 25ms帧长
            
            # 使用RMS能量检测
            rms = librosa.feature.rms(
                y=audio_data, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # 找到静音帧
            silence_frames = rms < threshold_linear
            
            # 将帧索引转换为时间
            times = librosa.frames_to_time(
                np.arange(len(silence_frames)), 
                sr=sr, 
                hop_length=hop_length
            )
            
            # 找到连续的静音区间
            silence_intervals = []
            in_silence = False
            silence_start = 0
            
            for i, is_silent in enumerate(silence_frames):
                if is_silent and not in_silence:
                    # 静音开始
                    in_silence = True
                    silence_start = times[i]
                elif not is_silent and in_silence:
                    # 静音结束
                    in_silence = False
                    silence_end = times[i]
                    
                    # 只保留足够长的静音区间
                    if silence_end - silence_start >= 0.1:  # 至少100ms
                        silence_intervals.append((silence_start, silence_end))
            
            # 处理音频结尾的静音
            if in_silence:
                silence_intervals.append((silence_start, times[-1]))
            
            logger.debug(f"检测到 {len(silence_intervals)} 个静音区间")
            return silence_intervals
            
        except Exception as e:
            logger.warning(f"静音检测失败: {e}")
            return []
    
    def _find_best_split_point(self, 
                              silence_intervals: List[Tuple[float, float]],
                              ideal_point: float,
                              min_point: float,
                              max_point: float) -> float:
        """在指定范围内找到最佳切分点"""
        
        # 在搜索范围内寻找静音区间
        candidate_points = []
        
        for silence_start, silence_end in silence_intervals:
            # 静音区间与搜索范围的交集
            overlap_start = max(silence_start, min_point)
            overlap_end = min(silence_end, max_point)
            
            if overlap_start < overlap_end:
                # 选择静音区间的中点作为候选切分点
                mid_point = (overlap_start + overlap_end) / 2
                distance_to_ideal = abs(mid_point - ideal_point)
                candidate_points.append((mid_point, distance_to_ideal))
        
        if candidate_points:
            # 选择距离理想点最近的候选点
            best_point, _ = min(candidate_points, key=lambda x: x[1])
            return best_point
        else:
            # 没有合适的静音区间，使用理想点
            return ideal_point
    
    def _create_segment_file(self, 
                            audio_data: np.ndarray,
                            sr: int,
                            start_time: float,
                            end_time: float,
                            output_path: str) -> bool:
        """创建切片文件"""
        try:
            # 计算样本位置
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # 确保索引在有效范围内
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if start_sample >= end_sample:
                logger.error(f"无效的切片范围: {start_sample} >= {end_sample}")
                return False
            
            # 提取音频段
            segment_audio = audio_data[start_sample:end_sample]
            
            # 应用淡入淡出，避免点击声
            segment_audio = self._apply_fade(segment_audio, sr)
            
            # 保存切片
            sf.write(output_path, segment_audio, sr)
            
            return True
            
        except Exception as e:
            logger.error(f"创建切片文件失败 {output_path}: {e}")
            return False
    
    def _apply_fade(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """应用淡入淡出效果"""
        fade_duration = 0.01  # 10ms淡入淡出
        fade_samples = int(fade_duration * sr)
        
        if len(audio) <= 2 * fade_samples:
            return audio
        
        # 淡入
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in
        
        # 淡出
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out
        
        return audio
    
    def _create_segment_task(self,
                            parent_audio_id: str,
                            segment_index: int,
                            start_time: float,
                            end_time: float,
                            file_path: str,
                            total_segments: int) -> SegmentTask:
        """创建切片任务对象"""
        
        # 生成切片任务ID
        task_id = f"{parent_audio_id}_seg_{segment_index:03d}"
        
        # 计算重叠区域
        overlap_start = 0.0
        overlap_end = 0.0
        
        if segment_index > 0:  # 不是第一个切片
            overlap_start = start_time
            overlap_end = min(start_time + self.overlap_duration, end_time)
        
        # 获取文件大小
        file_size_mb = 0.0
        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        except:
            pass
        
        # 设置优先级：首片段最高优先级
        priority = 10 if segment_index == 0 else 1
        
        return SegmentTask(
            task_id=task_id,
            parent_audio_id=parent_audio_id,
            segment_index=segment_index,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            file_path=file_path,
            file_size_mb=file_size_mb,
            status=SegmentStatus.CREATED,
            priority=priority
        )
    
    def analyze_segment_quality(self, segment_task: SegmentTask) -> None:
        """分析切片质量"""
        try:
            audio_data, sr = librosa.load(segment_task.file_path)
            
            # 检测是否包含语音
            segment_task.has_speech = self._detect_speech(audio_data, sr)
            
            # 计算静音比例
            segment_task.silence_ratio = self._calculate_silence_ratio(audio_data, sr)
            
            # 计算质量评分
            segment_task.audio_quality_score = self._calculate_segment_quality(audio_data, sr)
            
            logger.debug(f"切片质量分析完成 {segment_task.task_id}: "
                        f"包含语音={segment_task.has_speech}, "
                        f"静音比例={segment_task.silence_ratio:.2f}, "
                        f"质量评分={segment_task.audio_quality_score:.2f}")
            
        except Exception as e:
            logger.warning(f"切片质量分析失败 {segment_task.task_id}: {e}")
    
    def _detect_speech(self, audio_data: np.ndarray, sr: int) -> bool:
        """检测音频中是否包含语音"""
        try:
            # 简单的语音活动检测
            # 计算短时能量和谱质心
            rms_energy = np.sqrt(np.mean(audio_data**2))
            
            if rms_energy < 10**(-50/20):  # 非常低的能量
                return False
            
            # 计算谱质心
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            mean_centroid = np.mean(spectral_centroid)
            
            # 语音通常在300-3000Hz范围内有较强能量
            if 300 < mean_centroid < 3000:
                return True
            
            return False
            
        except Exception:
            # 出错时假设包含语音
            return True
    
    def _calculate_silence_ratio(self, audio_data: np.ndarray, sr: int) -> float:
        """计算静音比例"""
        try:
            threshold_linear = 10**(self.silence_threshold_db / 20)
            
            # 使用RMS计算
            hop_length = int(0.01 * sr)
            rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
            
            silence_frames = np.sum(rms < threshold_linear)
            total_frames = len(rms)
            
            return silence_frames / total_frames if total_frames > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_segment_quality(self, audio_data: np.ndarray, sr: int) -> float:
        """计算切片质量评分"""
        try:
            score = 1.0
            
            # 基于能量的评分
            rms_energy = np.sqrt(np.mean(audio_data**2))
            if rms_energy < 10**(-40/20):  # 太安静
                score *= 0.5
            elif rms_energy > 0.5:  # 可能削波
                score *= 0.8
            
            # 基于时长的评分
            duration = len(audio_data) / sr
            if duration < 1.0:  # 太短
                score *= 0.7
            
            return float(np.clip(score, 0.0, 1.0))
            
        except Exception:
            return 0.8
