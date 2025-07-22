"""简化的音频切片器"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

from ..models.segment_task import SegmentTask, SegmentStatus

logger = logging.getLogger(__name__)


class SimpleSlicer:
    """简化的音频切片器"""
    
    def __init__(self, 
                 segment_length: float = 10.0,
                 overlap_length: float = 2.0,
                 target_sr: int = 16000,
                 target_channels: int = 1):
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self.target_sr = target_sr
        self.target_channels = target_channels
    
    def slice_audio(self, 
                   audio_path: str, 
                   parent_id: str,
                   output_dir: str) -> List[SegmentTask]:
        """
        简单切片音频文件
        """
        try:
            logger.info(f"开始切片: {audio_path}")
            
            # 加载音频并转换格式
            audio_data, sr = librosa.load(
                audio_path,
                sr=self.target_sr,
                mono=(self.target_channels == 1)
            )
            
            total_duration = len(audio_data) / sr
            logger.info(f"音频加载完成: {total_duration:.2f}秒, {sr}Hz")
            
            # 计算切片位置
            segments_info = self._calculate_segments(total_duration)
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 生成切片文件和任务
            segment_tasks = []
            
            for i, (start_time, end_time) in enumerate(segments_info):
                # 计算样本位置
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # 提取音频段
                segment_audio = audio_data[start_sample:end_sample]
                
                # 保存切片文件
                segment_filename = f"{parent_id}_seg_{i:03d}.wav"
                segment_path = output_path / segment_filename
                
                sf.write(str(segment_path), segment_audio, sr)
                
                # 创建切片任务
                task = SegmentTask(
                    segment_id=f"{parent_id}_seg_{i:03d}",
                    parent_id=parent_id,
                    index=i,
                    file_path=str(segment_path),
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    status=SegmentStatus.READY,
                    priority=10 if i == 0 else 1  # 首片段高优先级
                )
                
                segment_tasks.append(task)
                logger.debug(f"切片 {i}: {start_time:.1f}s-{end_time:.1f}s")
            
            logger.info(f"切片完成: {len(segment_tasks)} 个片段")
            return segment_tasks
            
        except Exception as e:
            logger.error(f"音频切片失败: {e}")
            return []
    
    def _calculate_segments(self, total_duration: float) -> List[Tuple[float, float]]:
        """计算切片时间段"""
        segments = []
        current_start = 0.0
        
        while current_start < total_duration:
            # 当前片段结束时间
            segment_end = min(current_start + self.segment_length, total_duration)
            
            # 如果剩余时间太短，合并到当前片段
            remaining = total_duration - segment_end
            if remaining > 0 and remaining < 3.0:  # 少于3秒就合并
                segment_end = total_duration
            
            segments.append((current_start, segment_end))
            
            # 下一段开始时间（考虑重叠）
            current_start = segment_end - self.overlap_length
            
            # 如果已经到结尾，退出
            if segment_end >= total_duration:
                break
        
        return segments


# 全局实例
simple_slicer = SimpleSlicer()
