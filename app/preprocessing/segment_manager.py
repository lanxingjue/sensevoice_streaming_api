"""切片管理器"""
from typing import Dict, List, Optional
from pathlib import Path
import logging
import time

from ..models.audio_task import AudioTask, AudioTaskStatus
from ..models.segment_task import SegmentTask, SegmentStatus
from .intelligent_slicer import IntelligentAudioSlicer
from .audio_converter import EnhancedAudioConverter

logger = logging.getLogger(__name__)


class SegmentManager:
    """切片任务管理器"""
    
    def __init__(self, 
                 temp_dir: str = "./temp",
                 segment_length: float = 10.0,
                 overlap_duration: float = 2.0):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.audio_converter = EnhancedAudioConverter()
        self.audio_slicer = IntelligentAudioSlicer(
            segment_length=segment_length,
            overlap_duration=overlap_duration
        )
        
        # 存储管理
        self.audio_tasks: Dict[str, AudioTask] = {}
        self.segment_tasks: Dict[str, SegmentTask] = {}
        
        # 统计信息
        self.total_segments_created = 0
        self.total_segments_processed = 0
    
    def process_audio_file(self, audio_task: AudioTask) -> bool:
        """处理音频文件，生成切片任务"""
        try:
            audio_task.update_status(AudioTaskStatus.PREPROCESSING, "开始音频预处理")
            logger.info(f"开始处理音频任务: {audio_task.task_id}")
            
            # 1. 音频格式转换和增强
            processed_file_path = self._preprocess_audio(audio_task)
            if not processed_file_path:
                audio_task.set_error("音频预处理失败")
                return False
            
            audio_task.processed_file_path = processed_file_path
            audio_task.update_status(AudioTaskStatus.PREPROCESSING, "音频预处理完成，开始切片")
            
            # 2. 智能切片
            segment_tasks = self._create_segments(audio_task, processed_file_path)
            if not segment_tasks:
                audio_task.set_error("音频切片失败")
                return False
            
            # 3. 更新任务状态
            audio_task.segment_count = len(segment_tasks)
            audio_task.segment_tasks = [task.task_id for task in segment_tasks]
            audio_task.update_status(AudioTaskStatus.SEGMENTED, f"切片完成，共{len(segment_tasks)}个片段")
            
            # 4. 存储切片任务
            for segment_task in segment_tasks:
                self.segment_tasks[segment_task.task_id] = segment_task
            
            self.audio_tasks[audio_task.task_id] = audio_task
            self.total_segments_created += len(segment_tasks)
            
            logger.info(f"音频处理完成: {audio_task.task_id}, 生成 {len(segment_tasks)} 个切片")
            return True
            
        except Exception as e:
            logger.error(f"处理音频文件失败 {audio_task.task_id}: {e}")
            audio_task.set_error(f"处理失败: {str(e)}")
            return False
    
    def _preprocess_audio(self, audio_task: AudioTask) -> Optional[str]:
        """预处理音频文件"""
        try:
            # 分析原始音频
            analysis = self.audio_converter.analyze_audio(audio_task.original_file_path)
            if not analysis:
                logger.error(f"音频分析失败: {audio_task.original_file_path}")
                return None
            
            # 更新音频元数据
            metadata = audio_task.metadata
            metadata.original_sample_rate = analysis["sample_rate"]
            metadata.original_channels = analysis["channels"]
            metadata.duration_seconds = analysis["duration"]
            
            # 生成处理后文件路径
            processed_file_path = str(self.temp_dir / f"{audio_task.task_id}_processed.wav")
            
            # 执行增强转换
            success = self.audio_converter.convert_with_enhancement(
                audio_task.original_file_path,
                processed_file_path
            )
            
            if success:
                logger.info(f"音频预处理完成: {processed_file_path}")
                return processed_file_path
            else:
                logger.error(f"音频转换失败: {audio_task.original_file_path}")
                return None
                
        except Exception as e:
            logger.error(f"音频预处理异常 {audio_task.task_id}: {e}")
            return None
    
    def _create_segments(self, audio_task: AudioTask, processed_file_path: str) -> List[SegmentTask]:
        """创建音频切片"""
        try:
            # 创建切片输出目录
            segments_dir = self.temp_dir / f"{audio_task.task_id}_segments"
            segments_dir.mkdir(exist_ok=True)
            
            # 执行智能切片
            segment_tasks = self.audio_slicer.slice_audio(
                audio_file_path=processed_file_path,
                parent_audio_id=audio_task.task_id,
                output_dir=str(segments_dir)
            )
            
            # 分析每个切片的质量
            for segment_task in segment_tasks:
                self.audio_slicer.analyze_segment_quality(segment_task)
                segment_task.update_status(SegmentStatus.CREATED)
            
            return segment_tasks
            
        except Exception as e:
            logger.error(f"创建音频切片失败 {audio_task.task_id}: {e}")
            return []
    
    def get_audio_task(self, task_id: str) -> Optional[AudioTask]:
        """获取音频任务"""
        return self.audio_tasks.get(task_id)
    
    def get_segment_task(self, segment_id: str) -> Optional[SegmentTask]:
        """获取切片任务"""
        return self.segment_tasks.get(segment_id)
    
    def get_segments_by_audio(self, audio_task_id: str) -> List[SegmentTask]:
        """获取指定音频的所有切片任务"""
        if audio_task_id not in self.audio_tasks:
            return []
        
        audio_task = self.audio_tasks[audio_task_id]
        segments = []
        
        for segment_id in audio_task.segment_tasks:
            if segment_id in self.segment_tasks:
                segments.append(self.segment_tasks[segment_id])
        
        # 按切片序号排序
        segments.sort(key=lambda x: x.segment_index)
        return segments
    
    def get_ready_segments(self, limit: Optional[int] = None) -> List[SegmentTask]:
        """获取准备处理的切片任务"""
        ready_segments = [
            segment for segment in self.segment_tasks.values()
            if segment.status == SegmentStatus.CREATED
        ]
        
        # 按优先级排序（首片段优先）
        ready_segments.sort(key=lambda x: (-x.priority, x.created_at))
        
        if limit:
            return ready_segments[:limit]
        return ready_segments
    
    def update_segment_status(self, segment_id: str, status: SegmentStatus):
        """更新切片状态"""
        if segment_id in self.segment_tasks:
            self.segment_tasks[segment_id].update_status(status)
    
    def set_segment_result(self, 
                          segment_id: str, 
                          text: str, 
                          confidence: float, 
                          processing_time: float):
        """设置切片转写结果"""
        if segment_id in self.segment_tasks:
            segment_task = self.segment_tasks[segment_id]
            segment_task.set_transcription_result(text, confidence, processing_time)
            self.total_segments_processed += 1
            
            # 更新父任务进度
            self._update_audio_progress(segment_task.parent_audio_id)
    
    def _update_audio_progress(self, audio_task_id: str):
        """更新音频任务进度"""
        if audio_task_id not in self.audio_tasks:
            return
        
        audio_task = self.audio_tasks[audio_task_id]
        segments = self.get_segments_by_audio(audio_task_id)
        
        if not segments:
            return
        
        # 计算完成进度
        completed_segments = sum(
            1 for seg in segments 
            if seg.status == SegmentStatus.COMPLETED
        )
        
        progress_percent = (completed_segments / len(segments)) * 100
        audio_task.progress_percent = progress_percent
        
        # 更新状态
        if completed_segments == 0:
            audio_task.update_status(AudioTaskStatus.PROCESSING, "开始处理切片")
        elif completed_segments == len(segments):
            audio_task.update_status(AudioTaskStatus.COMPLETED, "所有切片处理完成")
        else:
            audio_task.update_status(
                AudioTaskStatus.PROCESSING, 
                f"处理中 {completed_segments}/{len(segments)} 切片"
            )
    
    def get_statistics(self) -> Dict:
        """获取管理器统计信息"""
        audio_stats = {
            status.value: sum(1 for task in self.audio_tasks.values() 
                            if task.status == status)
            for status in AudioTaskStatus
        }
        
        segment_stats = {
            status.value: sum(1 for task in self.segment_tasks.values() 
                            if task.status == status)
            for status in SegmentStatus
        }
        
        return {
            "audio_tasks": {
                "total": len(self.audio_tasks),
                "by_status": audio_stats
            },
            "segment_tasks": {
                "total": len(self.segment_tasks),
                "by_status": segment_stats,
                "total_created": self.total_segments_created,
                "total_processed": self.total_segments_processed
            },
            "storage": {
                "temp_dir": str(self.temp_dir),
                "temp_dir_size_mb": self._calculate_temp_dir_size()
            }
        }
    
    def _calculate_temp_dir_size(self) -> float:
        """计算临时目录大小"""
        try:
            total_size = 0
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # 转换为MB
        except Exception as e:
            logger.warning(f"计算临时目录大小失败: {e}")
            return 0.0
    
    def cleanup_completed_tasks(self, max_age_hours: float = 24.0):
        """清理完成的任务文件"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # 找出过期的已完成任务
            expired_tasks = []
            for audio_task in self.audio_tasks.values():
                if (audio_task.status == AudioTaskStatus.COMPLETED and
                    current_time - audio_task.updated_at > max_age_seconds):
                    expired_tasks.append(audio_task)
            
            # 清理过期任务
            for audio_task in expired_tasks:
                self._cleanup_audio_task_files(audio_task)
                
                # 移除任务记录
                del self.audio_tasks[audio_task.task_id]
                for segment_id in audio_task.segment_tasks:
                    if segment_id in self.segment_tasks:
                        del self.segment_tasks[segment_id]
            
            if expired_tasks:
                logger.info(f"清理了 {len(expired_tasks)} 个过期任务")
                
        except Exception as e:
            logger.error(f"清理过期任务失败: {e}")
    
    def _cleanup_audio_task_files(self, audio_task: AudioTask):
        """清理音频任务相关文件"""
        try:
            # 清理原始文件
            if audio_task.original_file_path:
                Path(audio_task.original_file_path).unlink(missing_ok=True)
            
            # 清理处理后文件
            if audio_task.processed_file_path:
                Path(audio_task.processed_file_path).unlink(missing_ok=True)
            
            # 清理切片文件目录
            segments_dir = self.temp_dir / f"{audio_task.task_id}_segments"
            if segments_dir.exists():
                for file_path in segments_dir.iterdir():
                    file_path.unlink(missing_ok=True)
                segments_dir.rmdir()
                
        except Exception as e:
            logger.warning(f"清理任务文件失败 {audio_task.task_id}: {e}")


# 全局管理器实例
segment_manager = SegmentManager()
