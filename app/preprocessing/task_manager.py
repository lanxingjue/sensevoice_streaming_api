"""简化的任务管理器 - 修复版"""
from typing import Dict, List, Optional
from pathlib import Path
import logging
import time

from ..models.audio_task import AudioTask, TaskStatus
from ..models.segment_task import SegmentTask, SegmentStatus
from .simple_slicer import simple_slicer

logger = logging.getLogger(__name__)


class TaskManager:
    """简化的任务管理器"""
    
    def __init__(self, temp_dir: str = "./temp"):
        self.temp_dir = Path(temp_dir)
        self.audio_tasks: Dict[str, AudioTask] = {}
        self.segment_tasks: Dict[str, SegmentTask] = {}
    
    def create_audio_task(self, task_id: str, filename: str, 
                         file_path: str, file_size_mb: float) -> AudioTask:
        """创建音频任务"""
        task = AudioTask(
            task_id=task_id,
            filename=filename,
            file_path=file_path,
            file_size_mb=file_size_mb,
            status=TaskStatus.UPLOADED
        )
        self.audio_tasks[task_id] = task
        return task
    
    async def process_audio_task(self, task_id: str) -> bool:
        """处理音频任务 - 异步版本"""
        if task_id not in self.audio_tasks:
            return False
        
        task = self.audio_tasks[task_id]
        
        try:
            start_time = time.time()
            task.update_status(TaskStatus.SLICING)
            
            # 验证音频文件
            from .stream_uploader import stream_uploader
            is_valid, duration, error = await stream_uploader.validate_audio_file(
                Path(task.file_path)
            )
            
            if not is_valid:
                task.update_status(TaskStatus.FAILED, error)
                return False
            
            task.duration_seconds = duration
            
            # 创建切片
            segments_dir = self.temp_dir / f"{task_id}_segments"
            segments = simple_slicer.slice_audio(
                task.file_path, 
                task_id, 
                str(segments_dir)
            )
            
            if not segments:
                task.update_status(TaskStatus.FAILED, "切片创建失败")
                return False
            
            # 保存切片任务
            for segment in segments:
                self.segment_tasks[segment.segment_id] = segment
            
            # 更新音频任务状态
            task.segment_count = len(segments)
            task.segments = [seg.segment_id for seg in segments]
            task.processing_time = time.time() - start_time
            task.update_status(TaskStatus.READY)
            
            logger.info(f"任务处理完成: {task_id}, {len(segments)} 个切片")
            return True
            
        except Exception as e:
            logger.error(f"任务处理失败 {task_id}: {e}")
            task.update_status(TaskStatus.FAILED, str(e))
            return False
    
    def get_audio_task(self, task_id: str) -> Optional[AudioTask]:
        """获取音频任务"""
        return self.audio_tasks.get(task_id)
    
    def get_segment_task(self, segment_id: str) -> Optional[SegmentTask]:
        """获取切片任务"""
        return self.segment_tasks.get(segment_id)
    
    # 🔧 修复：添加缺失的方法
    def get_segments_by_audio(self, audio_task_id: str) -> List[SegmentTask]:
        """获取指定音频的所有切片任务"""
        if audio_task_id not in self.audio_tasks:
            return []
        
        audio_task = self.audio_tasks[audio_task_id]
        segments = []
        
        for segment_id in audio_task.segments:
            if segment_id in self.segment_tasks:
                segments.append(self.segment_tasks[segment_id])
        
        # 按切片序号排序
        segments.sort(key=lambda x: x.index)
        return segments
    
    def get_ready_segments(self, limit: int = 128) -> List[SegmentTask]:
        """获取准备处理的切片"""
        ready_segments = [
            seg for seg in self.segment_tasks.values()
            if seg.status == SegmentStatus.READY
        ]
        
        # 按优先级排序（首片段优先）
        ready_segments.sort(key=lambda x: (-x.priority, x.created_at))
        
        return ready_segments[:limit]
    
    def update_segment_result(self, segment_id: str, 
                            text: str, confidence: float, proc_time: float):
        """更新切片结果"""
        if segment_id in self.segment_tasks:
            segment = self.segment_tasks[segment_id]
            segment.set_result(text, confidence, proc_time)
            
            # 检查父任务是否完成
            self._check_audio_task_completion(segment.parent_id)
    
    def _check_audio_task_completion(self, audio_id: str):
        """检查音频任务是否完成"""
        if audio_id not in self.audio_tasks:
            return
        
        audio_task = self.audio_tasks[audio_id]
        segments = [self.segment_tasks[sid] for sid in audio_task.segments 
                   if sid in self.segment_tasks]
        
        completed = sum(1 for seg in segments if seg.status == SegmentStatus.COMPLETED)
        total = len(segments)
        
        if completed == total:
            # 合并所有切片结果
            segments.sort(key=lambda x: x.index)
            full_text = " ".join(seg.text or "" for seg in segments)
            
            audio_task.update_status(TaskStatus.COMPLETED)
            # 可以在这里保存完整结果
            logger.info(f"音频任务完成: {audio_id}")
            
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "audio_tasks": len(self.audio_tasks),
            "segment_tasks": len(self.segment_tasks),
            "ready_segments": len([s for s in self.segment_tasks.values() 
                                 if s.status == SegmentStatus.READY])
        }


# 全局任务管理器
task_manager = TaskManager()
