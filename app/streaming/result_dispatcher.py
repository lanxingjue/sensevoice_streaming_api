"""结果分发器"""
import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional
from collections import defaultdict, deque
import time

from ..models.batch_result import BatchResult, SegmentResult
from ..models.segment_task import SegmentStatus
from ..preprocessing.task_manager import task_manager

logger = logging.getLogger(__name__)


class ResultDispatcher:
    """结果分发器"""
    
    def __init__(self):
        # 结果存储
        self.completed_results: Dict[str, SegmentResult] = {}
        self.first_segment_results: Dict[str, SegmentResult] = {}  # 按音频ID存储首片段结果
        
        # 回调函数
        self.first_segment_callbacks: List[Callable] = []
        self.all_segment_callbacks: List[Callable] = []
        
        # 统计信息
        self.dispatch_stats = {
            "total_dispatched": 0,
            "first_segments_dispatched": 0,
            "normal_segments_dispatched": 0,
            "dispatch_errors": 0
        }
        
        # 性能监控
        self.dispatch_times = deque(maxlen=1000)
        
        logger.info("结果分发器初始化完成")
    
    async def dispatch_batch_results(self, batch_result: BatchResult):
        """分发批次结果"""
        dispatch_start = time.time()
        
        try:
            logger.info(f"开始分发批次结果 {batch_result.batch_id}: "
                       f"{len(batch_result.segment_results)} 个结果")
            
            # 分类处理结果
            first_segments = batch_result.get_first_segments()
            normal_segments = batch_result.get_normal_segments()
            
            # 并行分发
            await asyncio.gather(
                self._dispatch_first_segments(first_segments),
                self._dispatch_normal_segments(normal_segments),
                return_exceptions=True
            )
            
            # 更新任务管理器中的结果
            await self._update_task_manager_results(batch_result.segment_results)
            
            # 更新统计
            self.dispatch_stats["total_dispatched"] += len(batch_result.segment_results)
            self.dispatch_stats["first_segments_dispatched"] += len(first_segments)
            self.dispatch_stats["normal_segments_dispatched"] += len(normal_segments)
            
            dispatch_time = time.time() - dispatch_start
            self.dispatch_times.append(dispatch_time)
            
            logger.info(f"批次结果分发完成 {batch_result.batch_id}: "
                       f"首片段={len(first_segments)}, 普通片段={len(normal_segments)}, "
                       f"耗时={dispatch_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"分发批次结果失败 {batch_result.batch_id}: {e}")
            self.dispatch_stats["dispatch_errors"] += 1
    
    async def _dispatch_first_segments(self, first_segments: List[SegmentResult]):
        """分发首片段结果 - 高优先级处理"""
        if not first_segments:
            return
        
        try:
            logger.info(f"分发 {len(first_segments)} 个首片段结果")
            
            for result in first_segments:
                # 存储首片段结果
                self.first_segment_results[result.parent_audio_id] = result
                self.completed_results[result.segment_id] = result
                
                # 调用首片段回调
                await self._call_first_segment_callbacks(result)
                
                logger.debug(f"首片段结果已分发: {result.segment_id} -> '{result.text[:50]}...'")
            
        except Exception as e:
            logger.error(f"分发首片段结果失败: {e}")
    
    async def _dispatch_normal_segments(self, normal_segments: List[SegmentResult]):
        """分发普通片段结果"""
        if not normal_segments:
            return
        
        try:
            logger.debug(f"分发 {len(normal_segments)} 个普通片段结果")
            
            for result in normal_segments:
                # 存储结果
                self.completed_results[result.segment_id] = result
                
                # 调用普通片段回调
                await self._call_all_segment_callbacks(result)
                
                logger.debug(f"普通片段结果已分发: {result.segment_id}")
            
        except Exception as e:
            logger.error(f"分发普通片段结果失败: {e}")
    
    async def _update_task_manager_results(self, segment_results: List[SegmentResult]):
        """更新任务管理器中的结果"""
        try:
            for result in segment_results:
                # 更新任务管理器中的切片结果
                task_manager.update_segment_result(
                    result.segment_id,
                    result.text,
                    result.confidence,
                    result.processing_time
                )
            
        except Exception as e:
            logger.error(f"更新任务管理器结果失败: {e}")
    
    async def _call_first_segment_callbacks(self, result: SegmentResult):
        """调用首片段回调函数"""
        for callback in self.first_segment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"首片段回调执行失败: {e}")
    
    async def _call_all_segment_callbacks(self, result: SegmentResult):
        """调用所有片段回调函数"""
        for callback in self.all_segment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"片段回调执行失败: {e}")
    
    def register_first_segment_callback(self, callback: Callable):
        """注册首片段结果回调"""
        self.first_segment_callbacks.append(callback)
        logger.info(f"注册首片段回调: {callback.__name__}")
    
    def register_all_segment_callback(self, callback: Callable):
        """注册所有片段结果回调"""
        self.all_segment_callbacks.append(callback)
        logger.info(f"注册片段回调: {callback.__name__}")
    
    def get_first_segment_result(self, audio_id: str) -> Optional[SegmentResult]:
        """获取音频的首片段结果"""
        return self.first_segment_results.get(audio_id)
    
    def get_segment_result(self, segment_id: str) -> Optional[SegmentResult]:
        """获取特定切片结果"""
        return self.completed_results.get(segment_id)
    
    def get_audio_segments_results(self, audio_id: str) -> List[SegmentResult]:
        """获取音频的所有已完成切片结果"""
        results = []
        for result in self.completed_results.values():
            if result.parent_audio_id == audio_id:
                results.append(result)
        
        # 按切片索引排序
        results.sort(key=lambda x: x.segment_index)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取分发统计信息"""
        avg_dispatch_time = 0.0
        if len(self.dispatch_times) > 0:
            avg_dispatch_time = sum(self.dispatch_times) / len(self.dispatch_times)
        
        return {
            **self.dispatch_stats,
            "avg_dispatch_time_ms": round(avg_dispatch_time * 1000, 2),
            "pending_first_segments": len(self.first_segment_results),
            "total_completed_segments": len(self.completed_results)
        }
    
    def cleanup_old_results(self, max_age_seconds: float = 3600):
        """清理旧结果"""
        try:
            current_time = time.time()
            cleanup_count = 0
            
            # 清理超时的结果
            expired_ids = []
            for segment_id, result in self.completed_results.items():
                # 假设SegmentResult有创建时间戳，实际可能需要维护单独的时间记录
                if hasattr(result, 'created_at'):
                    if current_time - result.created_at > max_age_seconds:
                        expired_ids.append(segment_id)
            
            for segment_id in expired_ids:
                del self.completed_results[segment_id]
                cleanup_count += 1
            
            # 清理首片段结果中的过期项
            expired_audio_ids = []
            for audio_id, result in self.first_segment_results.items():
                if hasattr(result, 'created_at'):
                    if current_time - result.created_at > max_age_seconds:
                        expired_audio_ids.append(audio_id)
            
            for audio_id in expired_audio_ids:
                del self.first_segment_results[audio_id]
                cleanup_count += 1
            
            if cleanup_count > 0:
                logger.info(f"清理过期结果: {cleanup_count} 项")
            
        except Exception as e:
            logger.error(f"清理过期结果失败: {e}")


# 全局结果分发器实例
result_dispatcher = ResultDispatcher()
