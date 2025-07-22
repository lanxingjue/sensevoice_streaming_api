"""双队列调度器 - 核心调度逻辑"""
import asyncio
import time
import logging
from typing import List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import uuid

from ..models.segment_task import SegmentTask, SegmentStatus
from ..models.batch_result import BatchResult, BatchStatus
from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    """队列统计信息"""
    first_segment_queue_size: int
    normal_segment_queue_size: int
    total_queued: int
    batches_created: int
    batches_completed: int
    avg_batch_size: float
    avg_wait_time: float


class DualQueueScheduler:
    """双队列调度器"""
    
    def __init__(self):
        # 双队列
        self.first_segment_queue = deque()
        self.normal_segment_queue = deque()
        
        # 修正后的配置参数访问方式
        self.batch_size = settings.streaming.batch_size  # 直接访问嵌套属性
        self.batch_timeout = settings.streaming.batch_timeout_ms / 1000.0
        self.max_queue_size = settings.streaming.max_queue_size
        self.max_concurrent_batches = settings.streaming.max_concurrent_batches
        self.queue_check_interval = settings.streaming.queue_check_interval_ms / 1000.0
        
        # 状态管理
        self.is_running = False
        self.batch_counter = 0
        self.total_batches_created = 0
        self.total_batches_completed = 0
        
        # 性能统计
        self.batch_creation_times = deque(maxlen=100)
        self.queue_wait_times = deque(maxlen=1000)
        
        # 锁和事件
        self.queue_lock = asyncio.Lock()
        self.batch_ready_event = asyncio.Event()
        
        logger.info(f"双队列调度器初始化: batch_size={self.batch_size}, timeout={self.batch_timeout}s")
    
    async def add_segment_task(self, segment_task: SegmentTask) -> bool:
        """添加切片任务到队列"""
        async with self.queue_lock:
            # 检查队列容量
            total_queued = len(self.first_segment_queue) + len(self.normal_segment_queue)
            if total_queued >= self.max_queue_size:
                logger.warning(f"队列已满，拒绝新任务: {segment_task.segment_id}")
                return False
            
            # 根据优先级分配到不同队列
            if segment_task.is_first_segment:
                self.first_segment_queue.append(segment_task)
                logger.debug(f"首片段加入队列: {segment_task.segment_id}")
            else:
                self.normal_segment_queue.append(segment_task)
                logger.debug(f"普通片段加入队列: {segment_task.segment_id}")
            
            # 更新任务状态
            segment_task.status = SegmentStatus.QUEUED
            
            # 记录入队时间
            segment_task.queued_at = time.time()
            
            # 通知有新任务
            self.batch_ready_event.set()
            
            return True
    
    async def get_next_batch(self) -> Optional[Tuple[List[SegmentTask], str]]:
        """获取下一个批次"""
        while self.is_running:
            try:
                batch_start_time = time.time()
                
                # 等待任务或超时
                if not await self._wait_for_tasks():
                    continue
                
                # 创建批次
                batch_tasks, batch_id = await self._create_batch()
                
                if batch_tasks:
                    # 记录批次创建时间
                    batch_creation_time = time.time() - batch_start_time
                    self.batch_creation_times.append(batch_creation_time)
                    
                    # 更新统计
                    self.total_batches_created += 1
                    
                    logger.info(f"创建批次 {batch_id}: {len(batch_tasks)} 个任务, "
                              f"耗时: {batch_creation_time*1000:.1f}ms")
                    
                    return batch_tasks, batch_id
                
            except Exception as e:
                logger.error(f"获取批次失败: {e}")
                await asyncio.sleep(0.1)
        
        return None
    
    async def _wait_for_tasks(self) -> bool:
        """等待任务或超时"""
        try:
            # 如果队列为空，等待新任务
            if self._is_queues_empty():
                await asyncio.wait_for(
                    self.batch_ready_event.wait(), 
                    timeout=self.batch_timeout
                )
                self.batch_ready_event.clear()
            
            # 如果队列中任务不足，等待更多任务或超时
            total_tasks = self._get_total_queue_size()
            if total_tasks < self.batch_size:
                try:
                    await asyncio.wait_for(
                        self._wait_for_more_tasks(), 
                        timeout=self.batch_timeout
                    )
                except asyncio.TimeoutError:
                    # 超时也是正常情况，继续处理现有任务
                    pass
            
            return not self._is_queues_empty()
            
        except asyncio.TimeoutError:
            return not self._is_queues_empty()
        except Exception as e:
            logger.error(f"等待任务异常: {e}")
            return False
    
    async def _wait_for_more_tasks(self):
        """等待更多任务"""
        initial_size = self._get_total_queue_size()
        
        while self._get_total_queue_size() < self.batch_size:
            await self.batch_ready_event.wait()
            self.batch_ready_event.clear()
            
            # 如果队列大小没有变化，可能是其他原因触发事件
            current_size = self._get_total_queue_size()
            if current_size == initial_size:
                await asyncio.sleep(0.01)  # 短暂休息
    
    async def _create_batch(self) -> Tuple[List[SegmentTask], str]:
        """创建批次"""
        async with self.queue_lock:
            batch_tasks = []
            batch_id = f"batch_{int(time.time()*1000)}_{self.batch_counter}"
            self.batch_counter += 1
            
            # 1. 优先处理首片段（确保快速响应）
            first_segment_count = 0
            while (len(batch_tasks) < self.batch_size and 
                   len(self.first_segment_queue) > 0):
                task = self.first_segment_queue.popleft()
                batch_tasks.append(task)
                first_segment_count += 1
                
                # 记录等待时间
                if hasattr(task, 'queued_at'):
                    wait_time = time.time() - task.queued_at
                    self.queue_wait_times.append(wait_time)
            
            # 2. 用普通片段填充剩余位置
            normal_segment_count = 0
            while (len(batch_tasks) < self.batch_size and 
                   len(self.normal_segment_queue) > 0):
                task = self.normal_segment_queue.popleft()
                batch_tasks.append(task)
                normal_segment_count += 1
                
                # 记录等待时间
                if hasattr(task, 'queued_at'):
                    wait_time = time.time() - task.queued_at
                    self.queue_wait_times.append(wait_time)
            
            # 更新任务状态
            for task in batch_tasks:
                task.status = SegmentStatus.PROCESSING
                task.batch_id = batch_id
            
            logger.debug(f"批次组成 {batch_id}: "
                        f"首片段={first_segment_count}, "
                        f"普通片段={normal_segment_count}, "
                        f"总计={len(batch_tasks)}")
            
            return batch_tasks, batch_id
    
    def _is_queues_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self.first_segment_queue) == 0 and len(self.normal_segment_queue) == 0
    
    def _get_total_queue_size(self) -> int:
        """获取队列总大小"""
        return len(self.first_segment_queue) + len(self.normal_segment_queue)
    
    async def start(self):
        """启动调度器"""
        if self.is_running:
            logger.warning("调度器已经在运行中")
            return
        
        self.is_running = True
        logger.info("双队列调度器启动")
    
    async def stop(self):
        """停止调度器"""
        self.is_running = False
        self.batch_ready_event.set()  # 唤醒等待的协程
        logger.info("双队列调度器停止")
    
    def get_statistics(self) -> QueueStats:
        """获取队列统计"""
        avg_batch_size = 0.0
        if self.total_batches_created > 0:
            avg_batch_size = self.batch_size  # 简化实现
        
        avg_wait_time = 0.0
        if len(self.queue_wait_times) > 0:
            avg_wait_time = sum(self.queue_wait_times) / len(self.queue_wait_times)
        
        return QueueStats(
            first_segment_queue_size=len(self.first_segment_queue),
            normal_segment_queue_size=len(self.normal_segment_queue),
            total_queued=self._get_total_queue_size(),
            batches_created=self.total_batches_created,
            batches_completed=self.total_batches_completed,
            avg_batch_size=avg_batch_size,
            avg_wait_time=avg_wait_time
        )
    
    def mark_batch_completed(self, batch_id: str):
        """标记批次完成"""
        self.total_batches_completed += 1
        logger.debug(f"批次完成: {batch_id}")


# 全局调度器实例
dual_queue_scheduler = DualQueueScheduler()
