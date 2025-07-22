"""批处理器 - 协调调度器和推理引擎"""
import asyncio
import logging
from typing import Optional
import time

from .dual_queue_scheduler import dual_queue_scheduler
from ..inference.batch_inference_engine import batch_inference_engine
from .result_dispatcher import result_dispatcher
from ..config.settings import settings

logger = logging.getLogger(__name__)


class BatchProcessor:
    """批处理器 - 流式处理的核心协调器"""
    
    def __init__(self):
        self.scheduler = dual_queue_scheduler
        self.inference_engine = batch_inference_engine
        self.dispatcher = result_dispatcher
        
        # 配置参数
        self.max_concurrent_batches = settings.streaming_max_concurrent_batches
        
        # 状态管理
        self.is_running = False
        self.processing_tasks = []
        self.total_processed_batches = 0
        
        # 性能监控
        self.start_time = None
        self.last_batch_time = None
        
        logger.info(f"批处理器初始化: 最大并发批次={self.max_concurrent_batches}")
    
    async def start(self):
        """启动批处理器"""
        if self.is_running:
            logger.warning("批处理器已经在运行")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        logger.info("启动批处理器...")
        
        # 启动调度器
        await self.scheduler.start()
        
        # 启动处理循环
        for i in range(self.max_concurrent_batches):
            task = asyncio.create_task(self._processing_loop(f"worker_{i}"))
            self.processing_tasks.append(task)
        
        logger.info(f"批处理器启动完成，{len(self.processing_tasks)} 个工作线程")
    
    async def stop(self):
        """停止批处理器"""
        if not self.is_running:
            return
        
        logger.info("正在停止批处理器...")
        
        self.is_running = False
        
        # 停止调度器
        await self.scheduler.stop()
        
        # 等待所有处理任务完成
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            self.processing_tasks.clear()
        
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"批处理器已停止，总运行时间: {total_time:.1f}秒，"
                   f"处理批次: {self.total_processed_batches}")
    
    async def _processing_loop(self, worker_name: str):
        """处理循环 - 工作线程主逻辑"""
        logger.info(f"工作线程启动: {worker_name}")
        
        while self.is_running:
            try:
                # 获取下一个批次
                batch_data = await self.scheduler.get_next_batch()
                
                if batch_data is None:
                    # 没有批次可处理，短暂休息
                    await asyncio.sleep(0.1)
                    continue
                
                batch_tasks, batch_id = batch_data
                
                if not batch_tasks:
                    continue
                
                # 处理批次
                await self._process_single_batch(batch_tasks, batch_id, worker_name)
                
                # 更新统计
                self.total_processed_batches += 1
                self.last_batch_time = time.time()
                
                # 标记批次完成
                self.scheduler.mark_batch_completed(batch_id)
                
            except Exception as e:
                logger.error(f"工作线程 {worker_name} 处理异常: {e}")
                await asyncio.sleep(1)  # 异常后休息1秒
        
        logger.info(f"工作线程退出: {worker_name}")
    
    async def _process_single_batch(self, batch_tasks, batch_id: str, worker_name: str):
        """处理单个批次"""
        batch_start = time.time()
        
        logger.info(f"{worker_name} 开始处理批次 {batch_id}: {len(batch_tasks)} 个任务")
        
        try:
            # 执行批量推理
            batch_result = await self.inference_engine.process_batch(batch_tasks, batch_id)
            
            # 分发结果
            await self.dispatcher.dispatch_batch_results(batch_result)
            
            batch_time = time.time() - batch_start
            throughput = len(batch_tasks) / batch_time
            
            logger.info(f"{worker_name} 批次处理完成 {batch_id}: "
                       f"耗时={batch_time*1000:.1f}ms, "
                       f"吞吐量={throughput:.1f} segments/s, "
                       f"成功率={batch_result.success_rate*100:.1f}%")
            
        except Exception as e:
            logger.error(f"{worker_name} 批次处理失败 {batch_id}: {e}")
    
    async def add_segment_task(self, segment_task) -> bool:
        """添加切片任务到处理队列"""
        if not self.is_running:
            logger.warning("批处理器未运行，无法添加任务")
            return False
        
        return await self.scheduler.add_segment_task(segment_task)
    
    def get_status(self) -> dict:
        """获取批处理器状态"""
        queue_stats = self.scheduler.get_statistics()
        inference_stats = self.inference_engine.get_statistics()
        dispatch_stats = self.dispatcher.get_statistics()
        gpu_status = self.inference_engine.get_gpu_status()
        
        uptime = time.time() - self.start_time if self.start_time else 0
        avg_batches_per_minute = 0
        if uptime > 0:
            avg_batches_per_minute = (self.total_processed_batches / uptime) * 60
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": round(uptime, 1),
            "total_processed_batches": self.total_processed_batches,
            "avg_batches_per_minute": round(avg_batches_per_minute, 1),
            "active_workers": len([t for t in self.processing_tasks if not t.done()]),
            "queue_stats": queue_stats.__dict__,
            "inference_stats": inference_stats,
            "dispatch_stats": dispatch_stats,
            "gpu_status": gpu_status,
            "last_batch_time": self.last_batch_time
        }
    
    def get_performance_metrics(self) -> dict:
        """获取性能指标"""
        return {
            "scheduler_metrics": {
                "avg_batch_creation_time": getattr(self.scheduler, 'avg_batch_creation_time', 0),
                "avg_queue_wait_time": getattr(self.scheduler, 'avg_queue_wait_time', 0)
            },
            "inference_metrics": self.inference_engine.get_statistics(),
            "dispatch_metrics": self.dispatcher.get_statistics()
        }


# 全局批处理器实例
batch_processor = BatchProcessor()
