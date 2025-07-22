"""批量推理引擎"""
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from ..models.segment_task import SegmentTask
from ..models.batch_result import BatchResult, SegmentResult, BatchStatus
from .sensevoice_service import sensevoice_service

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """批量推理引擎"""
    
    def __init__(self):
        self.model_service = sensevoice_service
        self.inference_stats = {
            "total_batches": 0,
            "total_segments": 0,
            "total_inference_time": 0.0,
            "avg_batch_time": 0.0,
            "success_rate": 0.0
        }
    
    async def process_batch(self, 
                          batch_tasks: List[SegmentTask], 
                          batch_id: str) -> BatchResult:
        """处理批次任务"""
        batch_start_time = time.time()
        
        # 创建批次结果对象
        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PROCESSING,
            batch_size=len(batch_tasks),
            first_segments_count=sum(1 for task in batch_tasks if task.is_first_segment),
            normal_segments_count=sum(1 for task in batch_tasks if not task.is_first_segment),
            processing_start=batch_start_time
        )
        
        try:
            logger.info(f"开始批量推理 {batch_id}: {len(batch_tasks)} 个切片")
            
            # 准备音频文件路径列表
            audio_paths = [task.file_path for task in batch_tasks]
            
            # 执行批量推理
            inference_start = time.time()
            batch_results = await self._batch_transcribe(audio_paths)
            inference_time = time.time() - inference_start
            
            batch_result.gpu_inference_time = inference_time
            
            # 处理推理结果
            if batch_results and len(batch_results) == len(batch_tasks):
                segment_results = self._process_inference_results(
                    batch_tasks, batch_results, inference_time
                )
                batch_result.segment_results = segment_results
                batch_result.status = BatchStatus.COMPLETED
                
                logger.info(f"批量推理完成 {batch_id}: "
                          f"成功 {len(segment_results)}/{len(batch_tasks)}, "
                          f"GPU推理耗时: {inference_time*1000:.1f}ms")
            else:
                # 推理失败
                batch_result.status = BatchStatus.FAILED
                batch_result.error_message = "批量推理返回结果不匹配"
                batch_result.failed_segments = [task.segment_id for task in batch_tasks]
                
                logger.error(f"批量推理失败 {batch_id}: 结果数量不匹配")
        
        except Exception as e:
            # 处理异常
            batch_result.status = BatchStatus.FAILED
            batch_result.error_message = str(e)
            batch_result.failed_segments = [task.segment_id for task in batch_tasks]
            
            logger.error(f"批量推理异常 {batch_id}: {e}")
        
        finally:
            # 完成时间统计
            batch_result.processing_end = time.time()
            batch_result.total_processing_time = batch_result.processing_end - batch_start_time
            
            # 更新统计信息
            self._update_statistics(batch_result)
        
        return batch_result
    
    async def _batch_transcribe(self, audio_paths: List[str]) -> Optional[List[Dict[str, Any]]]:
        """执行批量转写"""
        try:
            # 检查模型状态
            if not self.model_service.is_ready():
                logger.error("SenseVoice模型未就绪")
                return None
            
            # 验证音频文件存在
            valid_paths = []
            for path in audio_paths:
                if Path(path).exists():
                    valid_paths.append(path)
                else:
                    logger.warning(f"音频文件不存在: {path}")
            
            if not valid_paths:
                logger.error("没有有效的音频文件")
                return None
            
            # 执行批量推理
            if hasattr(self.model_service, 'batch_transcribe'):
                # 如果有专门的批量接口
                results = await asyncio.to_thread(
                    self.model_service.batch_transcribe, valid_paths
                )
            else:
                # 使用循环调用单个推理接口
                results = []
                for path in valid_paths:
                    result = await asyncio.to_thread(
                        self.model_service.transcribe, path
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"批量转写执行失败: {e}")
            return None
    
    def _process_inference_results(self, 
                                 batch_tasks: List[SegmentTask],
                                 inference_results: List[Dict[str, Any]],
                                 total_inference_time: float) -> List[SegmentResult]:
        """处理推理结果"""
        segment_results = []
        avg_processing_time = total_inference_time / len(batch_tasks)
        
        for i, (task, result) in enumerate(zip(batch_tasks, inference_results)):
            try:
                if result.get("success", False):
                    # 创建成功的切片结果
                    segment_result = SegmentResult(
                        segment_id=task.segment_id,
                        parent_audio_id=task.parent_id,
                        segment_index=task.index,
                        text=result.get("text", "").strip(),
                        confidence=result.get("confidence", 0.95),
                        processing_time=avg_processing_time,
                        start_time=task.start_time,
                        end_time=task.end_time,
                        duration=task.duration,
                        is_first_segment=task.is_first_segment,
                        priority=task.priority,
                        file_path=task.file_path
                    )
                    
                    segment_results.append(segment_result)
                    
                    logger.debug(f"切片转写成功 {task.segment_id}: '{segment_result.text[:50]}...'")
                else:
                    logger.warning(f"切片转写失败 {task.segment_id}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"处理切片结果失败 {task.segment_id}: {e}")
        
        return segment_results
    
    def _update_statistics(self, batch_result: BatchResult):
        """更新统计信息"""
        self.inference_stats["total_batches"] += 1
        self.inference_stats["total_segments"] += batch_result.batch_size
        self.inference_stats["total_inference_time"] += batch_result.gpu_inference_time
        
        # 计算平均值
        if self.inference_stats["total_batches"] > 0:
            self.inference_stats["avg_batch_time"] = (
                self.inference_stats["total_inference_time"] / 
                self.inference_stats["total_batches"]
            )
        
        # 计算成功率
        if batch_result.status == BatchStatus.COMPLETED:
            success_segments = len(batch_result.segment_results)
        else:
            success_segments = 0
        
        # 更新全局成功率（简化计算）
        if self.inference_stats["total_segments"] > 0:
            # 这里应该维护一个更精确的成功计数器
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        return self.inference_stats.copy()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """获取GPU状态信息"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
                
                return {
                    "device": f"cuda:{device}",
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_total_gb": round(memory_total, 2),
                    "memory_utilization": round(memory_allocated / memory_total * 100, 1)
                }
            else:
                return {"error": "CUDA不可用"}
        except Exception as e:
            return {"error": f"获取GPU状态失败: {e}"}


# 全局推理引擎实例
batch_inference_engine = BatchInferenceEngine()
