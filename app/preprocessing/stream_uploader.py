"""流式文件上传器"""
import aiofiles
import asyncio
import time
from pathlib import Path
from typing import AsyncIterator, Tuple
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)


class StreamUploader:
    """流式文件上传处理器"""
    
    def __init__(self, temp_dir: str = "./temp", chunk_size_mb: int = 1):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size_mb * 1024 * 1024  # 转换为字节
    
    async def save_uploaded_file(self, 
                                file: UploadFile, 
                                output_path: Path,
                                max_size_mb: int = 500) -> Tuple[bool, float, str]:
        """
        流式保存上传文件
        返回: (成功标志, 文件大小MB, 错误信息)
        """
        start_time = time.time()
        total_bytes = 0
        max_bytes = max_size_mb * 1024 * 1024
        
        try:
            async with aiofiles.open(output_path, "wb") as out_file:
                # 流式读取和写入
                while True:
                    # 读取一个块
                    chunk = await file.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    total_bytes += len(chunk)
                    
                    # 检查大小限制
                    if total_bytes > max_bytes:
                        await out_file.close()
                        output_path.unlink(missing_ok=True)
                        return False, 0.0, f"文件过大: {total_bytes/1024/1024:.1f}MB > {max_size_mb}MB"
                    
                    # 写入文件
                    await out_file.write(chunk)
                    
                    # 可选：显示进度
                    if total_bytes % (10 * 1024 * 1024) == 0:  # 每10MB
                        progress_mb = total_bytes / 1024 / 1024
                        elapsed = time.time() - start_time
                        speed = progress_mb / elapsed if elapsed > 0 else 0
                        logger.info(f"上传进度: {progress_mb:.1f}MB, 速度: {speed:.1f}MB/s")
            
            file_size_mb = total_bytes / 1024 / 1024
            upload_time = time.time() - start_time
            
            logger.info(f"文件上传完成: {file_size_mb:.2f}MB, 耗时: {upload_time:.2f}秒")
            return True, file_size_mb, ""
            
        except Exception as e:
            # 清理部分文件
            output_path.unlink(missing_ok=True)
            logger.error(f"文件上传失败: {e}")
            return False, 0.0, str(e)
    
    async def validate_audio_file(self, file_path: Path) -> Tuple[bool, float, str]:
        """
        验证音频文件
        返回: (有效标志, 时长秒, 错误信息)
        """
        try:
            import librosa
            
            # 获取音频基本信息，不加载全部数据
            duration = librosa.get_duration(path=str(file_path))
            
            if duration <= 0:
                return False, 0.0, "音频文件无效或损坏"
            
            if duration > 7200:  # 2小时限制
                return False, duration, f"音频过长: {duration/3600:.1f}小时 > 2小时"
            
            logger.info(f"音频验证通过: 时长 {duration:.1f}秒")
            return True, duration, ""
            
        except Exception as e:
            logger.error(f"音频验证失败: {e}")
            return False, 0.0, f"音频格式验证失败: {str(e)}"


# 全局实例
stream_uploader = StreamUploader()
