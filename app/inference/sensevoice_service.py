"""SenseVoice推理服务"""
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from funasr import AutoModel
except ImportError:
    logger.warning("FunASR未安装，使用模拟模式")
    AutoModel = None


class SenseVoiceService:
    """SenseVoice推理服务"""
    
    def __init__(self, 
                 model_name: str = "iic/SenseVoiceSmall",
                 device: str = "cuda:0",
                 trust_remote_code: bool = True):
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.model = None
        self._is_initialized = False
    
    def initialize(self) -> bool:
        """初始化模型"""
        try:
            if AutoModel is None:
                logger.warning("使用模拟推理模式 - FunASR未安装")
                self._is_initialized = True
                return True
            
            logger.info(f"正在加载SenseVoice模型: {self.model_name}")
            start_time = time.time()
            
            self.model = AutoModel(
                model=self.model_name,
                trust_remote_code=self.trust_remote_code,
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            
            # 模型预热
            self._warmup_model()
            self._is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            return False
    
    def _warmup_model(self) -> None:
        """模型预热"""
        try:
            if self.model is None:
                return
                
            logger.info("正在进行模型预热...")
            # 创建一个短暂的静音音频进行预热
            import numpy as np
            import soundfile as sf
            
            # 生成1秒的静音音频
            dummy_audio = np.zeros(16000)  # 16kHz, 1秒
            temp_file = "temp_warmup.wav"
            sf.write(temp_file, dummy_audio, 16000)
            
            # 进行一次推理预热
            result = self.model.generate(
                input=temp_file,
                language="auto",
                use_itn=True
            )
            
            # 清理临时文件
            Path(temp_file).unlink(missing_ok=True)
            logger.info("模型预热完成")
            
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """转写单个音频文件"""
        if not self._is_initialized:
            return {
                "success": False,
                "error": "模型未初始化"
            }
        
        try:
            start_time = time.time()
            
            if self.model is None:
                # 模拟推理结果
                processing_time = time.time() - start_time + 0.5  # 模拟处理时间
                return {
                    "success": True,
                    "text": f"这是对音频文件 {Path(audio_path).name} 的模拟转写结果。",
                    "confidence": 0.95,
                    "processing_time": processing_time
                }
            
            # 实际推理
            result = self.model.generate(
                input=audio_path,
                language="auto",  # 自动检测语言
                use_itn=True     # 使用逆文本归一化
            )
            
            processing_time = time.time() - start_time
            
            # 提取结果文本
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            else:
                text = str(result) if result else ""
            
            return {
                "success": True,
                "text": text.strip(),
                "confidence": 0.95,  # SenseVoice通常不返回置信度，使用默认值
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"音频转写失败 {audio_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def is_ready(self) -> bool:
        """检查服务是否就绪"""
        return self._is_initialized
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": self._is_initialized,
            "trust_remote_code": self.trust_remote_code
        }


# 全局服务实例
sensevoice_service = SenseVoiceService()
