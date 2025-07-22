#!/usr/bin/env python3
"""配置加载测试脚本"""

from app.config.settings import settings

def test_config_loading():
    """测试配置是否正确加载"""
    
    print("=== 配置加载验证 ===")
    
    # 测试基本配置
    print(f"服务器地址: {settings.server_host}:{settings.server_port}")
    print(f"模型设备: {settings.model_device}")
    
    # 测试嵌套配置访问
    print(f"\n=== 流式处理配置 ===")
    print(f"批处理大小: {settings.streaming.batch_size}")
    print(f"批处理超时: {settings.streaming.batch_timeout_ms}ms")
    print(f"最大队列大小: {settings.streaming.max_queue_size}")
    print(f"最大并发批次: {settings.streaming.max_concurrent_batches}")
    
    # 测试音频配置
    print(f"\n=== 音频处理配置 ===")
    print(f"目标采样率: {settings.audio_preprocessing.target_sample_rate}Hz")
    print(f"目标声道: {settings.audio_preprocessing.target_channels}")
    print(f"切片长度: {settings.audio_segmentation.segment_length_seconds}秒")
    print(f"重叠长度: {settings.audio_segmentation.overlap_seconds}秒")
    
    # 测试向后兼容性
    print(f"\n=== 向后兼容性测试 ===")
    print(f"streaming_batch_size: {settings.streaming_batch_size}")
    print(f"streaming_batch_timeout_ms: {settings.streaming_batch_timeout_ms}")
    
    print("\n✅ 配置加载测试完成")

if __name__ == "__main__":
    test_config_loading()
