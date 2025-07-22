sensevoice_streaming_api/
├── requirements.txt
├── config.yaml
├── app/
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── upload.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── request_models.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── sensevoice_service.py
│   └── utils/
│       ├── __init__.py
│       └── audio_utils.py
└── temp/


# 1. 确保环境激活
conda activate sensevoice_stream

# 2. 进入项目目录
cd /data1/wang/wangjianbin/asr/sensevoice_streaming_api

# 3. 检查并创建必要目录
mkdir -p logs temp

# 4. 检查端口占用并清理
sudo pkill -f uvicorn
sleep 2

# 5. 启动服务
python -m app.main

# 或者指定不同端口启动
# uvicorn app.main:app --host 0.0.0.0 --port 8001
