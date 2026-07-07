# 基于 Python 3.11（3.13 太新，sentence-transformers 不兼容）
  FROM python:3.11-slim

  # 设置工作目录
  WORKDIR /app

  # 先复制依赖文件，利用 Docker 缓存层
  COPY requirements.txt .

  # 装依赖（用清华镜像，国内快）
  RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

  # 复制项目代码
  COPY . .

  # 告诉外面：这个容器用 8000 端口
  EXPOSE 8000

  # 容器启动时自动跑 FastAPI
  CMD ["python", "src/api/server.py"]