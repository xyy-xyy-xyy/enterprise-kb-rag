# 企业知识库 RAG —— 运行镜像
#
# 镜像里只有 app/ 和运行依赖：没有测试、没有 .env、没有密钥、没有真实文档。
# 密钥与文档一律在运行时注入（见 docker-compose.yml 与 README「Docker 部署」）。
FROM python:3.13-slim

# PYTHONUNBUFFERED：日志实时打到 docker logs，否则卡在缓冲区里看不到
# PYTHONDONTWRITEBYTECODE：容器文件系统不需要 .pyc，少一层写盘
# GRADIO_ANALYTICS_ENABLED：app/main.py 导入时会向 api.gradio.app 查版本号，容器里不需要这次外网请求
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先单独 COPY 依赖清单再安装：只改业务代码时这一层能命中构建缓存，
# 不必把几个 GB 的依赖重装一遍
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# 文档目录（compose 会把宿主机的 ./data/docs 挂到这里）与本地索引目录。
# data/ 本身被 .dockerignore 排除，所以这里显式建出来。
RUN mkdir -p /app/data/docs /app/data/faiss_index

EXPOSE 8000

# 必须写 --host 0.0.0.0：只监听 127.0.0.1 的话容器外访问不到
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
