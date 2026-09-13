"""集中读取 .env 配置，供其他模块导入。"""

import logging
import os

from dotenv import load_dotenv

# 读取项目根目录的 .env（app/ 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBEDDING_MODEL = "text-embedding-v2"  # DashScope 默认嵌入模型

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")

# 向量存储后端：faiss 或 qdrant
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()

# Qdrant 配置
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_kb")

# 检索返回的片段数
TOP_K = int(os.getenv("TOP_K", "5"))

# 混合检索（向量 + BM25，RRF 融合）：设为 false 可退回纯向量检索
HYBRID_RETRIEVAL = os.getenv("HYBRID_RETRIEVAL", "true").lower() == "true"
# 召回候选数 = TOP_K * 该倍数，融合后再截断到 TOP_K
HYBRID_CANDIDATE_MULTIPLIER = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "3"))

# Reranker 重排（走 DashScope API，不下载本地模型）
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "10"))
# 注意：gte-rerank 已于 2026-05-30 下线，官方迁移目标为 qwen3-rerank
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 相对路径统一以项目根目录为基准，避免在 app/ 下运行时找不到文件
def resolve_path(path: str) -> str:
    """把相对路径解析为相对项目根目录的绝对路径。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


INDEX_DIR = resolve_path(FAISS_INDEX_PATH)
DOCS_PATH = resolve_path(DOCS_DIR)


def setup_logging() -> None:
    """配置全局日志（幂等，重复调用不会叠加 handler）。"""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(LOG_LEVEL)
        return
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate() -> None:
    """启动前检查关键配置，缺失时给出清晰提示。"""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未找到 DASHSCOPE_API_KEY，请在项目根目录的 .env 中配置后重试。"
        )
