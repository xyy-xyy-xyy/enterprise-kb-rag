"""向量索引的创建 / 保存 / 加载 / 检索，支持 FAISS 和 Qdrant 双后端。

通过 .env 中 VECTOR_BACKEND=faiss 或 VECTOR_BACKEND=qdrant 切换。
"""

import logging
import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from app import config

logger = logging.getLogger(__name__)

# FAISS 文件名
_INDEX_FILE = "index.faiss"
_PKL_FILE = "index.pkl"


class IndexNotReadyError(RuntimeError):
    """索引尚未构建时抛出，调用方据此返回友好提示。"""


_embeddings = None


def get_embeddings() -> DashScopeEmbeddings:
    """返回 DashScope 嵌入模型（进程内复用同一个实例）。"""
    global _embeddings
    if _embeddings is None:
        config.validate()
        _embeddings = DashScopeEmbeddings(
            model=config.EMBEDDING_MODEL,
            dashscope_api_key=config.DASHSCOPE_API_KEY,
        )
        logger.debug("已初始化嵌入模型 %s", config.EMBEDDING_MODEL)
    return _embeddings


# ============================ 后端抽象层 ============================


def index_exists() -> bool:
    """判断索引是否已就绪。"""
    if config.VECTOR_BACKEND == "qdrant":
        return _qdrant_collection_exists()
    return os.path.exists(os.path.join(config.INDEX_DIR, _INDEX_FILE)) and os.path.exists(
        os.path.join(config.INDEX_DIR, _PKL_FILE)
    )


def create_index(documents: list[Document]):
    """用文档分块新建索引，返回 vectorstore。"""
    if not documents:
        raise ValueError("文档分块为空，无法建立索引。")
    logger.info("正在向量化 %d 个分块并建立新索引...", len(documents))
    if config.VECTOR_BACKEND == "qdrant":
        return _qdrant_create(documents)
    return _faiss_create(documents)


def add_documents(vectorstore, documents: list[Document]):
    """把新分块追加到已有索引。"""
    if not documents:
        return vectorstore
    logger.info("正在向量化 %d 个分块并追加到已有索引...", len(documents))
    if config.VECTOR_BACKEND == "qdrant":
        vectorstore.add_documents(documents)
        return vectorstore
    vectorstore.add_documents(documents)
    logger.info("追加完成，索引现有 %d 条向量。", vectorstore.index.ntotal)
    return vectorstore


def save_index(vectorstore) -> str:
    """持久化索引。FAISS 写磁盘，Qdrant 已实时写入无需额外操作。"""
    if config.VECTOR_BACKEND == "qdrant":
        logger.info("Qdrant 索引已实时持久化，collection=%s", config.QDRANT_COLLECTION)
        return config.QDRANT_COLLECTION
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.INDEX_DIR)
    logger.info("索引已保存到 %s", config.INDEX_DIR)
    return config.INDEX_DIR


def load_index():
    """加载索引；不存在时抛出 IndexNotReadyError。"""
    if config.VECTOR_BACKEND == "qdrant":
        return _qdrant_load()
    return _faiss_load()


def get_index():
    """加载索引的语义化封装。"""
    return load_index()


def search(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """相似度检索，返回 [(Document, score)]。

    FAISS: score 越小越相关（L2 距离）
    Qdrant: score 越大越相关（余弦相似度）
    统一返回时对 Qdrant 不做翻转，调用方按后端类型自行判断。
    """
    if not query or not query.strip():
        raise ValueError("检索内容不能为空。")
    vectorstore = load_index()
    results = vectorstore.similarity_search_with_score(query, k=k)
    logger.info("检索 %r 命中 %d 个片段。", query[:30], len(results))
    return results


def get_indexed_sources(vectorstore) -> set[str]:
    """读取索引中已存在的文档来源，用于避免重复入库。"""
    if config.VECTOR_BACKEND == "qdrant":
        return _qdrant_get_sources(vectorstore)
    try:
        docs = vectorstore.docstore._dict.values()
    except AttributeError:
        logger.debug("无法读取索引中的文档来源，跳过去重检查。")
        return set()
    return {d.metadata.get("source") for d in docs if d.metadata.get("source")}


def get_all_documents() -> list[Document]:
    """取出索引中的全量分块，供本地构建 BM25 索引使用。

    仅索引元数据与正文，不重算向量；演示规模够用。
    """
    if config.VECTOR_BACKEND == "qdrant":
        return _qdrant_get_all_documents()
    return _faiss_get_all_documents()


def get_indexed_hashes(vectorstore=None) -> dict[str, str]:
    """读取索引中的 {sm3_hash: file_name}，用于按内容去重。

    注意 `vectorstore` 只对 FAISS 后端生效：FAISS 的分块都在已加载的 docstore 里，
    传进来可直接复用。Qdrant 的分块存在服务端，无论传什么都得再 scroll 一次
    才能拿到 payload，因此该参数在 Qdrant 下会被忽略（属预期，非遗漏）。

    旧索引里的分块没有 sm3_hash 字段，遇到必须跳过而不是 KeyError。
    """
    if config.VECTOR_BACKEND == "qdrant":
        return _qdrant_get_hashes()
    return _faiss_get_hashes(vectorstore)


def _collect_hashes(metadatas) -> dict[str, str]:
    """把 metadata 列表收成 {sm3_hash: file_name}，缺字段的条目直接跳过。

    同一个摘要可能对应多个分块（同一文档的所有分块共享一个 SM3），
    setdefault 保留最先遇到的文件名即可。
    """
    hashes: dict[str, str] = {}
    for meta in metadatas:
        digest = meta.get("sm3_hash")
        if digest:
            hashes.setdefault(digest, meta.get("file_name"))
    return hashes


def reset_index() -> None:
    """删除索引（用于重建）。"""
    if config.VECTOR_BACKEND == "qdrant":
        _qdrant_reset()
        return
    for name in (_INDEX_FILE, _PKL_FILE):
        path = os.path.join(config.INDEX_DIR, name)
        if os.path.exists(path):
            os.remove(path)
            logger.info("已删除旧索引文件：%s", path)


# ============================ FAISS 后端 ============================


def _faiss_create(documents: list[Document]):
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(documents, get_embeddings())
    logger.info("FAISS 索引建立完成，共 %d 条向量。", vectorstore.index.ntotal)
    return vectorstore


def _faiss_load():
    from langchain_community.vectorstores import FAISS

    if not index_exists():
        raise IndexNotReadyError(
            f"未找到索引文件（{config.INDEX_DIR}），请先运行 "
            f"`python -m app.ingestion` 入库。"
        )
    vectorstore = FAISS.load_local(
        config.INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    logger.debug("已加载 FAISS 索引：%d 条向量。", vectorstore.index.ntotal)
    return vectorstore


def _faiss_get_all_documents() -> list[Document]:
    """FAISS 的分块都放在 docstore 里，直接取出来即可。"""
    vectorstore = load_index()
    try:
        return list(vectorstore.docstore._dict.values())
    except AttributeError:
        # 不同 langchain 版本内部结构可能变化，降级为“BM25 无候选”而非报错
        logger.warning("无法读取 FAISS 索引中的文档，BM25 召回将跳过。")
        return []


def _faiss_get_hashes(vectorstore=None) -> dict[str, str]:
    """从 FAISS docstore 里收集内容摘要。"""
    try:
        docs = (vectorstore or load_index()).docstore._dict.values()
    except AttributeError:
        logger.debug("无法读取 FAISS 索引中的文档摘要，跳过去重检查。")
        return {}
    return _collect_hashes(
        (doc.metadata or {}) for doc in docs
    )


# ============================ Qdrant 后端 ============================


def _get_qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def _qdrant_collection_exists() -> bool:
    try:
        client = _get_qdrant_client()
        collections = client.get_collections()
        return any(c.name == config.QDRANT_COLLECTION for c in collections.collections)
    except Exception as exc:
        logger.warning("连接 Qdrant 失败：%s", exc)
        return False


def _qdrant_vector_store_class():
    """延迟导入 langchain_qdrant —— 只有真的用 Qdrant 后端时才需要这个包。

    为什么不做顶层导入：FAISS 后端（默认）完全不需要它，顶层导入会让
    `import app.vector_store` 在只装了 `qdrant-client` 的环境里直接失败。

    为什么把 ImportError 换成 RuntimeError：裸的 ModuleNotFoundError 指向的是
    "某个模块没了"，而不是"环境没装齐"，排查时容易跑偏（阶段五 CI 就踩过一次：
    requirements.txt 漏声明这个包 → 用例报出的却是"索引未就绪"，根因被藏住）。
    这里刻意**不吞掉**这个错误，只是把提示换成能直接照做的动作。
    """
    try:
        from langchain_qdrant import QdrantVectorStore
    except ImportError as exc:
        raise RuntimeError(
            "缺少 langchain-qdrant 依赖，无法使用 Qdrant 后端；"
            "请执行 `pip install -r requirements.txt` 后重试。"
        ) from exc
    return QdrantVectorStore


def _qdrant_create(documents: list[Document]):
    QdrantVectorStore = _qdrant_vector_store_class()
    from qdrant_client.models import Distance, VectorParams

    client = _get_qdrant_client()
    embedding_dim = len(get_embeddings().embed_query("test"))
    client.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=get_embeddings(),
    )
    vectorstore.add_documents(documents)
    count = client.count(collection_name=config.QDRANT_COLLECTION).count
    logger.info("Qdrant 索引建立完成，collection=%s，共 %d 条向量。",
                config.QDRANT_COLLECTION, count)
    return vectorstore


def _qdrant_load():
    QdrantVectorStore = _qdrant_vector_store_class()

    if not _qdrant_collection_exists():
        raise IndexNotReadyError(
            f"Qdrant collection '{config.QDRANT_COLLECTION}' 不存在，"
            f"请先运行 `python -m app.ingestion` 入库。"
        )
    client = _get_qdrant_client()
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=get_embeddings(),
    )
    count = client.count(collection_name=config.QDRANT_COLLECTION).count
    logger.debug("已加载 Qdrant 索引：collection=%s，%d 条向量。",
                 config.QDRANT_COLLECTION, count)
    return vectorstore


def _qdrant_get_sources(vectorstore) -> set[str]:
    try:
        client = _get_qdrant_client()
        # 必须取 payload，否则 point.payload 为 None，去重会永远失效；
        # 只取 metadata.source 一个字段，避免把全部正文都拉回来
        points, _ = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=10000,
            with_payload=["metadata.source"],
            with_vectors=False,
        )
        sources = set()
        for point in points:
            # LangChain 写入的 payload 为 {page_content, metadata}，source 在 metadata 里
            meta = (point.payload or {}).get("metadata") or {}
            src = meta.get("source")
            if src:
                sources.add(src)
        return sources
    except Exception as exc:
        logger.debug("读取 Qdrant 文档来源失败：%s", exc)
        return set()


def _qdrant_get_hashes() -> dict[str, str]:
    """从 Qdrant payload 里收集内容摘要（只取需要的两个字段）。"""
    try:
        client = _get_qdrant_client()
        # 必须显式取 payload，否则 point.payload 为 None，去重会永远失效；
        # 只取 metadata 里的两个字段，避免把全部正文都拉回来
        points, _ = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=10000,
            with_payload=["metadata.sm3_hash", "metadata.file_name"],
            with_vectors=False,
        )
    except Exception as exc:
        logger.debug("读取 Qdrant 内容摘要失败：%s", exc)
        return {}

    return _collect_hashes(
        # LangChain 写入的 payload 为 {page_content, metadata}
        ((point.payload or {}).get("metadata") or {}) for point in points
    )


def _qdrant_get_all_documents() -> list[Document]:
    if not _qdrant_collection_exists():
        raise IndexNotReadyError(
            f"Qdrant collection '{config.QDRANT_COLLECTION}' 不存在，"
            f"请先运行 `python -m app.ingestion` 入库。"
        )
    client = _get_qdrant_client()
    points, _ = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    documents = []
    for point in points:
        payload = point.payload or {}
        content = payload.get("page_content")
        if not content:
            continue
        # LangChain 的 payload 结构为 {page_content, metadata}
        documents.append(
            Document(page_content=content, metadata=payload.get("metadata") or {})
        )
    return documents


def _qdrant_reset():
    try:
        client = _get_qdrant_client()
        client.delete_collection(collection_name=config.QDRANT_COLLECTION)
        logger.info("已删除 Qdrant collection：%s", config.QDRANT_COLLECTION)
    except Exception as exc:
        logger.warning("删除 Qdrant collection 失败：%s", exc)
