"""问答检索：混合检索（向量 + BM25，RRF 融合 + Reranker 重排）→ 拼上下文 → 调 LLM。

HYBRID_RETRIEVAL=false 时可退回纯向量检索，便于对照实验与回归。
"""

import hashlib
import logging
from collections.abc import Iterator

import jieba
from dashscope import TextReRank
from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi

from app import config, vector_store

logger = logging.getLogger(__name__)

# RRF（倒数排名融合）的平滑常数，取自原论文的通用取值
RRF_K = 60

# BM25 索引缓存：(文档指纹, BM25 实例, 文档列表)
_bm25_cache: tuple[tuple, BM25Okapi | None, list[Document]] | None = None

PROMPT_TEMPLATE = """你是一个企业知识库助手。请根据以下参考资料回答用户问题。
如果资料中没有相关信息，请如实说明"知识库中未找到相关内容"。
回答时请引用来源文档和页码。

参考资料：
{context}

用户问题：{question}"""

NO_RESULT_ANSWER = "知识库中未找到相关内容。"

_llm = None


def get_llm() -> ChatTongyi:
    """返回通义千问模型实例（进程内复用）。

    必须显式 streaming=True：ChatTongyi 默认 streaming=False，此时 .stream()
    会退化成"生成完整段后一次性返回"（实测只有 1 个 chunk），前端看不到打字机效果。
    打开后 .stream() 才会逐 token 吐出（实测 50+ chunk）。
    """
    global _llm
    if _llm is None:
        config.validate()
        _llm = ChatTongyi(
            model_name=config.LLM_MODEL,
            dashscope_api_key=config.DASHSCOPE_API_KEY,
            streaming=True,
        )
        logger.debug("已初始化 LLM %s（streaming=True）", config.LLM_MODEL)
    return _llm


def _tokenize(text: str) -> list[str]:
    """中文分词。

    必须用 jieba：中文没有空格，按空格切会把整段连成一个 token，
    BM25 会完全失效且不报任何错。
    """
    return [token for token in jieba.cut(text) if token.strip()]


def _chunk_key(doc: Document) -> str:
    """RRF 融合用的唯一 key。

    必须用 chunk_id：同一页 PDF 会切成多个分块，用 source/page 当 key 会把
    同页的多个块累加到同一个 key、映射回 Document 时只捞出一个，其余白算。
    旧索引（阶段一入库，尚无 chunk_id）退回正文摘要，保证不同分块不会被并成一块。
    """
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return chunk_id
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def _get_bm25_index() -> tuple[BM25Okapi | None, list[Document]]:
    """构建并缓存 BM25 索引；索引内容变化时自动重建。"""
    global _bm25_cache

    documents = vector_store.get_all_documents()
    signature = tuple(_chunk_key(doc) for doc in documents)
    if _bm25_cache is not None and _bm25_cache[0] == signature:
        return _bm25_cache[1], _bm25_cache[2]

    if not documents:
        logger.warning("索引中没有文档，BM25 召回将跳过。")
        _bm25_cache = (signature, None, [])
        return None, []

    corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(corpus)
    logger.info("已构建 BM25 索引，共 %d 个分块。", len(documents))
    _bm25_cache = (signature, bm25, documents)
    return bm25, documents


def _bm25_recall(bm25: BM25Okapi, documents: list[Document], query: str, n: int) -> list[Document]:
    """BM25 关键词召回，截断到 n 条，并丢弃 0 分文档。

    BM25 分数为 0 表示与查询没有任何关键词重叠；此时 get_top_n 仍会按原始顺序
    返回 n 条 0 分文档，而 RRF 只看名次不看分数，这些噪声会拿到和真实命中相同的
    权重，把无关分块顶进候选。所以这里按分数过滤而不是直接取 top_n。
    """
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in ranked[:n] if scores[i] > 0]


def _rrf_fuse(ranked_lists: list[list[Document]]) -> list[tuple[Document, float]]:
    """倒数排名融合：score = Σ 1/(RRF_K + rank)，按 chunk_id 归并同一分块。"""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _chunk_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

    fused = [(documents[key], score) for key, score in scores.items()]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def _rerank(query: str, results: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """调用 DashScope rerank API 重排候选；失败则原样返回（降级不中断问答）。

    走 API 而非本地 bge-reranker-v2-m3 之类的模型，避免下载约 2GB 权重。
    """
    try:
        response = TextReRank.call(
            model=config.RERANK_MODEL,
            query=query,
            documents=[doc.page_content for doc, _score in results],
            top_n=len(results),
            api_key=config.DASHSCOPE_API_KEY,
        )
    except Exception:
        logger.exception("Rerank 调用异常，降级为 RRF 顺序。")
        return results

    if response.status_code != 200 or not response.output:
        logger.warning(
            "Rerank 调用失败（status=%s, code=%s），降级为 RRF 顺序。",
            response.status_code,
            getattr(response, "code", ""),
        )
        return results

    reranked = []
    for item in response.output.results:
        index = getattr(item, "index", None)
        score = getattr(item, "relevance_score", None)
        if index is None or not 0 <= index < len(results):
            continue
        reranked.append((results[index][0], score))

    if not reranked:
        logger.warning("Rerank 返回结果为空，降级为 RRF 顺序。")
        return results

    # API 已按相关性返回，这里再排一次以防顺序不保证
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def hybrid_search(query: str, k: int = None) -> list[tuple[Document, float]]:
    """混合检索：向量召回 + BM25 召回 → RRF 融合 → Reranker 重排。

    返回格式与 vector_store.search 一致：[(Document, score)]，便于无缝替换。
    HYBRID_RETRIEVAL=false 时退化为纯向量检索。
    """
    k = k or config.TOP_K

    if not config.HYBRID_RETRIEVAL:
        logger.info("HYBRID_RETRIEVAL=false，使用纯向量检索。")
        return vector_store.search(query, k=k)

    candidate_k = max(k, config.TOP_K * config.HYBRID_CANDIDATE_MULTIPLIER)

    # 1) 向量召回
    vector_hits = [doc for doc, _score in vector_store.search(query, k=candidate_k)]

    # 2) BM25 关键词召回（中文用 jieba 分词）
    bm25, documents = _get_bm25_index()
    bm25_hits = _bm25_recall(bm25, documents, query, candidate_k) if bm25 else []

    if any(not doc.metadata.get("chunk_id") for doc in vector_hits + bm25_hits):
        logger.warning("部分分块缺少 chunk_id（旧索引），建议执行 --rebuild 重建。")

    # 3) RRF 融合
    fused = _rrf_fuse([vector_hits, bm25_hits])
    if not fused:
        return []

    # 4) Reranker 重排 top-N，再截断到 k
    candidates = fused[: max(k, config.RERANK_TOP_N)]
    if config.RERANK_ENABLED and len(candidates) > 1:
        candidates = _rerank(query, candidates)

    logger.info(
        "混合检索 %r：向量 %d 条 / BM25 %d 条 → 融合 %d 条 → 返回 %d 条。",
        query[:30], len(vector_hits), len(bm25_hits), len(fused), min(k, len(candidates)),
    )
    return candidates[:k]


def _build_context(results) -> str:
    """把检索到的片段拼成参考资料文本。"""
    blocks = []
    for i, (doc, _score) in enumerate(results, start=1):
        source = doc.metadata.get("file_name") or doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{i}] 来源：{source}, 第{page}页\n内容：{doc.page_content}")
    return "\n\n".join(blocks)


def _to_sources(results) -> list[dict]:
    """把检索结果转成前端的来源列表。"""
    return [
        {
            "source": doc.metadata.get("source", "未知来源"),
            "file_name": doc.metadata.get("file_name")
            or doc.metadata.get("source", "未知来源"),
            "page": doc.metadata.get("page", "?"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:100],
        }
        for doc, _score in results
    ]


def _prepare_prompt(question: str, k: int) -> tuple[str, list[dict]]:
    """检索 → 拼装 prompt，返回 (prompt, sources)。

    无命中时返回 ("", [])：此时不发请求，直接用兜底文案。
    """
    results = hybrid_search(question, k=k)
    if not results:
        logger.info("检索无结果，直接返回兜底回答。")
        return "", []
    return (
        PROMPT_TEMPLATE.format(context=_build_context(results), question=question),
        _to_sources(results),
    )


def answer_question(question: str, k: int = None) -> dict:
    """检索并生成回答，返回 {"answer": str, "sources": List[dict]}。

    索引缺失时抛出 vector_store.IndexNotReadyError，由调用方决定如何提示。
    """
    config.setup_logging()
    k = k or config.TOP_K

    if not question or not question.strip():
        return {"answer": "请输入问题。", "sources": []}

    prompt, sources = _prepare_prompt(question, k)
    if not prompt:
        return {"answer": NO_RESULT_ANSWER, "sources": []}

    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("调用 LLM 失败。")
        return {
            "answer": "调用大模型失败，请稍后重试或检查 DASHSCOPE_API_KEY 与网络。",
            "sources": sources,
        }

    logger.info("已生成回答（参考 %d 个片段）。", len(sources))
    return {"answer": response.content, "sources": sources}


def stream_answer(question: str, k: int = None) -> Iterator[dict]:
    """流式问答：以事件形式产出，供 Gradio 打字机效果与 SSE 接口共用。

    事件类型：
        {"type": "sources", "sources": [...]}  检索完成，先把来源交给前端
        {"type": "delta",   "text": "..."}     逐 token 增量文本
        {"type": "done",    "answer": str, "sources": [...]}
        {"type": "error",   "message": str}    检索或生成失败（已吐出的文本不回滚）

    与 answer_question 共用 _prepare_prompt，保证两条路径的检索行为完全一致。
    """
    config.setup_logging()
    k = k or config.TOP_K

    if not question or not question.strip():
        yield {"type": "error", "message": "请输入问题。"}
        return

    try:
        prompt, sources = _prepare_prompt(question, k)
    except vector_store.IndexNotReadyError as exc:
        yield {"type": "error", "message": str(exc)}
        return
    except Exception as exc:
        logger.exception("检索失败。")
        yield {"type": "error", "message": f"检索失败：{exc}"}
        return

    if not prompt:
        yield {"type": "delta", "text": NO_RESULT_ANSWER}
        yield {"type": "done", "answer": NO_RESULT_ANSWER, "sources": []}
        return

    yield {"type": "sources", "sources": sources}

    parts: list[str] = []
    try:
        for chunk in get_llm().stream([HumanMessage(content=prompt)]):
            text = chunk.content or ""
            if not text:
                continue
            parts.append(text)
            yield {"type": "delta", "text": text}
    except Exception:
        logger.exception("流式调用 LLM 失败。")
        yield {"type": "error", "message": "⚠️ 生成中断，请稍后重试。"}
        return

    answer = "".join(parts)
    logger.info("已流式生成回答（参考 %d 个片段，%d 个增量块）。", len(sources), len(parts))
    yield {"type": "done", "answer": answer, "sources": sources}
