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
引用来源时，请原样使用参考资料中给出的位置标识（如"第 3 页，第 2-4 段"或"无页码，第 12-15 段"）。
若参考资料标注为"无页码"，就写"无页码"并给出段号；禁止输出"第 0 页"，也不要自行推算页码。
{history_block}参考资料：
{context}

用户问题：{question}"""

REWRITE_TEMPLATE = """把用户的追问改写成一句能独立理解、用于企业知识库检索的查询词。
要求：
1. 只输出改写后的查询本身，不要解释、不要引号、不要换行。
2. 把上文中的指代（"它""这个""那…呢"等）替换成具体名词。
3. 只保留追问真正要问的对象，不要把上文无关内容带进来。

对话历史：
{history}

用户追问：{question}"""

NO_RESULT_ANSWER = "知识库中未找到相关内容。"

_llm = None
_rewrite_llm = None


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


def get_rewrite_llm() -> ChatTongyi:
    """查询改写专用实例（streaming=False）。

    不复用 get_llm()：那是 streaming=True 的实例，改写只需要一次完整返回，
    分开可以避免动到主回答链路。
    """
    global _rewrite_llm
    if _rewrite_llm is None:
        config.validate()
        _rewrite_llm = ChatTongyi(
            model_name=config.LLM_MODEL,
            dashscope_api_key=config.DASHSCOPE_API_KEY,
            streaming=False,
        )
        logger.debug("已初始化查询改写 LLM %s（streaming=False）", config.LLM_MODEL)
    return _rewrite_llm


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


def _location_label(meta: dict) -> str:
    """根据 chunk metadata 生成位置标识，供拼接进参考资料与引用。

    - PDF（page>=1）：「第 N 页」；Word（page=0 或无物理页）：「无页码」。
    - 均带段落范围「第 M 段 / 第 M-N 段」。M 与 N 相等时只写一段。
    - 任何情况下都不返回"第 0 页"——Word 统一显示"无页码"。
    - 缺 paragraph_* 字段时降级：PDF 只写页、Word 只写"无页码"。
    """
    page = meta.get("page", 0)
    para_start = meta.get("paragraph_start")
    para_end = meta.get("paragraph_end")

    if para_start is not None and para_end is not None:
        para = (
            f"第 {para_start} 段"
            if para_start == para_end
            else f"第 {para_start}-{para_end} 段"
        )
    else:
        para = ""

    if page and page != 0:
        loc = f"第 {page} 页"
    else:
        loc = "无页码"

    return f"{loc}{('，' + para) if para else ''}"


def _build_context(results) -> str:
    """把检索到的片段拼成参考资料文本，每个片段带编号 [i] 与位置标识。

    [i] 的编号顺序与 _to_sources 的 index 完全一致（同一 results 列表、同序枚举），
    这是回答正文 (来源：[2]) 能与界面来源列表对齐的前提。
    """
    blocks = []
    for i, (doc, _score) in enumerate(results, start=1):
        source = doc.metadata.get("file_name") or doc.metadata.get("source", "未知来源")
        loc = _location_label(doc.metadata)
        blocks.append(f"[{i}] 来源：{source}（{loc}）\n内容：{doc.page_content}")
    return "\n\n".join(blocks)


def _to_sources(results) -> list[dict]:
    """把检索结果转成前端的来源列表，序号与 _build_context 的 [i] 严格对齐。

    index 是本次提问的临时序号（1-based），与参考资料里的 [i] 一致，
    便于用户把正文里的 (来源：[2]) 与界面来源列表对上。
    注意：index 只是临时编号，不持久化、不写进索引、不参与去重。
    """
    return [
        {
            "index": i,
            "source": doc.metadata.get("source", "未知来源"),
            "file_name": doc.metadata.get("file_name")
            or doc.metadata.get("source", "未知来源"),
            "page": doc.metadata.get("page", 0),
            "paragraph_start": doc.metadata.get("paragraph_start"),
            "paragraph_end": doc.metadata.get("paragraph_end"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:100],
        }
        for i, (doc, _score) in enumerate(results, start=1)
    ]


def _sanitize_history(history) -> list[dict]:
    """清洗外部传入的 history，只保留 role 为 user/assistant 且内容非空的项。

    REST 是公开入口，history 来自请求体，必须容忍畸形数据而不是抛异常。
    """
    cleaned = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        content = item.get("content")
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _format_history(history: list[dict]) -> str:
    """渲染成"用户：…/助手：…"纯文本。

    只取最后 MAX_HISTORY_TURNS 轮（一轮 = 2 条消息），单条超长按
    MAX_HISTORY_CHARS 截断，避免长对话把 prompt 撑爆。
    """
    if config.MAX_HISTORY_TURNS <= 0:
        return ""
    recent = history[-config.MAX_HISTORY_TURNS * 2 :]
    lines = []
    for item in recent:
        content = item["content"]
        if len(content) > config.MAX_HISTORY_CHARS:
            content = content[: config.MAX_HISTORY_CHARS] + "…"
        speaker = "用户" if item["role"] == "user" else "助手"
        lines.append(f"{speaker}：{content}")
    return "\n".join(lines)


def _history_block(history: list[dict]) -> str:
    """生成拼进 PROMPT_TEMPLATE 的历史块；无历史时返回空串，不占版面。

    历史只用于消解指代，必须明写"不得当作事实来源"，否则模型会拿上一轮的
    回答当依据，脱离本轮参考资料、引用也会失真。

    结尾必须是两个换行：模板里 `{history_block}` 与「参考资料：」同行，
    无历史时占位符替换为空串，prompt 才能与单轮版本逐字节一致；
    有历史时这两个换行负责把历史块与参考资料分隔开。
    """
    if not history:
        return ""
    return (
        "以下是对话历史，仅用于理解指代（例如“它”“这个比赛”指什么）。\n"
        "回答必须依据下面的参考资料，不得把对话历史当作事实来源。\n\n"
        f"{_format_history(history)}\n\n"
    )


def _condense_query(question: str, history: list[dict]) -> str:
    """把追问改写成可独立检索的查询；任何失败都退回原 question。

    「那交通费怎么算？」这类追问几乎没有可召回的关键词，直接拿去检索会捞回
    一堆无关分块，所以先借 history 把指代补全。改写失败绝不能让这一轮挂掉。
    """
    if not history:
        return question

    try:
        response = get_rewrite_llm().invoke(
            [
                HumanMessage(
                    content=REWRITE_TEMPLATE.format(
                        history=_format_history(history), question=question
                    )
                )
            ]
        )
    except Exception:
        logger.warning("查询改写失败，退回原始问题。", exc_info=True)
        return question

    raw = (response.content or "").strip()
    if not raw:
        logger.warning("查询改写结果为空，退回原始问题。")
        return question

    # 模型可能带解释文字或引号，只取第一行并去掉首尾引号
    text = raw.splitlines()[0].strip().strip('"').strip("'").strip()
    if not text:
        logger.warning("查询改写结果为空，退回原始问题。")
        return question
    return text


def _prepare_prompt(
    question: str, k: int, history: list[dict] | None = None
) -> tuple[str, list[dict]]:
    """检索 → 拼装 prompt，返回 (prompt, sources)。

    无命中时返回 ("", [])：此时不发请求，直接用兜底文案。
    有历史时先改写查询再检索，但拼给 LLM 的仍是用户原话（改写只影响检索）。
    """
    history = _sanitize_history(history or [])
    search_query = (
        _condense_query(question, history)
        if (config.MULTITURN_REWRITE and history)
        else question
    )
    if search_query != question:
        logger.info("多轮检索改写：%r → %r", question, search_query)

    results = hybrid_search(search_query, k=k)
    if not results:
        logger.info("检索无结果，直接返回兜底回答。")
        return "", []
    return (
        PROMPT_TEMPLATE.format(
            context=_build_context(results),
            question=question,
            history_block=_history_block(history),
        ),
        _to_sources(results),
    )


def answer_question(
    question: str, k: int = None, history: list[dict] | None = None
) -> dict:
    """检索并生成回答，返回 {"answer": str, "sources": List[dict]}。

    history 为可选的多轮对话历史 [{"role","content"}]；
    不传时行为与单轮完全一致。
    索引缺失时抛出 vector_store.IndexNotReadyError，由调用方决定如何提示。
    """
    config.setup_logging()
    k = k or config.TOP_K

    if not question or not question.strip():
        return {"answer": "请输入问题。", "sources": []}

    prompt, sources = _prepare_prompt(question, k, history)
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


def stream_answer(
    question: str, k: int = None, history: list[dict] | None = None
) -> Iterator[dict]:
    """流式问答：以事件形式产出，供 Gradio 打字机效果与 SSE 接口共用。

    事件类型：
        {"type": "sources", "sources": [...]}  检索完成，先把来源交给前端
        {"type": "delta",   "text": "..."}     逐 token 增量文本
        {"type": "done",    "answer": str, "sources": [...]}
        {"type": "error",   "message": str}    检索或生成失败（已吐出的文本不回滚）

    history 为可选的多轮历史；Gradio 与 /api/ask/stream 都靠它实现多轮。
    与 answer_question 共用 _prepare_prompt，保证两条路径的检索行为完全一致。
    """
    config.setup_logging()
    k = k or config.TOP_K

    if not question or not question.strip():
        yield {"type": "error", "message": "请输入问题。"}
        return

    try:
        prompt, sources = _prepare_prompt(question, k, history)
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
