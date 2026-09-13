"""问答检索：向量检索 → 拼上下文 → 调 LLM → 返回答案 + 来源。"""

import logging

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

from app import config, vector_store

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """你是一个企业知识库助手。请根据以下参考资料回答用户问题。
如果资料中没有相关信息，请如实说明"知识库中未找到相关内容"。
回答时请引用来源文档和页码。

参考资料：
{context}

用户问题：{question}"""

NO_RESULT_ANSWER = "知识库中未找到相关内容。"

_llm = None


def get_llm() -> ChatTongyi:
    """返回通义千问模型实例（进程内复用）。"""
    global _llm
    if _llm is None:
        config.validate()
        _llm = ChatTongyi(
            model_name=config.LLM_MODEL,
            dashscope_api_key=config.DASHSCOPE_API_KEY,
        )
        logger.debug("已初始化 LLM %s", config.LLM_MODEL)
    return _llm


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
            "snippet": doc.page_content[:100],
        }
        for doc, _score in results
    ]


def answer_question(question: str, k: int = None) -> dict:
    """检索并生成回答，返回 {"answer": str, "sources": List[dict]}。

    索引缺失时抛出 vector_store.IndexNotReadyError，由调用方决定如何提示。
    """
    config.setup_logging()
    k = k or config.TOP_K

    if not question or not question.strip():
        return {"answer": "请输入问题。", "sources": []}

    results = vector_store.search(question, k=k)
    if not results:
        logger.info("检索无结果，直接返回兜底回答。")
        return {"answer": NO_RESULT_ANSWER, "sources": []}

    context = _build_context(results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("调用 LLM 失败。")
        return {
            "answer": "调用大模型失败，请稍后重试或检查 DASHSCOPE_API_KEY 与网络。",
            "sources": _to_sources(results),
        }

    logger.info("已生成回答（参考 %d 个片段）。", len(results))
    return {"answer": response.content, "sources": _to_sources(results)}
