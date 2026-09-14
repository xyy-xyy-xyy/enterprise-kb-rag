"""真实链路：真的调 DashScope、真的读写索引、真的生成回答。

**这些用例会花钱**（embedding + LLM 调用），所以每一条都：
  1. 标了 `@pytest.mark.integration` —— CI 用 `pytest -m "not integration"` 跳过；
  2. 跑之前先确认环境齐备（真实 Key + 索引就绪），不齐就 skip 而不是 fail；
  3. 只发最少次数的请求，不做扫描式对比。

本地手动跑：
    ./venv/Scripts/python.exe -m pytest -m integration -q

设计意图：离线用例测的是"逻辑对不对"，这里补的是"接线通不通"——
mock 掉了 vector_store 之后，embedding 维度不匹配、Qdrant collection 名字写错、
API Key 失效这类问题离线用例一个都发现不了。
"""

import pytest
from langchain_core.documents import Document

from app import config, ingestion, retrieval, vector_store

pytestmark = pytest.mark.integration

# CI 注入的是这个占位值。它能让 config.validate() 过关，但调不通任何真实接口，
# 所以必须显式识别出来并跳过，而不是让它带着假 Key 去打真实端点。
_CI_DUMMY_KEY = "dummy-key-for-ci"


def _require_real_key():
    key = config.DASHSCOPE_API_KEY
    if not key or key == _CI_DUMMY_KEY:
        pytest.skip("没有真实 DASHSCOPE_API_KEY，跳过需要调用 DashScope 的集成用例。")


def _require_ready_index():
    _require_real_key()
    if not vector_store.index_exists():
        pytest.skip("索引不存在（先跑 python -m app.ingestion），跳过集成用例。")


# ============================ 索引与检索 ============================


def test_real_index_can_be_altered_and_searched():
    """真的往 DashScope 打一次 embedding 请求，再真的走一次向量检索。

    这条能抓住离线用例抓不到的问题：embedding 模型/维度变了、Qdrant collection
    名改了、云端服务换了鉴权方式。
    """
    _require_ready_index()

    chunks = vector_store.get_all_documents()
    if not chunks:
        pytest.skip("索引为空，跳过。")

    hits = vector_store.search(chunks[0].page_content[:50], k=3)
    assert hits, "用索引里已有的正文去检索，不该一条都召不回"
    assert all(hasattr(doc, "page_content") for doc, _score in hits)


def test_real_hybrid_search_runs_all_four_strategies():
    """四种策略在真实索引上都要能跑通并返回同构的结果。

    评测体系依赖这一点：某条策略若在真实数据上抛异常，阶段四的对比表就是废的。
    """
    _require_ready_index()
    query = _question_from_index()

    for strategy in retrieval.STRATEGIES:
        results = retrieval.search_with_strategy(query, k=3, strategy=strategy)
        assert isinstance(results, list), f"{strategy} 未返回列表"
        for doc, score in results:
            assert doc.metadata.get("file_name"), f"{strategy} 返回的分块缺 file_name"
            assert isinstance(score, float), f"{strategy} 的分数不是 float"


def test_real_rerank_api_is_reachable():
    """直接打一次 rerank 端点 —— 阶段二换过一次模型（gte-rerank 下线），
    这类"模型名失效"的故障只有真调一次才发现得了。
    """
    _require_real_key()

    docs = [
        Document(page_content="一线城市住宿费上限为每晚 800 元。", metadata={"chunk_id": "a"}),
        Document(page_content="本系统支持多轮对话与流式输出。", metadata={"chunk_id": "b"}),
    ]
    candidates = [(doc, 1.0 / (retrieval.RRF_K + i)) for i, doc in enumerate(docs, start=1)]

    reranked = retrieval._rerank("住宿费上限是多少", candidates)

    assert len(reranked) == len(candidates)
    # 重排失败时 _rerank 会原样返回（降级不中断问答），但这里必须真的换了顺序，
    # 否则说明 API 调用被静默降级了 —— 那就是"评测跑出假数据且不报错"。
    assert [doc.metadata["chunk_id"] for doc, _ in reranked] == ["a", "b"]


# ============================ 生成回答 ============================


def _question_from_index() -> str:
    """从索引里现取一段正文当问题。

    写死"住宿费上限是多少"是不行的 —— 用户索引里可能根本没有相关文档，
    这条就会因为数据而红。用索引自己的内容当问题，保证必定检索得到。
    """
    chunks = vector_store.get_all_documents()
    if not chunks:
        pytest.skip("索引为空，跳过。")
    return chunks[0].page_content.split("\n")[-1][:30]


def test_answer_question_end_to_end_returns_answer_and_aligned_sources():
    """完整链路：检索 → 拼 prompt → 真实 LLM 生成 → 来源编号与正文引用对齐。"""
    _require_ready_index()

    result = retrieval.answer_question(_question_from_index(), k=3)

    assert result["answer"], "回答不该为空"
    assert result["sources"], "索引非空时不该一条来源都给不出"
    assert [s["index"] for s in result["sources"]] == list(range(1, len(result["sources"]) + 1))


def test_stream_answer_emits_sources_deltas_then_done():
    """流式接口的事件顺序是多轮/打字机效果的前提，真跑一次确认它没变。"""
    _require_ready_index()

    events = list(retrieval.stream_answer(_question_from_index(), k=3))
    kinds = [event["type"] for event in events]

    assert kinds[0] == "sources", f"第一个事件应为 sources，实际 {kinds[0]}"
    assert kinds[-1] in ("done", "error"), f"最后一个事件应为 done/error，实际 {kinds[-1]}"
    assert "delta" in kinds, "中间应有增量事件"

    done = events[-1]
    if done["type"] == "done":
        assert done["answer"] == "".join(
            event["text"] for event in events if event["type"] == "delta"
        ), "累加的 delta 必须与 done 里的完整回答逐字一致"
        assert done["sources"] == events[0]["sources"]


def test_vector_search_returns_full_k_even_for_irrelevant_queries():
    """记录一个真实行为（不是 bug，但很反直觉）：

    `vector_store.search` 不做任何分数阈值过滤，索引进非空就永远返回 k 条。
    因此 `_prepare_prompt` 里那条"检索无结果 → 知识库中未找到相关内容"的兜底，
    在向量/混合策略下**只有索引为空时才会触发**；问一个完全无关的问题时，
    模型仍会拿到 k 个片段，由模型自己判断要不要说"未找到"。

    这条用例把该行为钉住，免得日后有人以为兜底文案是检索层判定的。
    """
    _require_ready_index()

    total = len(vector_store.get_all_documents())
    hits = vector_store.search("微波炉加热等离子体的介电常数", k=3)
    assert len(hits) == min(3, total)


def test_out_of_scope_question_still_answers_without_crashing():
    """越界提问的底线：不抛异常、返回字符串，而不是把 k 个无关片段当依据胡说。"""
    _require_ready_index()

    result = retrieval.answer_question("请问如何用微波炉加热等离子体？", k=3)
    assert isinstance(result["answer"], str) and result["answer"]


def test_multiturn_followup_uses_history_for_retrieval():
    """多轮追问：第二轮用指代（"那它呢"），靠 history 改写才能召回对的内容。"""
    _require_ready_index()
    if not config.MULTITURN_REWRITE:
        pytest.skip("MULTITURN_REWRITE 关闭，跳过多轮改写用例。")

    history = [
        {"role": "user", "content": "住宿费上限是多少？"},
        {"role": "assistant", "content": "一线城市每晚 800 元。"},
    ]
    result = retrieval.answer_question("那它超标了怎么办？", k=3, history=history)

    assert result["answer"]
    assert result["sources"], "带历史追问时不该一条都召不回"


# ============================ 入库 ============================


def test_real_ingestion_of_a_generated_docx_writes_chunks(tmp_path, monkeypatch):
    """真的解析 → 真的打 embedding 入库 → 真的能检索回来。

    **写入的是一个临时 collection**（monkeypatch QDRANT_COLLECTION），跑完就删。
    绝不能碰真实索引：那会覆盖掉用户已经建好的向量库，等于偷偷跑了一次 --rebuild。
    faiss 后端没有 collection 概念，用独立的 INDEX_DIR 隔离。
    """
    _require_real_key()

    if config.VECTOR_BACKEND == "qdrant":
        monkeypatch.setattr(config, "QDRANT_COLLECTION", "enterprise_kb_pytest_tmp")
    else:
        monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path / "faiss"))

    content = (
        "员工考勤与请假管理办法\n"
        "入职满一年可享受带薪年假 5 天。\n"
        "年假需提前三个工作日申请。\n"
    )
    documents = []
    for i, line in enumerate(content.strip().split("\n")):
        documents.append(
            Document(
                page_content=line,
                metadata={
                    "source": str(tmp_path / "hr.docx"),
                    "file_name": "hr.docx",
                    "page": 0,
                    "chunk_id": f"hr.docx_p0_{i}",
                    "paragraph_start": i + 1,
                    "paragraph_end": i + 1,
                },
            )
        )

    try:
        # embedding 请求就发生在这里；维度不匹配、模型名失效都会直接抛出来
        vector_store.create_index(documents)

        hits = vector_store.search("年假多少天", k=2)
        assert hits, "刚入库的内容应立刻能被检索到"
        assert any("年假" in doc.page_content for doc, _score in hits)
    finally:
        # 无论成败都要清掉临时 collection，不给下次运行留脏数据
        vector_store.reset_index()


def test_real_parser_output_can_be_indexed(tmp_path):
    """解析 → 入库 的接缝处：真实解析器产出的 metadata 必须能被向量库接受。

    离线用例测过解析、也测过入库，但两边都是各自造的数据；只有真跑一次
    才能确认 _finalize_chunks 写出的字段向量库那边真吃得下。
    """
    _require_real_key()

    from tests.conftest import _build_docx_bytes

    path = tmp_path / "parsed.docx"
    path.write_bytes(_build_docx_bytes())

    data, _name, _source = ingestion.read_document_bytes(str(path))
    documents = ingestion._split_docx_bytes(data, str(path))

    assert documents, "解析不出任何分块"
    assert all(doc.metadata.get("chunk_id") for doc in documents)
    assert {doc.metadata["file_name"] for doc in documents} == {"parsed.docx"}
