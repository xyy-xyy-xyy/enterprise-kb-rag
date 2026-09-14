"""检索层纯函数：分词、RRF 融合、引用定位、历史渲染、策略推导。

本文件里没有一次真实网络调用 —— `vector_store.search` / `get_all_documents`
全部被替换成假数据，检索链路本身的行为由此变得可断言。
"""

import hashlib

import pytest
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app import config, retrieval


@pytest.fixture
def patched_bm25(monkeypatch, fake_documents):
    """把 BM25 语料换成 fake_documents，并清掉模块级缓存。

    `_bm25_cache` 是模块全局变量，不清掉的话上一条用例建的索引会串到下一条。
    """
    monkeypatch.setattr(retrieval.vector_store, "get_all_documents", lambda: fake_documents)
    monkeypatch.setattr(retrieval, "_bm25_cache", None)
    return fake_documents


def _ids(results):
    return [doc.metadata["chunk_id"] for doc, _score in results]


def _id_of(documents, file_name):
    """按文件名取 chunk_id —— 断言里直接写中文 chunk_id 太长，可读性差。"""
    return next(d.metadata["chunk_id"] for d in documents if d.metadata["file_name"] == file_name)


# ============================ _tokenize ============================


def test_tokenize_splits_chinese_into_words_not_characters_or_whole_line():
    """按空格切会把整句变成一个 token，BM25 会静默失效 —— 这条守住分词这件事本身。"""
    tokens = retrieval._tokenize("一线城市住宿费上限")
    assert "住宿费" in tokens
    assert "一线城市住宿费上限" not in tokens
    assert len(tokens) > 1


def test_tokenize_drops_whitespace_only_tokens():
    assert "" not in retrieval._tokenize("住宿费 800 元")
    assert all(token.strip() for token in retrieval._tokenize("住宿费 800 元"))


# ============================ _chunk_key ============================


def test_chunk_key_prefers_chunk_id():
    doc = Document(page_content="甲", metadata={"chunk_id": "a.pdf_p1_3"})
    assert retrieval._chunk_key(doc) == "a.pdf_p1_3"


def test_chunk_key_falls_back_to_content_hash_for_legacy_index(fake_documents):
    """阶段一的旧索引没有 chunk_id，此时必须按正文区分，不能塌缩成同一个 key。"""
    legacy_a = Document(page_content="甲", metadata={})
    legacy_b = Document(page_content="乙", metadata={})
    assert retrieval._chunk_key(legacy_a) == hashlib.md5("甲".encode()).hexdigest()
    assert retrieval._chunk_key(legacy_a) != retrieval._chunk_key(legacy_b)


def test_chunk_key_separates_two_chunks_from_the_same_page(fake_documents):
    """同一页切成多块时 page 相同、chunk_id 必须不同，否则融合会把它们并成一块。"""
    doc_a, _b, _c, _d = fake_documents
    same_page = Document(page_content="另一段", metadata={"chunk_id": doc_a.metadata["chunk_id"] + "x"})
    assert retrieval._chunk_key(doc_a) != retrieval._chunk_key(same_page)


# ============================ _rrf_fuse ============================


def test_rrf_fuse_ranks_by_summed_reciprocal_rank(fake_documents):
    """精确顺序：A(向量1+BM25 2) > C(向量3+BM25 1) > B(仅向量2)。

    这条是 RRF 的核心契约 —— 只看名次、不看原始分数，两个列表都靠前的赢。
    """
    a, b, c, _d = fake_documents
    fused = retrieval._rrf_fuse([[a, b, c], [c, a]])

    assert _ids(fused) == [a.metadata["chunk_id"], c.metadata["chunk_id"], b.metadata["chunk_id"]]
    assert fused[0][1] == pytest.approx(1.0 / (retrieval.RRF_K + 1) + 1.0 / (retrieval.RRF_K + 2))
    assert fused[1][1] == pytest.approx(1.0 / (retrieval.RRF_K + 3) + 1.0 / (retrieval.RRF_K + 1))


def test_rrf_fuse_merges_the_same_chunk_from_both_lists_into_one_entry(fake_documents):
    """同一个分块被两路召回时必须归并成一条、分数相加；否则候选窗口里全是重复项。"""
    a, _b, _c, _d = fake_documents
    fused = retrieval._rrf_fuse([[a], [a]])
    assert len(fused) == 1
    assert fused[0][1] == pytest.approx(2.0 / (retrieval.RRF_K + 1))


def test_rrf_fuse_ignores_empty_lists_and_empty_input():
    assert retrieval._rrf_fuse([]) == []
    assert retrieval._rrf_fuse([[], []]) == []


def test_rrf_fuse_handles_a_single_list_as_plain_ranking(fake_documents):
    a, b, _c, _d = fake_documents
    assert _ids(retrieval._rrf_fuse([[a, b]])) == [
        a.metadata["chunk_id"],
        b.metadata["chunk_id"],
    ]


# ============================ _bm25_recall / bm25_search ============================


def test_bm25_recall_drops_zero_score_documents(patched_bm25):
    """BM25 取 top_n 时会把 0 分文档也带出来，RRF 只看名次，这些噪声会被当真实命中。"""
    bm25, documents = retrieval._get_bm25_index()
    hits = retrieval._bm25_recall(bm25, documents, "住宿费上限", n=10)
    assert [doc.metadata["chunk_id"] for doc in hits] == [
        _id_of(documents, "星尘计划差旅报销管理办法.docx")
    ]


def test_bm25_recall_drops_terms_whose_idf_collapses_to_zero(patched_bm25):
    """真实且反直觉：4 篇语料里"住宿费"命中 2 篇，IDF 恰好为 0 → 全部分数为 0 → 全被过滤。

    `_bm25_recall` 用 `score > 0` 当"有无关键词重叠"的判据，但 BM25Okapi 对
    出现在半数以上文档里的词会把 IDF 压成 0，此时"有重叠"也会得 0 分被丢掉。
    小语料上单关键词查询召回为空，就是这个原因。
    """
    bm25, documents = retrieval._get_bm25_index()
    assert retrieval._bm25_recall(bm25, documents, "住宿费", n=10) == []

    # 补上一个有区分度的词，同一条文档立刻能被召回
    hits = retrieval._bm25_recall(bm25, documents, "住宿费上限", n=10)
    assert [doc.metadata["chunk_id"] for doc in hits] == [
        _id_of(documents, "星尘计划差旅报销管理办法.docx")
    ]


def test_bm25_recall_returns_empty_when_nothing_matches(patched_bm25):
    bm25, documents = retrieval._get_bm25_index()
    assert retrieval._bm25_recall(bm25, documents, "量子纠缠超导", n=10) == []


def test_bm25_recall_respects_the_n_limit():
    """6 篇语料里"报销"命中 2 篇（IDF>0，不会被 0 分过滤吃掉），n 才能真的起到截断作用。"""
    documents = [
        Document(page_content=text, metadata={"chunk_id": f"d{i}"})
        for i, text in enumerate(
            [
                "报销单据需在十五个工作日内提交。",
                "报销流程由财务部负责审核。",
                "年假天数为五天。",
                "考勤以打卡记录为准。",
                "公司实行弹性工作制。",
                "会议室需要提前预约。",
            ]
        )
    ]
    bm25 = BM25Okapi([retrieval._tokenize(doc.page_content) for doc in documents])

    assert len(retrieval._bm25_recall(bm25, documents, "报销", n=1)) == 1
    assert len(retrieval._bm25_recall(bm25, documents, "报销", n=5)) == 2


def test_bm25_search_scores_by_rank_not_raw_bm25(patched_bm25):
    """分数折算成 1/(RRF_K+rank)：两路召回同量纲，日志里的分数才有可比性。"""
    results = retrieval.bm25_search("住宿费上限", k=5)

    assert len(results) == 1
    assert results[0][1] == pytest.approx(1.0 / (retrieval.RRF_K + 1))


def test_bm25_search_score_depends_only_on_rank_not_term_frequency(monkeypatch):
    """词频高、文档短的那条 BM25 原始分会明显更大，但对外分数必须只由名次决定。

    否则融合时 BM25 侧会因为原始分数量纲不同而压过向量侧 —— 这正是折算的理由。
    """
    documents = [
        Document(page_content="报销 报销 报销 报销 报销 报销 报销 报销。", metadata={"chunk_id": "high"}),
        Document(page_content="报销单据需在十五个工作日内提交至财务部门审核处理完毕。", metadata={"chunk_id": "low"}),
        Document(page_content="年假天数为五天。", metadata={"chunk_id": "none"}),
    ]
    bm25 = BM25Okapi([retrieval._tokenize(doc.page_content) for doc in documents])
    monkeypatch.setattr(retrieval, "_get_bm25_index", lambda: (bm25, documents))

    results = retrieval.bm25_search("报销", k=5)
    assert _ids(results) == ["high", "low"]
    assert [score for _doc, score in results] == pytest.approx(
        [1.0 / (retrieval.RRF_K + 1), 1.0 / (retrieval.RRF_K + 2)]
    )


def test_bm25_search_returns_empty_on_empty_index(monkeypatch):
    monkeypatch.setattr(retrieval.vector_store, "get_all_documents", list)
    monkeypatch.setattr(retrieval, "_bm25_cache", None)
    assert retrieval.bm25_search("住宿费") == []


# ============================ _resolve_strategy ============================


@pytest.mark.parametrize("strategy", list(retrieval.STRATEGIES))
def test_resolve_strategy_passes_explicit_values_through(strategy):
    assert retrieval._resolve_strategy(strategy) == strategy


def test_resolve_strategy_rejects_unknown_value_loudly():
    """静默退回默认策略会让评测跑出假数据且不报错 —— 这是最危险的一类失败。"""
    with pytest.raises(ValueError, match="未知的检索策略"):
        retrieval._resolve_strategy("bogus")


@pytest.mark.parametrize(
    "hybrid,rerank,expected",
    [
        (False, False, retrieval.STRATEGY_VECTOR),
        (False, True, retrieval.STRATEGY_VECTOR),
        (True, False, retrieval.STRATEGY_HYBRID),
        (True, True, retrieval.STRATEGY_HYBRID_RERANK),
    ],
)
def test_resolve_strategy_derives_from_config(monkeypatch, hybrid, rerank, expected):
    """strategy=None 时线上行为由 config 决定，三个分支都必须对得上。"""
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", hybrid)
    monkeypatch.setattr(config, "RERANK_ENABLED", rerank)
    assert retrieval._resolve_strategy(None) == expected


def test_resolve_strategy_none_ignores_hybrid_when_hybrid_disabled(monkeypatch):
    """HYBRID_RETRIEVAL=false 时即使 RERANK_ENABLED=true 也必须退回纯向量。"""
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    assert retrieval._resolve_strategy(None) == retrieval.STRATEGY_VECTOR


# ============================ search_with_strategy ============================


def test_vector_strategy_returns_vector_store_results_untouched(monkeypatch, fake_documents):
    a, b, _c, _d = fake_documents
    expected = [(a, 0.91), (b, 0.82)]
    monkeypatch.setattr(retrieval.vector_store, "search", lambda q, k=None: expected)
    assert retrieval.search_with_strategy("住宿费", k=2, strategy="vector") is expected


def test_hybrid_fuses_both_recall_paths(monkeypatch, fake_documents, patched_bm25):
    """向量只给 [A, B]，BM25 只命中 C —— 融合后三者都应在，且两个 rank-1 排在 B 前。"""
    a, b, c, _d = fake_documents
    monkeypatch.setattr(retrieval.vector_store, "search", lambda q, k=None: [(a, 0.9), (b, 0.8)])

    results = retrieval.search_with_strategy("超标", k=3, strategy="hybrid")
    order = _ids(results)

    assert set(order) == {a.metadata["chunk_id"], b.metadata["chunk_id"], c.metadata["chunk_id"]}
    assert order.index(a.metadata["chunk_id"]) < order.index(b.metadata["chunk_id"])
    assert order.index(c.metadata["chunk_id"]) < order.index(b.metadata["chunk_id"])


def test_hybrid_rerank_returns_reranker_order_not_rrf_order(monkeypatch, fake_documents, patched_bm25):
    """把 _rerank 换成"整个倒过来"，输出顺序若跟着变，说明重排结果确实是最终顺序。"""
    a, b, c, _d = fake_documents
    monkeypatch.setattr(retrieval.vector_store, "search", lambda q, k=None: [(a, 0.9), (b, 0.8)])
    monkeypatch.setattr(retrieval, "_rerank", lambda query, results: list(reversed(results)))

    rrf_order = _ids(retrieval.search_with_strategy("超标", k=3, strategy="hybrid"))
    rerank_order = _ids(retrieval.search_with_strategy("超标", k=3, strategy="hybrid_rerank"))

    assert rerank_order == list(reversed(rrf_order))


def test_hybrid_returns_empty_when_both_paths_recall_nothing(monkeypatch, fake_documents, patched_bm25):
    monkeypatch.setattr(retrieval.vector_store, "search", lambda q, k=None: [])
    assert retrieval.search_with_strategy("量子纠缠", k=3, strategy="hybrid") == []


def test_hybrid_search_wrapper_follows_config(monkeypatch, fake_documents):
    """hybrid_search 是对外入口，HYBRID_RETRIEVAL=false 时必须还是纯向量。"""
    a, _b, _c, _d = fake_documents
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(retrieval.vector_store, "search", lambda q, k=None: [(a, 0.9)])
    assert retrieval.hybrid_search("住宿费") == [(a, 0.9)]


def test_hybrid_candidates_truncates_to_rerank_window(monkeypatch, fake_documents, patched_bm25):
    """重排只看窗口内的候选，超出窗口的必须在调 API 前就砍掉（省钱）。"""
    a, b, c, d = fake_documents
    monkeypatch.setattr(config, "RERANK_TOP_N", 2)
    monkeypatch.setattr(
        retrieval.vector_store, "search", lambda q, k=None: [(a, 0.9), (b, 0.8), (c, 0.7), (d, 0.6)]
    )

    # 窗口 = max(k, RERANK_TOP_N)：k 更大时不能被重排窗口砍到少于 k 条
    candidates, n_vector, _n_bm25, n_fused = retrieval._hybrid_candidates("住宿费上限", k=4)
    assert len(candidates) == 4

    # k 更小时窗口由 RERANK_TOP_N 决定，4 条候选被截到 2 条
    candidates, n_vector, _n_bm25, n_fused = retrieval._hybrid_candidates("住宿费上限", k=1)
    assert len(candidates) == 2
    assert n_vector == 4
    assert n_fused == 4


# ============================ _location_label ============================


@pytest.mark.parametrize(
    "meta,expected",
    [
        # PDF：有页码 + 段落区间
        ({"page": 3, "paragraph_start": 2, "paragraph_end": 4}, "第 3 页，第 2-4 段"),
        # Word：无物理页 + 段落区间（不是"第 0 页"）
        ({"page": 0, "paragraph_start": 6, "paragraph_end": 10}, "无页码，第 6-10 段"),
        # 单段落不写成 "第 5-5 段"
        ({"page": 5, "paragraph_start": 5, "paragraph_end": 5}, "第 5 页，第 5 段"),
        ({"page": 0, "paragraph_start": 7, "paragraph_end": 7}, "无页码，第 7 段"),
        # 降级：缺 paragraph_* 时只写页
        ({"page": 3}, "第 3 页"),
        ({"page": 0}, "无页码"),
        ({}, "无页码"),
    ],
)
def test_location_label_exact_strings(meta, expected):
    assert retrieval._location_label(meta) == expected


def test_location_label_never_emits_page_zero():
    """提示词里明令禁止"第 0 页"；Word 的 page=0 必须走"无页码"。"""
    for meta in ({"page": 0}, {"page": 0, "paragraph_start": 1, "paragraph_end": 2}, {}):
        assert "第 0 页" not in retrieval._location_label(meta)


# ============================ _build_context / _to_sources ============================


def test_build_context_numbers_blocks_from_one_in_order(fake_documents):
    a, b, c, d = fake_documents
    context = retrieval._build_context([(a, 0.9), (b, 0.8), (c, 0.7), (d, 0.6)])

    assert context.startswith("[1] 来源：星尘计划差旅报销管理办法.docx（无页码，第 2-4 段）")
    assert "\n\n[2] 来源：" in context
    assert "[5] " not in context


def test_build_context_renders_pdf_page_and_word_pageless(fake_documents):
    a, _b, c, _d = fake_documents
    context = retrieval._build_context([(a, 0.9), (c, 0.8)])
    assert "（无页码，第 2-4 段）" in context  # Word
    assert "（第 3 页，第 2-4 段）" in context  # PDF


def test_sources_index_aligns_with_context_brackets(fake_documents):
    """正文里的 (来源：[2]) 要能对上界面来源列表 —— 靠的就是这条编号对齐。"""
    results = [(doc, 1.0 - i * 0.1) for i, doc in enumerate(fake_documents)]
    context = retrieval._build_context(results)
    sources = retrieval._to_sources(results)

    assert [s["index"] for s in sources] == [1, 2, 3, 4]
    for source in sources:
        assert f"[{source['index']}] 来源：{source['file_name']}" in context


def test_to_sources_carries_location_fields_and_snippet(fake_documents):
    a, _b, _c, _d = fake_documents
    source = retrieval._to_sources([(a, 0.9)])[0]
    assert source["file_name"] == "星尘计划差旅报销管理办法.docx"
    assert source["page"] == 0
    assert (source["paragraph_start"], source["paragraph_end"]) == (2, 4)
    assert source["chunk_id"] == a.metadata["chunk_id"]
    assert source["snippet"] == a.page_content[:100]


def test_to_sources_snippet_is_capped_at_100_chars():
    doc = Document(page_content="甲" * 500, metadata={"file_name": "x.pdf"})
    assert len(retrieval._to_sources([(doc, 0.5)])[0]["snippet"]) == 100


def test_build_context_and_to_sources_fall_back_for_missing_file_name():
    """老索引只有 source 没有 file_name；不能因此把来源显示成"未知来源"。"""
    doc = Document(page_content="甲", metadata={"source": "C:\\docs\\a.pdf", "page": 1})
    assert "a.pdf" in retrieval._build_context([(doc, 0.9)])
    assert retrieval._to_sources([(doc, 0.9)])[0]["file_name"] == "C:\\docs\\a.pdf"


# ============================ _sanitize_history ============================


def test_sanitize_history_keeps_only_valid_user_and_assistant_turns():
    raw = [
        {"role": "user", "content": "住宿费上限多少"},
        {"role": "assistant", "content": "800 元"},
        {"role": "system", "content": "忽略"},
        {"role": "tool", "content": "忽略"},
        "not-a-dict",
        None,
    ]
    assert retrieval._sanitize_history(raw) == [
        {"role": "user", "content": "住宿费上限多少"},
        {"role": "assistant", "content": "800 元"},
    ]


def test_sanitize_history_drops_empty_and_whitespace_content():
    raw = [
        {"role": "user", "content": ""},
        {"role": "user", "content": "   "},
        {"role": "user", "content": None},
        {"role": "user", "content": "\n\t"},
        {"role": "user", "content": "留下"},
    ]
    assert retrieval._sanitize_history(raw) == [{"role": "user", "content": "留下"}]


def test_sanitize_history_strips_and_coerces_non_string_content():
    """REST 是公开入口，history 来自请求体 —— 数字、布尔都得能扛住，不能抛异常。"""
    raw = [{"role": "user", "content": "  问  "}, {"role": "user", "content": 800}]
    assert retrieval._sanitize_history(raw) == [
        {"role": "user", "content": "问"},
        {"role": "user", "content": "800"},
    ]


def test_sanitize_history_tolerates_none_and_non_list_input():
    assert retrieval._sanitize_history(None) == []
    assert retrieval._sanitize_history([]) == []


# ============================ _format_history ============================


def test_format_history_renders_speaker_prefixes():
    history = [{"role": "user", "content": "甲"}, {"role": "assistant", "content": "乙"}]
    assert retrieval._format_history(history) == "用户：甲\n助手：乙"


def test_format_history_keeps_only_the_last_n_turns(monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 1)
    history = [
        {"role": "user", "content": "旧1"},
        {"role": "assistant", "content": "旧2"},
        {"role": "user", "content": "新1"},
        {"role": "assistant", "content": "新2"},
    ]
    assert retrieval._format_history(history) == "用户：新1\n助手：新2"


def test_format_history_truncates_long_content_with_ellipsis(monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_CHARS", 5)
    rendered = retrieval._format_history([{"role": "user", "content": "甲" * 50}])
    assert rendered == "用户：" + "甲" * 5 + "…"


def test_format_history_returns_empty_when_turns_disabled(monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 0)
    assert retrieval._format_history([{"role": "user", "content": "甲"}]) == ""


def test_format_history_returns_empty_for_no_history():
    assert retrieval._format_history([]) == ""


# ============================ _history_block ============================


def test_history_block_is_empty_string_without_history():
    """空串是硬要求：模板里 {history_block} 与「参考资料：」同行，换成 "\\n\\n"
    会凭空多出一个空行，单轮问答的 prompt 就不再逐字节一致了。"""
    assert retrieval._history_block([]) == ""
    assert retrieval._history_block(None) == ""


def test_history_block_warns_against_treating_history_as_evidence():
    block = retrieval._history_block([{"role": "user", "content": "甲"}])
    assert "不得把对话历史当作事实来源" in block
    assert "用户：甲" in block


def test_history_block_ends_with_blank_line_to_separate_from_references():
    block = retrieval._history_block([{"role": "user", "content": "甲"}])
    assert block.endswith("\n\n")


def test_single_turn_prompt_has_no_blank_line_before_references():
    """坑 4 的核心断言：空历史渲染出来的 prompt 里，参考资料标题紧跟指令行。"""
    rendered = retrieval.PROMPT_TEMPLATE.format(context="CTX", question="Q", history_block="")
    assert "也不要自行推算页码。\n参考资料：" in rendered
    assert "\n\n参考资料：" not in rendered


def test_multi_turn_prompt_separates_history_from_references():
    block = retrieval._history_block([{"role": "user", "content": "甲"}])
    rendered = retrieval.PROMPT_TEMPLATE.format(context="CTX", question="Q", history_block=block)
    assert "\n\n参考资料：" in rendered


def test_prompt_template_keeps_context_and_question_in_place():
    rendered = retrieval.PROMPT_TEMPLATE.format(context="CTX", question="Q", history_block="")
    assert "CTX\n\n用户问题：Q" in rendered


# ============================ _prepare_prompt ============================


def test_prepare_prompt_returns_empty_on_no_hits(monkeypatch, fake_documents):
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(retrieval.vector_store, "search", lambda q, k=None: [])
    assert retrieval._prepare_prompt("住宿费", k=3) == ("", [])


def test_prepare_prompt_numbers_sources_and_embeds_context(monkeypatch, fake_documents):
    a, b, _c, _d = fake_documents
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(retrieval.vector_store, "search", lambda q, k=None: [(a, 0.9), (b, 0.8)])

    prompt, sources = retrieval._prepare_prompt("住宿费上限是多少", k=2)

    assert "[1] 来源：星尘计划差旅报销管理办法.docx" in prompt
    assert "用户问题：住宿费上限是多少" in prompt
    assert [s["index"] for s in sources] == [1, 2]


def test_prepare_prompt_uses_original_question_even_when_rewriting(monkeypatch, fake_documents):
    """改写只影响检索，拼给模型的必须仍是用户原话，否则回答会答非所问。"""
    a, _b, _c, _d = fake_documents
    seen = {}

    def fake_search(query, k=None):
        seen["query"] = query
        return [(a, 0.9)]

    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(config, "MULTITURN_REWRITE", True)
    monkeypatch.setattr(retrieval.vector_store, "search", fake_search)
    monkeypatch.setattr(retrieval, "_condense_query", lambda q, h: "改写后的独立查询")

    prompt, _sources = retrieval._prepare_prompt(
        "那它怎么算？", k=2, history=[{"role": "user", "content": "住宿费上限"}]
    )

    assert seen["query"] == "改写后的独立查询"
    assert "用户问题：那它怎么算？" in prompt


def test_prepare_prompt_skips_rewriting_without_history(monkeypatch, fake_documents):
    a, _b, _c, _d = fake_documents
    seen = {}
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(config, "MULTITURN_REWRITE", True)
    monkeypatch.setattr(
        retrieval.vector_store,
        "search",
        lambda q, k=None: (seen.setdefault("query", q), [(a, 0.9)])[1],
    )
    monkeypatch.setattr(
        retrieval, "_condense_query", lambda q, h: pytest.fail("无历史时不该调改写")
    )

    retrieval._prepare_prompt("住宿费上限是多少", k=2)
    assert seen["query"] == "住宿费上限是多少"


# ============================ _condense_query ============================


def _fake_rewrite_llm(content=None, exc=None):
    """造一个假的改写模型；exc 非空时调用即抛。"""

    class _Response:
        def __init__(self, text):
            self.content = text

    class _LLM:
        def invoke(self, messages):
            if exc:
                raise exc
            return _Response(content)

    return _LLM()


def test_condense_query_returns_question_verbatim_without_history(monkeypatch):
    """没有历史就没有指代可消解 —— 不该白发一次 LLM 请求。"""
    monkeypatch.setattr(
        retrieval, "get_rewrite_llm", lambda: pytest.fail("无历史时不该调改写模型")
    )
    assert retrieval._condense_query("住宿费上限是多少？", []) == "住宿费上限是多少？"


def test_condense_query_uses_the_rewritten_text(monkeypatch):
    monkeypatch.setattr(retrieval, "get_rewrite_llm", lambda: _fake_rewrite_llm("住宿费超标怎么办"))
    history = [{"role": "user", "content": "住宿费上限"}]

    assert retrieval._condense_query("那它超标呢？", history) == "住宿费超标怎么办"


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_condense_query_falls_back_when_the_model_returns_nothing(monkeypatch, bad):
    monkeypatch.setattr(retrieval, "get_rewrite_llm", lambda: _fake_rewrite_llm(bad))
    assert retrieval._condense_query("那它呢？", [{"role": "user", "content": "甲"}]) == "那它呢？"


def test_condense_query_falls_back_when_the_model_call_fails(monkeypatch):
    """改写失败绝不能让这一轮问答挂掉 —— 这条是整个多轮链路的兜底。"""
    monkeypatch.setattr(
        retrieval, "get_rewrite_llm", lambda: _fake_rewrite_llm(exc=RuntimeError("API 抖动"))
    )
    history = [{"role": "user", "content": "住宿费上限"}]

    assert retrieval._condense_query("那它呢？", history) == "那它呢？"


def test_condense_query_keeps_only_the_first_line_and_strips_quotes(monkeypatch):
    """模型爱输出"改写后的查询"外加解释 —— 只取第一行并去掉引号。"""
    monkeypatch.setattr(
        retrieval, "get_rewrite_llm", lambda: _fake_rewrite_llm('"住宿费超标"\n说明：我补全了指代')
    )
    assert retrieval._condense_query("那它呢？", [{"role": "user", "content": "甲"}]) == "住宿费超标"


# ============================ get_llm / get_rewrite_llm ============================


def test_get_llm_is_cached_and_uses_streaming(monkeypatch):
    """同一个进程内复用实例；streaming=True 是打字机效果的硬前提
    （默认 False 时 .stream() 只吐 1 个 chunk）。"""
    monkeypatch.setattr(retrieval, "_llm", None)
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "test-key")

    first = retrieval.get_llm()
    assert retrieval.get_llm() is first, "应复用同一个实例"
    assert first.streaming is True


def test_get_rewrite_llm_is_a_separate_instance_with_streaming_off(monkeypatch):
    """改写链路必须与主回答链路分开，否则动到 get_llm 会影响流式输出。"""
    monkeypatch.setattr(retrieval, "_llm", None)
    monkeypatch.setattr(retrieval, "_rewrite_llm", None)
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "test-key")

    assert retrieval.get_rewrite_llm() is not retrieval.get_llm()
    assert retrieval.get_rewrite_llm().streaming is False


# ============================ answer_question / stream_answer ============================


def _stub_prompt(monkeypatch, prompt="PROMPT", sources=None):
    monkeypatch.setattr(
        retrieval, "_prepare_prompt", lambda q, k, history=None, strategy=None: (prompt, sources or [])
    )


def test_answer_question_rejects_a_blank_question_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(retrieval, "get_llm", lambda: pytest.fail("空问题不该调模型"))
    for blank in ("", "   ", None):
        assert retrieval.answer_question(blank) == {"answer": "请输入问题。", "sources": []}


def test_answer_question_returns_the_fallback_when_retrieval_finds_nothing(monkeypatch):
    _stub_prompt(monkeypatch, prompt="")
    monkeypatch.setattr(retrieval, "get_llm", lambda: pytest.fail("无检索结果不该调模型"))

    assert retrieval.answer_question("住宿费上限是多少？") == {
        "answer": retrieval.NO_RESULT_ANSWER,
        "sources": [],
    }


def test_answer_question_degrades_gracefully_when_the_llm_call_fails(monkeypatch):
    """模型调用失败要给出可读提示 + 保留来源，而不是把异常抛给界面。"""
    _stub_prompt(monkeypatch, sources=[{"index": 1}])

    class _Boom:
        def invoke(self, messages):
            raise RuntimeError("连接超时")

    monkeypatch.setattr(retrieval, "get_llm", lambda: _Boom())

    result = retrieval.answer_question("住宿费上限是多少？")

    assert "调用大模型失败" in result["answer"]
    assert result["sources"] == [{"index": 1}], "失败时来源仍要给出，便于用户自查"


def test_answer_question_returns_the_model_output_on_success(monkeypatch):
    _stub_prompt(monkeypatch, sources=[{"index": 1}])

    class _Ok:
        def invoke(self, messages):
            return type("R", (), {"content": "每晚 800 元。"})()

    monkeypatch.setattr(retrieval, "get_llm", lambda: _Ok())

    assert retrieval.answer_question("住宿费上限是多少？") == {
        "answer": "每晚 800 元。",
        "sources": [{"index": 1}],
    }


def test_stream_answer_reports_blank_question_as_an_error_event(monkeypatch):
    events = list(retrieval.stream_answer("   "))
    assert events == [{"type": "error", "message": "请输入问题。"}]


def test_stream_answer_reports_index_not_ready_as_an_error_event(monkeypatch):
    """索引缺失是用户最常见的第一次运行状态 —— 必须变成可读提示，不是堆栈。"""
    from app import vector_store

    def raise_not_ready(q, k, history=None, strategy=None):
        raise vector_store.IndexNotReadyError("索引不存在，请先运行 python -m app.ingestion")

    monkeypatch.setattr(retrieval, "_prepare_prompt", raise_not_ready)

    events = list(retrieval.stream_answer("住宿费上限是多少？"))

    assert events[0]["type"] == "error"
    assert "索引不存在" in events[0]["message"]


def test_stream_answer_emits_done_only_when_retrieval_is_empty(monkeypatch):
    """无命中时只发 delta + done，不发 sources 事件（来源列表是空的）。"""
    _stub_prompt(monkeypatch, prompt="")

    events = list(retrieval.stream_answer("住宿费上限是多少？"))

    assert [e["type"] for e in events] == ["delta", "done"]
    assert events[0]["text"] == retrieval.NO_RESULT_ANSWER
    assert events[1]["sources"] == []


def test_stream_answer_accumulates_deltas_into_the_final_answer(monkeypatch):
    _stub_prompt(monkeypatch, sources=[{"index": 1}])

    class _Chunk:
        def __init__(self, text):
            self.content = text

    class _Streaming:
        def stream(self, messages):
            for piece in ["每晚", "", "800", " 元。"]:
                yield _Chunk(piece)

    monkeypatch.setattr(retrieval, "get_llm", lambda: _Streaming())

    events = list(retrieval.stream_answer("住宿费上限是多少？"))

    assert events[0] == {"type": "sources", "sources": [{"index": 1}]}
    assert [e["type"] for e in events[1:]] == ["delta", "delta", "delta", "done"]
    assert events[-1]["answer"] == "每晚800 元。"
    assert events[-1]["answer"] == "".join(
        e["text"] for e in events if e["type"] == "delta"
    ), "done 里的完整回答必须等于所有 delta 的拼接"


def test_stream_answer_reports_a_mid_stream_failure_as_an_error_event(monkeypatch):
    """生成中途断开：已吐出的文本不回滚，但要明确告知中断，不能让前端以为答完了。"""
    _stub_prompt(monkeypatch, sources=[{"index": 1}])

    class _Chunk:
        content = "每晚"

    class _Streaming:
        def stream(self, messages):
            yield _Chunk()
            raise RuntimeError("连接中断")

    monkeypatch.setattr(retrieval, "get_llm", lambda: _Streaming())

    events = list(retrieval.stream_answer("住宿费上限是多少？"))

    assert [e["type"] for e in events] == ["sources", "delta", "error"]
    assert "生成中断" in events[-1]["message"]
