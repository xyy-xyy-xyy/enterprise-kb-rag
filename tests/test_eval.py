"""评测层：数据集校验、来源索引、统计、裁判输出的容错解析。

这里没有一次 LLM 调用 —— `_extract_json` / `_coerce_score` / `average_scores`
是纯函数，`validate_qa_pairs` 只吃 Document 列表。

评测体系最危险的失效模式是"数据集本身是编的却一路绿灯"，所以本文件的重点
是**校验器能不能拦住假数据**，而不是校验器能不能放过真数据。
"""

import json

import pytest
from langchain_core.documents import Document

from app import retrieval
from app.eval import answer_eval, dataset, retrieval_eval


@pytest.fixture
def eval_documents() -> list[Document]:
    """两份文档、三个分块 —— 同一文档切多块，用来验证 build_source_index 的归并。"""
    return [
        Document(
            page_content="[标题] 第一章 住宿标准\n一线城市住宿费上限为每晚 800 元。",
            metadata={"file_name": "星尘计划差旅报销管理办法.docx", "chunk_id": "a_p0_0"},
        ),
        Document(
            page_content="住宿费超标部分需由员工自行承担。",
            metadata={"file_name": "星尘计划差旅报销管理办法.docx", "chunk_id": "a_p0_1"},
        ),
        Document(
            page_content="入职满一年可享受带薪年假 5 天。",
            metadata={"file_name": "员工考勤与请假管理办法.docx", "chunk_id": "b_p0_0"},
        ),
    ]


def _pair(**overrides) -> dict:
    pair = {
        "id": "q001",
        "question": "一线城市住宿费上限是多少？",
        "expected_answer": "每晚 800 元。",
        "expected_source": "星尘计划差旅报销管理办法.docx",
        "type": "事实型",
        "evidence": "一线城市住宿费上限为每晚 800 元。",
    }
    pair.update(overrides)
    return pair


# ============================ build_source_index ============================


def test_build_source_index_merges_chunks_of_the_same_file(eval_documents):
    index = dataset.build_source_index(eval_documents)
    assert set(index) == {"星尘计划差旅报销管理办法.docx", "员工考勤与请假管理办法.docx"}
    assert "每晚 800 元" in index["星尘计划差旅报销管理办法.docx"]
    assert "自行承担" in index["星尘计划差旅报销管理办法.docx"]


def test_build_source_index_skips_chunks_without_file_name():
    docs = [Document(page_content="甲", metadata={"source": "C:\\docs\\a.pdf"})]
    assert dataset.build_source_index(docs) == {}


def test_build_source_index_is_empty_for_no_documents():
    assert dataset.build_source_index([]) == {}


# ============================ validate_qa_pairs ============================


def test_validate_passes_a_genuine_pair(eval_documents):
    result = dataset.validate_qa_pairs([_pair()], eval_documents)
    assert result == {"total": 1, "passed": 1, "failed": []}


def test_validate_catches_fabricated_evidence(eval_documents):
    """核心防线：evidence 在对应文档里找不到 → 这条题多半是编的，必须拦下。"""
    fabricated = _pair(evidence="一线城市住宿费上限为每晚 1200 元。")  # 数字被改过
    result = dataset.validate_qa_pairs([fabricated], eval_documents)

    assert result["passed"] == 0
    assert len(result["failed"]) == 1
    assert result["failed"][0]["id"] == "q001"
    assert "找不到" in result["failed"][0]["reason"]


def test_validate_catches_evidence_taken_from_the_wrong_document(eval_documents):
    """最隐蔽的一类错误：evidence 真实存在，但被安到了另一份文档头上。

    三份差旅文档文件名相近，写反了肉眼极难发现 —— 必须靠 evidence 与
    expected_source 的交叉校验兜住。
    """
    mismatched = _pair(expected_source="员工考勤与请假管理办法.docx")
    result = dataset.validate_qa_pairs([mismatched], eval_documents)

    assert result["passed"] == 0
    assert "找不到" in result["failed"][0]["reason"]


def test_validate_reports_every_missing_field(eval_documents):
    result = dataset.validate_qa_pairs([_pair(expected_answer="", type="")], eval_documents)
    assert result["passed"] == 0
    assert "缺字段" in result["failed"][0]["reason"]
    assert "expected_answer" in result["failed"][0]["reason"]
    assert "type" in result["failed"][0]["reason"]


def test_validate_rejects_unknown_question_type(eval_documents):
    result = dataset.validate_qa_pairs([_pair(type="主观型")], eval_documents)
    assert result["passed"] == 0
    assert "type 非法" in result["failed"][0]["reason"]


def test_validate_rejects_unknown_expected_source(eval_documents):
    result = dataset.validate_qa_pairs([_pair(expected_source="不存在的文件.docx")], eval_documents)
    assert result["passed"] == 0
    assert "expected_source 不在索引中" in result["failed"][0]["reason"]


def test_validate_ignores_whitespace_differences_in_evidence(eval_documents):
    """标题增强会给正文插换行，数据集里抄的 evidence 不带这些换行 —— 不该误判。"""
    spaced = _pair(evidence="一线城市住宿费上限为\n每晚 800 元。")
    assert dataset.validate_qa_pairs([spaced], eval_documents)["passed"] == 1


def test_validate_matches_evidence_across_chunk_boundaries(eval_documents):
    """evidence 落在同一文档的另一个分块里也必须算通过（按整篇全文比对）。"""
    other_chunk = _pair(evidence="住宿费超标部分需由员工自行承担。")
    assert dataset.validate_qa_pairs([other_chunk], eval_documents)["passed"] == 1


def test_validate_counts_total_and_passed_over_a_mixed_batch(eval_documents):
    pairs = [
        _pair(id="q001"),
        _pair(id="q002", evidence="编造的一句话。"),
        _pair(id="q003", question="年假多少天？", expected_source="员工考勤与请假管理办法.docx",
              expected_answer="5 天。", evidence="入职满一年可享受带薪年假 5 天。"),
    ]
    result = dataset.validate_qa_pairs(pairs, eval_documents)
    assert result["total"] == 3
    assert result["passed"] == 2
    assert [f["id"] for f in result["failed"]] == ["q002"]


def test_validate_handles_empty_dataset(eval_documents):
    assert dataset.validate_qa_pairs([], eval_documents) == {"total": 0, "passed": 0, "failed": []}


# ============================ summarize_qa_pairs ============================


def test_summarize_counts_by_type_and_by_source():
    summary = dataset.summarize_qa_pairs(
        [_pair(id="q1", type="事实型"), _pair(id="q2", type="推理型"),
         _pair(id="q3", type="推理型", expected_source="另一份.docx")]
    )
    assert summary["total"] == 3
    assert summary["by_type"] == {"事实型": 1, "推理型": 2}
    assert summary["by_source"] == {"星尘计划差旅报销管理办法.docx": 2, "另一份.docx": 1}


def test_summarize_tolerates_missing_fields():
    summary = dataset.summarize_qa_pairs([{"id": "q1"}])
    assert summary["by_type"] == {"未知": 1}
    assert summary["by_source"] == {"未知": 1}


def test_summarize_handles_empty_dataset():
    assert dataset.summarize_qa_pairs([]) == {"total": 0, "by_type": {}, "by_source": {}}


# ============================ _extract_json ============================


def test_extract_json_parses_plain_object():
    assert answer_eval._extract_json('{"accuracy": 4}') == {"accuracy": 4}


def test_extract_json_unwraps_code_fences():
    """模型极爱加 ```json 包裹，这是四道容错里的第一道。"""
    assert answer_eval._extract_json('```json\n{"accuracy": 5}\n```') == {"accuracy": 5}
    assert answer_eval._extract_json('```\n{"accuracy": 5}\n```') == {"accuracy": 5}


def test_extract_json_strips_preamble_and_trailing_chatter():
    """第二道容错：前后常有"好的，我来评分："这类寒暄。"""
    text = '好的，我来评分：\n{"accuracy": 4, "relevance": 5}\n以上是我的评分。'
    assert answer_eval._extract_json(text) == {"accuracy": 4, "relevance": 5}


def test_extract_json_returns_none_on_malformed_json_without_raising():
    """第三道容错：解析失败返回 None 交给调用方计 failed，整轮评测不能因此崩掉。"""
    assert answer_eval._extract_json('{"accuracy": 4,') is None


def test_extract_json_returns_none_when_no_object_present():
    assert answer_eval._extract_json("我无法评分。") is None
    assert answer_eval._extract_json("") is None
    assert answer_eval._extract_json(None) is None


def test_extract_json_rejects_non_object_json():
    """第四道容错：模型偶尔吐一个数组或裸数字，那不是评分结果。"""
    assert answer_eval._extract_json("[1, 2, 3]") is None
    assert answer_eval._extract_json("42") is None


def test_extract_json_keeps_nested_braces_intact():
    """用"第一个 { 到最后一个 }"而不是非贪婪匹配，就是为了不截断嵌套结构。"""
    parsed = answer_eval._extract_json('{"detail": {"reason": "太短"}, "accuracy": 2}')
    assert parsed == {"detail": {"reason": "太短"}, "accuracy": 2}


# ============================ _coerce_score ============================


@pytest.mark.parametrize(
    "value,expected",
    [
        (4, 4),
        ("4", 4),
        (4.7, 4),  # 向零取整，不四舍五入
        ("4.7", 4),
        (5, 5),
        (0, 0),
        (5.9, 5),
        (6, 5),  # 越界 clamp 到 5
        (99, 5),
        (-3, 0),  # 越界 clamp 到 0
        ("5 分", 0),  # 非数字记 0，不抛异常
        ("abc", 0),
        (None, 0),
        ([], 0),
        (True, 1),  # bool 是 int 的特例，按 1 处理而不是崩掉
    ],
)
def test_coerce_score_clamps_and_never_raises(value, expected):
    assert answer_eval._coerce_score(value) == expected


# ============================ average_scores ============================


def test_average_scores_means_each_dimension():
    records = [
        {"judge": {"accuracy": 4, "relevance": 4, "completeness": 4}},
        {"judge": {"accuracy": 2, "relevance": 4, "completeness": 3}},
    ]
    result = answer_eval.average_scores(records)
    assert result == {"accuracy": 3.0, "relevance": 4.0, "completeness": 3.5}


def test_average_scores_skips_failed_records_entirely():
    """judge 为 None 的记录是"没测成"，不能按 0 分计入均分 —— 那会把成绩拉垮。"""
    records = [
        {"judge": {"accuracy": 4, "relevance": 4, "completeness": 4}},
        {"judge": None},
        {"judge": {}},
    ]
    assert answer_eval.average_scores(records) == {
        "accuracy": 4.0,
        "relevance": 4.0,
        "completeness": 4.0,
    }


def test_average_scores_returns_none_when_nothing_was_scored():
    """全失败时返回 None 而不是 0 —— 0 会被误读成"答得很差"。"""
    result = answer_eval.average_scores([{"judge": None}, {}])
    assert result == {"accuracy": None, "relevance": None, "completeness": None}


def test_average_scores_handles_empty_input():
    assert answer_eval.average_scores([]) == {
        "accuracy": None,
        "relevance": None,
        "completeness": None,
    }


def test_average_scores_covers_every_declared_field():
    """SCORE_FIELDS 加了新维度而 average_scores 忘了算，这里会红。"""
    records = [{"judge": {field: 3 for field in answer_eval.SCORE_FIELDS}}]
    assert set(answer_eval.average_scores(records)) == set(answer_eval.SCORE_FIELDS)


# ============================ evaluate_retrieval ============================
#
# 阶段四那张四方案对比表就是这几个数字算出来的。指标算错了，对比表再漂亮
# 也没有意义，所以这部分必须钉死 —— 尤其是"未命中要不要留在分母里"。


def _fake_search(monkeypatch, mapping):
    """把 search_with_strategy 换成 {question: [file_name, ...]} 的查表。"""

    def fake_search(query, k=None, strategy=None):
        return [
            (
                Document(page_content="x", metadata={"file_name": name, "chunk_id": f"{name}_{i}"}),
                1.0 / (retrieval.RRF_K + i + 1),
            )
            for i, name in enumerate(mapping.get(query, []))
        ]

    monkeypatch.setattr(retrieval_eval.retrieval, "search_with_strategy", fake_search)


def _qa(qid, question, source):
    return {
        "id": qid,
        "question": question,
        "expected_answer": "x",
        "expected_source": source,
        "type": "事实型",
        "evidence": "x",
    }


def test_evaluate_retrieval_computes_hit_rates_and_mrr(monkeypatch):
    """三条题：rank1 命中、rank2 命中、未命中。

    Hit@1 = 1/3，Hit@k = 2/3，MRR = (1 + 1/2 + 0) / 3 = 0.5
    """
    pairs = [_qa("q1", "甲", "a.pdf"), _qa("q2", "乙", "b.pdf"), _qa("q3", "丙", "c.pdf")]
    _fake_search(
        monkeypatch,
        {"甲": ["a.pdf", "b.pdf"], "乙": ["a.pdf", "b.pdf"], "丙": ["a.pdf", "b.pdf"]},
    )

    result = retrieval_eval.evaluate_retrieval(pairs, strategy="vector", k=5)

    assert result["hit_rate@1"] == pytest.approx(1 / 3)
    assert result["hit_rate@k"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx(0.5)
    assert result["total"] == 3
    assert result["strategy"] == "vector"
    assert result["k"] == 5


def test_evaluate_retrieval_keeps_misses_in_the_denominator(monkeypatch):
    """核心防作弊断言：未命中的题必须留在分母里。

    只统计命中的题会让 Hit@1 永远是 1.0 —— 这是最常见的一种（往往是无意的）
    评测造假，所以单独钉一条。
    """
    pairs = [_qa("q1", "甲", "a.pdf"), _qa("q2", "乙", "b.pdf")]
    _fake_search(monkeypatch, {"甲": ["a.pdf"], "乙": []})

    result = retrieval_eval.evaluate_retrieval(pairs, strategy="vector", k=5)

    assert result["total"] == 2
    assert result["hit_rate@1"] == pytest.approx(0.5), "命中了 1 条，分母必须是 2 而不是 1"
    assert result["mrr"] == pytest.approx(0.5)


def test_evaluate_retrieval_records_rank_and_top1_source_per_question(monkeypatch):
    pairs = [_qa("q1", "甲", "b.pdf")]
    _fake_search(monkeypatch, {"甲": ["a.pdf", "b.pdf", "c.pdf"]})

    detail = retrieval_eval.evaluate_retrieval(pairs, strategy="hybrid", k=5)["details"][0]

    assert detail == {
        "id": "q1",
        "question": "甲",
        "type": "事实型",
        "expected_source": "b.pdf",
        "hit": True,
        "rank": 2,
        "top1_source": "a.pdf",
    }


def test_evaluate_retrieval_marks_unhit_questions_with_rank_zero(monkeypatch):
    pairs = [_qa("q1", "甲", "z.pdf")]
    _fake_search(monkeypatch, {"甲": ["a.pdf"]})

    detail = retrieval_eval.evaluate_retrieval(pairs, strategy="vector", k=5)["details"][0]

    assert detail["hit"] is False
    assert detail["rank"] == 0
    assert detail["top1_source"] == "a.pdf"


def test_evaluate_retrieval_survives_a_single_question_failing(monkeypatch):
    """单条检索抛异常不能中断整轮评测 —— 否则 76 条 × 4 策略里一次网络抖动就全白跑。

    曾踩过的坑（已修）：日志用 `%d` 绑主键 id
        logger.warning("[%s] 第 %d 条检索失败：%s", strategy, pair.get("id"), exc)
    数据集若用字符串主键（如 "q001"），`%d` 会抛
    `TypeError: %d format: a number is required, not str` —— 异常发生在 except 块
    **内部**，于是"单条失败不中断整轮"的承诺恰好在最需要它的时候落空，整轮评测崩掉。
    当前 `data/eval/qa_pairs.json` 用的是 int 主键，所以线上没触发，属潜伏缺陷。
    已改为 `%s`，本用例用字符串主键守住这个回归。
    """
    _fake_search(monkeypatch, {})

    def exploding_search(query, k=None, strategy=None):
        if query == "坏":
            raise RuntimeError("模拟检索故障")
        return [(Document(page_content="x", metadata={"file_name": "a.pdf"}), 0.5)]

    monkeypatch.setattr(retrieval_eval.retrieval, "search_with_strategy", exploding_search)
    pairs = [_qa("q001", "坏", "a.pdf"), _qa("q002", "好", "a.pdf")]

    result = retrieval_eval.evaluate_retrieval(pairs, strategy="vector", k=5)

    assert result["total"] == 2
    assert [d["hit"] for d in result["details"]] == [False, True]
    assert result["hit_rate@k"] == pytest.approx(0.5)


def test_evaluate_retrieval_groups_metrics_by_question_type(monkeypatch):
    """按题型分组的数字是"事实型是不是明显比推理型好答"这个结论的依据。"""
    pairs = [
        _qa("q1", "甲", "a.pdf"),
        _qa("q2", "乙", "a.pdf"),
    ]
    pairs[1]["type"] = "推理型"
    _fake_search(monkeypatch, {"甲": ["a.pdf"], "乙": []})

    by_type = retrieval_eval.evaluate_retrieval(pairs, strategy="vector", k=5)["by_type"]

    assert by_type["事实型"]["total"] == 1
    assert by_type["事实型"]["hit_rate@k"] == pytest.approx(1.0)
    assert by_type["推理型"]["total"] == 1
    assert by_type["推理型"]["hit_rate@k"] == pytest.approx(0.0)
    assert by_type["推理型"]["mrr"] == pytest.approx(0.0)


def test_evaluate_retrieval_handles_an_empty_dataset(monkeypatch):
    """没有题时返回 0 而不是 ZeroDivisionError（total 用 `len(pairs) or 1` 兜底）。"""
    result = retrieval_eval.evaluate_retrieval([], strategy="vector", k=5)
    assert result["total"] == 0
    assert result["hit_rate@1"] == 0.0
    assert result["mrr"] == 0.0


def test_evaluate_retrieval_declares_every_metric_it_returns(monkeypatch):
    _fake_search(monkeypatch, {})
    result = retrieval_eval.evaluate_retrieval([_qa("q1", "甲", "a.pdf")], strategy="vector")
    assert set(retrieval_eval.METRIC_KEYS) <= set(result)


# ============================ load_qa_pairs ============================


def test_load_qa_pairs_reads_a_json_array(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text('[{"id": 1, "question": "甲"}]', encoding="utf-8")
    assert dataset.load_qa_pairs(str(path)) == [{"id": 1, "question": "甲"}]


def test_load_qa_pairs_keeps_chinese_readable(tmp_path):
    """必须按 UTF-8 读：中文数据集用默认编码打开会崩或变乱码。"""
    path = tmp_path / "qa.json"
    path.write_text('[{"id": 1, "question": "住宿费上限是多少？"}]', encoding="utf-8")
    assert dataset.load_qa_pairs(str(path))[0]["question"] == "住宿费上限是多少？"


def test_load_qa_pairs_raises_when_the_dataset_is_missing(tmp_path):
    """数据集不存在必须报错 —— 评测没有数据集就没有意义，不能静默返回空跑完流程。"""
    with pytest.raises(FileNotFoundError, match="评测数据集不存在"):
        dataset.load_qa_pairs(str(tmp_path / "nope.json"))


def test_load_qa_pairs_raises_when_the_top_level_is_not_an_array(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text('{"id": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="顶层应为数组"):
        dataset.load_qa_pairs(str(path))


def test_load_qa_pairs_raises_on_invalid_json(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        dataset.load_qa_pairs(str(path))


def test_shipped_dataset_loads_and_carries_every_required_field():
    """真实数据集（76 条）必须能被加载且字段齐全 —— 这是评测能不能跑的前提。"""
    import os

    path = os.path.join("data", "eval", "qa_pairs.json")
    if not os.path.exists(path):
        pytest.skip("数据集不在工作区（data/ 可能未被克隆），跳过。")

    pairs = dataset.load_qa_pairs(path)

    assert len(pairs) == 76
    for pair in pairs:
        missing = [f for f in dataset.REQUIRED_FIELDS if not pair.get(f)]
        assert not missing, f"{pair.get('id')} 缺字段 {missing}"
        assert pair["type"] in dataset.VALID_TYPES
