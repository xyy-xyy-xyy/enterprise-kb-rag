"""评测报告的生成逻辑：格式化、结论段落、局限性声明。

`data/eval/report.md` 就是 `render_report()` 的输出 —— 阶段四的交付物本身。
报告里的数字如果算错或写死，结论就不可信，所以这里的断言尽量精确到字符串。

全程纯函数：喂一个手搓的 payload，不碰索引、不调 API。
"""

import pytest

from app.eval import run_eval


# ============================ _fmt / _pct / _delta ============================


def test_fmt_uses_fixed_decimals():
    assert run_eval._fmt(0.5) == "0.500"
    assert run_eval._fmt(0.123456, digits=2) == "0.12"
    assert run_eval._fmt(1.0, digits=1) == "1.0"


def test_fmt_renders_none_as_a_dash_not_zero():
    """没测成 ≠ 0 分。把 None 显示成 0 会把"未评估"误报成"零分"。"""
    assert run_eval._fmt(None) == "—"


def test_pct_formats_a_ratio_as_a_percentage():
    assert run_eval._pct(0.0) == "0.0%"
    assert run_eval._pct(0.7647) == "76.5%"
    assert run_eval._pct(1.0) == "100.0%"
    assert run_eval._pct(None) == "—"


def test_delta_reports_percentage_points_with_a_sign():
    """差值用"百分点"而不是百分比：76.5% 比 71.1% 高的是 5.4pp，不是 7.6%。"""
    assert run_eval._delta(0.7647, 0.7105) == "+5.4pp"
    assert run_eval._delta(0.7105, 0.7647) == "-5.4pp"
    assert run_eval._delta(0.5, 0.5) == "+0.0pp"


def test_delta_is_a_dash_when_either_side_is_unscored():
    assert run_eval._delta(None, 0.5) == "—"
    assert run_eval._delta(0.5, None) == "—"


# ============================ payload 构造 ============================


def _by_type(fact, reasoning, compare):
    return {
        "事实型": {"total": 30, "hit_rate@k": fact, "mrr": fact},
        "推理型": {"total": 30, "hit_rate@k": reasoning, "mrr": reasoning},
        "对比型": {"total": 16, "hit_rate@k": compare, "mrr": compare},
    }


def _strategy(
    label,
    hit1,
    hitk,
    mrr,
    accuracy=4.0,
    completeness=4.0,
    types=None,
    judge_failed=0,
    details=None,
):
    return {
        "label": label,
        "retrieval": {
            "hit_rate@1": hit1,
            "hit_rate@k": hitk,
            "mrr": mrr,
            "k": 5,
            "by_type": types or _by_type(hitk, hitk, hitk),
        },
        "avg_scores": {"accuracy": accuracy, "relevance": 4.0, "completeness": completeness},
        "judge_failed": judge_failed,
        "retrieval_details": details or [],
    }


@pytest.fixture
def payload():
    return {
        "generated_at": "2026-09-14 12:00:00",
        "backend": "qdrant",
        "embedding_model": "text-embedding-v2",
        "llm_model": "qwen-plus",
        "judge_model": "qwen-max",
        "top_k": 5,
        "dataset_size": 76,
        "strategy_order": ["vector", "bm25", "hybrid", "hybrid_rerank"],
        "meta": {"chunk_count": 120, "doc_count": 6},
        "strategies": {
            "vector": _strategy("纯向量检索", 0.7105, 0.9868, 0.8312, 4.2, 3.9),
            "bm25": _strategy("纯 BM25", 0.6579, 0.9737, 0.7901, 4.0, 3.8),
            "hybrid": _strategy("混合检索（RRF）", 0.7632, 0.9868, 0.8604, 4.3, 3.9),
            "hybrid_rerank": _strategy("混合 + Reranker", 0.7895, 1.0, 0.8808, 4.4, 4.0),
        },
        "conclusion": "（结论占位）",
    }


# ============================ render_report ============================


def test_render_report_starts_with_a_title_and_a_metadata_line(payload):
    report = run_eval.render_report(payload)

    assert report.startswith("# RAG 检索方案对比评测报告\n")
    assert "> 生成时间：2026-09-14 12:00:00" in report
    assert "数据集：76 条 QA" in report


def test_render_report_renders_the_experiment_config_table(payload):
    report = run_eval.render_report(payload)

    assert "| 向量后端 | `qdrant` |" in report
    assert "| 索引规模 | 120 个分块 / 6 份文档 |" in report
    assert "| Embedding | `text-embedding-v2` |" in report
    assert "| 生成模型 | `qwen-plus` |" in report
    assert "| 裁判模型 | `qwen-max` |" in report
    assert "| 检索 top-k | 5 |" in report


def test_render_report_emits_one_row_per_strategy_in_the_declared_order(payload):
    report = run_eval.render_report(payload)
    section = report.split("## 三")[0]

    positions = [
        section.index("纯向量检索"),
        section.index("纯 BM25"),
        section.index("混合检索（RRF）"),
        section.index("混合 + Reranker"),
    ]
    assert positions == sorted(positions), "方案顺序必须与 strategy_order 一致"


def test_render_report_skips_strategies_absent_from_the_results(payload):
    """跑失败或被 --strategies 限定的方案不该在报告里留一行空白。"""
    payload["strategies"].pop("bm25")
    report = run_eval.render_report(payload)

    assert "纯 BM25" not in report
    assert "纯向量检索" in report


def test_render_report_breaks_hit_rate_down_by_question_type(payload):
    payload["strategies"]["hybrid_rerank"]["retrieval"]["by_type"] = _by_type(0.9, 0.6, 0.5)
    report = run_eval.render_report(payload)

    assert "## 三、分题型 Hit@5" in report
    assert "90.0%（30 题）" in report
    assert "60.0%（30 题）" in report
    assert "50.0%（16 题）" in report


def test_render_report_uses_a_dash_for_a_type_with_no_questions(payload):
    payload["strategies"]["vector"]["retrieval"]["by_type"].pop("对比型")
    report = run_eval.render_report(payload)

    assert "| 纯向量检索 | 98.7%（30 题） | 98.7%（30 题） | — |" in report


def test_render_report_flags_hit_at_k_ceiling_when_all_strategies_tie(payload):
    """四方案 Hit@k 打平时必须说穿：这个指标已经失去区分度了。

    不写的话读者会把"大家都是 100%"读成"四套方案一样好"，而真实差异
    在 Hit@1 与 MRR 上。
    """
    for name in payload["strategies"]:
        payload["strategies"][name]["retrieval"]["hit_rate@k"] = 1.0

    report = run_eval.render_report(payload)

    assert "Hit@5 已触顶、失去区分度" in report
    assert "四个方案的 Hit@5 都是 100.0%" in report
    assert "实际差异要看 Hit@1 与 MRR" in report


def test_render_report_omits_the_ceiling_note_when_hit_at_k_differs(payload):
    report = run_eval.render_report(payload)
    assert "已触顶、失去区分度" not in report


def test_render_report_only_warns_about_low_completeness_when_it_is_actually_low(payload):
    """完整性均分正常时不该出现"绝对值偏低"的免责声明 —— 那是裁判模板有 bug 时的补丁。"""
    report = run_eval.render_report(payload)
    assert "完整性分的绝对值偏低" not in report


def test_render_report_warns_when_every_strategy_scores_low_on_completeness(payload):
    for name in payload["strategies"]:
        payload["strategies"][name]["avg_scores"]["completeness"] = 2.5

    report = run_eval.render_report(payload)

    assert "完整性分的绝对值偏低" in report
    assert "2.5~2.5 分区间" in report


def test_render_report_ignores_unscored_completeness_when_deciding_to_warn(payload):
    """全部没打分（None）时不该误报"分数偏低"。"""
    for name in payload["strategies"]:
        payload["strategies"][name]["avg_scores"]["completeness"] = None

    report = run_eval.render_report(payload)
    assert "完整性分的绝对值偏低" not in report


def test_render_report_always_states_the_limitations(payload):
    """局限性一节是这个交付物的诚实性所在，不能因为数字好看就被省掉。"""
    report = run_eval.render_report(payload)

    assert "## 五、局限性" in report
    assert "单次采样" in report
    assert "LLM-as-judge 有偏差" in report
    assert "命中判定为文档级" in report
    assert "数据集规模有限" in report
    assert "知识库规模小" in report


def test_render_report_mentions_the_judge_model_in_the_limitations(payload):
    report = run_eval.render_report(payload)
    assert "`qwen-max` 评 `qwen-plus` 生成的答案" in report


def test_render_report_includes_the_conclusion_verbatim(payload):
    payload["conclusion"] = "混合检索在 Hit@1 上比纯向量高 5.3pp。"
    report = run_eval.render_report(payload)
    assert "## 四、结论\n\n混合检索在 Hit@1 上比纯向量高 5.3pp。" in report


def test_render_report_notes_a_rejudge_timestamp_when_present(payload):
    payload["rejudged_at"] = "2026-09-14 18:00:00"
    assert "裁判重打分：2026-09-14 18:00:00" in run_eval.render_report(payload)


def test_render_report_does_not_hardcode_numbers(payload):
    """换一组数字，报告里的数字必须跟着变 —— 防的是模板里写死。"""
    for name in payload["strategies"]:
        payload["strategies"][name]["retrieval"]["hit_rate@1"] = 0.1234

    report = run_eval.render_report(payload)

    assert "12.3%" in report
    assert "71.1%" not in report, "旧的向量 Hit@1 不该还留在报告里"


# ============================ build_conclusion ============================


def test_build_conclusion_compares_each_strategy_against_the_vector_baseline(payload):
    conclusion = run_eval.build_conclusion(
        payload["strategies"], payload["strategy_order"]
    )

    assert "**混合检索（RRF）** 相比纯向量检索" in conclusion
    assert "**混合 + Reranker** 相比纯向量检索" in conclusion
    # 0.7895 - 0.7105 = +7.9pp
    assert "Hit@1 +7.9pp" in conclusion


def test_build_conclusion_does_not_claim_hybrid_always_wins(payload):
    """结论必须由数字生成：这里让混合方案全面变差，结论就不能还说"提升"。"""
    baseline = payload["strategies"]["vector"]
    payload["strategies"]["hybrid"] = _strategy("混合检索（RRF）", 0.5, 0.5, 0.5, 3.0, 3.0)
    payload["strategies"]["hybrid_rerank"] = _strategy("混合 + Reranker", 0.5, 0.5, 0.5, 3.0, 3.0)
    payload["strategies"]["vector"] = baseline

    conclusion = run_eval.build_conclusion(
        payload["strategies"], payload["strategy_order"]
    )

    assert "-21.1pp" in conclusion
    assert "+" not in conclusion.split("**混合检索（RRF）**")[1].split("\n")[0]


def test_build_conclusion_says_so_when_the_baseline_is_missing():
    """没跑纯向量基线时必须明说，而不是随便找个参照物硬比。"""
    results = {"hybrid": _strategy("混合检索（RRF）", 0.7, 0.9, 0.8)}
    conclusion = run_eval.build_conclusion(results, ["hybrid"])
    assert "未跑纯向量基线" in conclusion


def test_build_conclusion_names_the_best_and_worst_question_type(payload):
    payload["strategies"]["vector"]["retrieval"]["by_type"] = _by_type(0.5, 0.9, 0.7)
    payload["strategies"]["hybrid_rerank"]["retrieval"]["by_type"] = _by_type(0.9, 0.9, 0.7)

    conclusion = run_eval.build_conclusion(
        payload["strategies"], payload["strategy_order"]
    )

    assert "事实型" in conclusion
    # 事实型 +40pp 增益最大；推理型与对比型增益为 0，最小的是它们之一
    assert "40.0pp" in conclusion or "+40.0pp" in conclusion


def test_build_conclusion_discloses_failed_judgements(payload):
    """有答案没打上分时必须报出来 —— 否则读者以为均分是全部 76 条的结果。"""
    payload["strategies"]["hybrid"]["judge_failed"] = 3
    payload["strategies"]["bm25"]["judge_failed"] = 1

    conclusion = run_eval.build_conclusion(
        payload["strategies"], payload["strategy_order"]
    )

    assert "本轮共有 4 条答案打分失败" in conclusion


def test_build_conclusion_stays_quiet_when_every_answer_was_scored(payload):
    conclusion = run_eval.build_conclusion(
        payload["strategies"], payload["strategy_order"]
    )
    assert "打分失败" not in conclusion


def test_build_conclusion_skips_score_comparison_when_unscored(payload):
    """没打分（None）时只比检索指标，不能拿 None 做减法崩掉。"""
    for name in payload["strategies"]:
        payload["strategies"][name]["avg_scores"] = {
            "accuracy": None,
            "relevance": None,
            "completeness": None,
        }

    conclusion = run_eval.build_conclusion(
        payload["strategies"], payload["strategy_order"]
    )

    assert "答案准确性" not in conclusion
    assert "Hit@1" in conclusion


# ============================ _rerank_shift_note ============================


def _detail(qid, rank, top1):
    return {"id": qid, "rank": rank, "top1_source": top1}


def test_rerank_shift_note_is_none_without_both_hybrid_strategies():
    """只有 hybrid 没有 hybrid_rerank（或反之）时无从比较，不能硬凑一句话出来。"""
    assert run_eval._rerank_shift_note({}) is None
    assert run_eval._rerank_shift_note({"hybrid": _strategy("混合", 0.7, 0.9, 0.8)}) is None


def test_rerank_shift_note_is_none_when_rerank_did_not_move_anything(payload):
    """重排没有把任何一道题的第一位顶掉时，报告里不该多出一段解释。"""
    details = [_detail(1, 1, "a.pdf"), _detail(2, 2, "b.pdf")]
    payload["strategies"]["hybrid"] = _strategy("混合检索（RRF）", 0.79, 0.99, 0.88, details=details)
    payload["strategies"]["hybrid_rerank"] = _strategy(
        "混合 + Reranker", 0.79, 0.99, 0.88, details=details
    )

    assert run_eval._rerank_shift_note(payload["strategies"]) is None


def test_rerank_shift_note_explains_the_hit_at_1_drop_with_real_counts(payload):
    """Reranker 让 Hit@1 掉了，必须给出**实测**的题数和名次，而不是"指标看不见"这种猜测。"""
    payload["strategies"]["hybrid"] = _strategy(
        "混合检索（RRF）",
        0.80,
        1.0,
        0.90,
        details=[
            _detail(1, 1, "差旅费用报销实施细则2025.pdf"),
            _detail(2, 1, "差旅费用报销实施细则2025.pdf"),
            _detail(3, 1, "住宿标准.pdf"),
            _detail(4, 3, "住宿标准.pdf"),  # 本来就没排第一，不算被顶掉
        ],
    )
    payload["strategies"]["hybrid_rerank"] = _strategy(
        "混合 + Reranker",
        0.77,
        1.0,
        0.88,
        details=[
            _detail(1, 3, "星尘计划差旅报销管理办法.pdf"),  # 换了另一份文档
            _detail(2, 2, "差旅费用报销实施细则2025.pdf"),  # 同一份文档内下沉
            _detail(3, 1, "住宿标准.pdf"),
            _detail(4, 2, "住宿标准.pdf"),
        ],
    )

    note = run_eval._rerank_shift_note(payload["strategies"])

    assert "Hit@1 比 **混合检索（RRF）** 低 -3.0pp" in note
    assert "2 道题在纯 RRF 下正确文档排在第 1 位" in note, "第 4 题本来就不是第 1，不该算进来"
    assert "平均掉到第 2.5 位" in note
    assert "其中 1 道的第 1 位直接换成了别的文档" in note, "只有第 1 题换了文档，第 2 题是同一份"
    assert "同主题的另一份文档" in note


def test_rerank_shift_note_does_not_blame_the_metric(payload):
    """曾经的版本把 Hit@1 下降归因于"文档级指标看不见"，那与实测不符（是重排失误）。

    这段解释必须指向重排本身，不能用来把下降说成指标口径问题。
    """
    payload["strategies"]["hybrid"] = _strategy(
        "混合检索（RRF）", 0.80, 1.0, 0.90, details=[_detail(1, 1, "a.pdf")]
    )
    payload["strategies"]["hybrid_rerank"] = _strategy(
        "混合 + Reranker", 0.77, 1.0, 0.88, details=[_detail(1, 2, "b.pdf")]
    )

    note = run_eval._rerank_shift_note(payload["strategies"])

    assert "指标口径的盲区" not in note
    assert "区分度不足" in note


def test_render_report_includes_the_rerank_note_when_present(payload):
    """结论段落经 build_conclusion 落到 report.md 里，中间不能丢。"""
    payload["strategies"]["hybrid"] = _strategy(
        "混合检索（RRF）", 0.80, 1.0, 0.90, details=[_detail(1, 1, "a.pdf")]
    )
    payload["strategies"]["hybrid_rerank"] = _strategy(
        "混合 + Reranker", 0.77, 1.0, 0.88, details=[_detail(1, 2, "b.pdf")]
    )
    payload["conclusion"] = run_eval.build_conclusion(
        payload["strategies"], payload["strategy_order"]
    )

    assert "值得注意" in run_eval.render_report(payload)
