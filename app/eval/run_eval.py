"""评测入口：四种检索方案 × QA 数据集 → 检索指标 + LLM 打分 → results.json + report.md。

用法：
    python -m app.eval.run_eval --validate        # 只校验数据集
    python -m app.eval.run_eval --limit 5 --no-judge
    python -m app.eval.run_eval                    # 全量

设计要点：所有策略切换都走 `search_with_strategy(strategy=...)`，
**绝不修改全局 config**（config.HYBRID_RETRIEVAL / RERANK_ENABLED 是模块级变量，
改了不恢复会污染后续调用，而且不报任何错 —— 这是本阶段最容易踩的坑）。
"""

import argparse
import datetime
import json
import logging
import os

from app import config, retrieval, vector_store
from app.eval.answer_eval import SCORE_FIELDS, average_scores, judge_answer
from app.eval.dataset import load_qa_pairs, summarize_qa_pairs, validate_qa_pairs
from app.eval.retrieval_eval import METRIC_KEYS, evaluate_retrieval

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/eval/results.json"

STRATEGY_LABELS = {
    "vector": "纯向量检索",
    "bm25": "纯 BM25",
    "hybrid": "混合检索（RRF）",
    "hybrid_rerank": "混合 + Reranker",
}

TYPE_ORDER = ("事实型", "推理型", "对比型")

LIMITATIONS = """\
- **单次采样**：答案由 LLM 生成，本身有随机性；本次每个方案只跑了一轮，未做多次采样取均值。
- **LLM-as-judge 有偏差**：已用不同模型当裁判（`{judge_model}` 评 `{llm_model}` 生成的答案）
  来缓解 self-enhancement bias，但裁判模型自身的偏好并未消除。
- **数据集规模有限**：{dataset_size} 条题目的统计显著性有限，百分点级的差异不宜过度解读。
- **命中判定为文档级**：只判断"有没有捞到对的那份文档"，未精确到分块；
  真实问答中"捞对了文档但捞错了段落"同样会导致答错，本指标看不出来。
- **知识库规模小**：索引共 {chunk_count} 个分块 / {doc_count} 份文档，
  四套方案在这么小的语料上召回高度重叠，结论外推到大规模语料需谨慎。
{completeness_caveat}{hit_at_k_ceiling}"""


def _fmt(value, digits: int = 3) -> str:
    """把指标格式化成固定小数位；None 显示为 "—"（没测成 ≠ 0 分）。"""
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _pct(value) -> str:
    """把 0-1 的比例格式化成百分数。"""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _delta(a, b) -> str:
    """a 相对 b 的百分点差，带正负号。"""
    if a is None or b is None:
        return "—"
    return f"{(a - b) * 100:+.1f}pp"


def run_strategy(
    pairs: list[dict], strategy: str, k: int, documents, judge: bool
) -> dict:
    """跑一种方案：检索指标 → 逐条生成答案 → LLM 打分。"""
    logger.info("=" * 60)
    logger.info("开始评测方案：%s（%s）", strategy, STRATEGY_LABELS.get(strategy, strategy))
    logger.info("=" * 60)

    retrieval_result = evaluate_retrieval(pairs, strategy, k=k, documents=documents)

    records: list[dict] = []
    judge_failed = 0

    for i, pair in enumerate(pairs, start=1):
        try:
            result = retrieval.answer_question(pair["question"], k=k, strategy=strategy)
            answer = result.get("answer", "")
        except Exception as exc:
            logger.warning("[%s] 第 %s 条生成失败：%s", strategy, pair.get("id"), exc)
            answer = ""

        record = {
            "id": pair.get("id"),
            "question": pair["question"],
            "type": pair.get("type"),
            "expected_source": pair["expected_source"],
            "answer": answer,
            "judge": None,
        }

        if judge:
            scored = judge_answer(pair["question"], pair["expected_answer"], answer)
            if scored is None:
                judge_failed += 1
            record["judge"] = scored

        records.append(record)
        if i % 10 == 0 or i == len(pairs):
            logger.info("[%s] 生成+打分进度 %d/%d（打分失败 %d 条）", strategy, i, len(pairs), judge_failed)

    rubric = average_scores(records)
    logger.info(
        "[%s] 三维均分 准确性=%s 相关性=%s 完整性=%s（%d 条打分失败）",
        strategy,
        _fmt(rubric["accuracy"], 2),
        _fmt(rubric["relevance"], 2),
        _fmt(rubric["completeness"], 2),
        judge_failed,
    )

    return {
        "strategy": strategy,
        "label": STRATEGY_LABELS.get(strategy, strategy),
        "retrieval": {key: retrieval_result[key] for key in ("total", "k", *METRIC_KEYS, "by_type")},
        "avg_scores": rubric,
        "judge_failed": judge_failed,
        "records": records,
        "retrieval_details": retrieval_result["details"],
    }


def _rerank_shift_note(results: dict) -> str | None:
    """逐条对比 hybrid 与 hybrid_rerank 的名次变化，给出有据可查的解释。

    返回 None 表示没有值得说的变化（此时不要硬凑一段结论）。
    """
    hy, rr = results.get("hybrid"), results.get("hybrid_rerank")
    if not hy or not rr:
        return None

    hy_d = {d["id"]: d for d in hy["retrieval_details"]}
    rr_d = {d["id"]: d for d in rr["retrieval_details"]}

    # 纯 RRF 下正确文档排第 1、重排后掉出第 1 的题
    shifted = [
        i
        for i in hy_d
        if hy_d[i].get("rank") == 1 and (rr_d.get(i) or {}).get("rank", 0) != 1
    ]
    if not shifted:
        return None

    # 掉到了第几位
    ranks = [rr_d[i]["rank"] for i in shifted if rr_d[i].get("rank")]
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    # 被谁顶掉了：top1 换成了另一份文档
    replaced = sum(
        1
        for i in shifted
        if (rr_d[i].get("top1_source") or "") != (hy_d[i].get("top1_source") or "")
    )

    lines = [
        f"- 值得注意：**{rr['label']}** 的 Hit@1 比 **{hy['label']}** 低 "
        f"{_delta(rr['retrieval']['hit_rate@1'] - hy['retrieval']['hit_rate@1'], 0)}。"
        f"逐条核对发现：{len(shifted)} 道题在纯 RRF 下正确文档排在第 1 位，"
        f"经 Reranker 重排后被**同主题的另一份文档**挤到后面（平均掉到第 {avg_rank:.1f} 位，"
        f"其中 {replaced} 道的第 1 位直接换成了别的文档）。"
    ]
    lines.append(
        "  原因是重排模型按语义相关性打分，在内容高度相似、只有条款数字不同的文档上"
        "（如两份差旅制度都写「一线城市住宿费上限」）区分度不足；"
        "而 RRF 融合保留了 BM25 对文档名与专有名词的字面精确匹配能力。"
    )
    lines.append(
        f"  答案准确性未随之下降，是因为正确文档仍留在 top-{hy['retrieval']['k']} 内、LLM 能从中挑对；"
        "但「首位就是对的」这件事确实变差了，在生产配置里要权衡。"
    )
    return "\n".join(lines)


def build_conclusion(results: dict, strategies: list[str]) -> str:
    """根据实际数据生成结论段落 —— 不写死模板话术，数字变了结论就跟着变。"""
    if "vector" not in results:
        return "（本次未跑纯向量基线，无法给出对比结论。）"

    base = results["vector"]
    lines: list[str] = []

    def metric(name, key):
        return results[name]["retrieval"][key]

    # 逐个方案与纯向量基线比，而不是预设"混合一定更好"
    for name in strategies:
        if name == "vector":
            continue
        line = (
            f"- **{results[name]['label']}** 相比纯向量检索："
            f"Hit@1 {_delta(metric(name, 'hit_rate@1'), metric('vector', 'hit_rate@1'))}、"
            f"Hit@{base['retrieval']['k']} {_delta(metric(name, 'hit_rate@k'), metric('vector', 'hit_rate@k'))}、"
            f"MRR {_delta(metric(name, 'mrr'), metric('vector', 'mrr'))}"
        )
        scored = results[name]["avg_scores"]
        base_scored = base["avg_scores"]
        if scored["accuracy"] is not None and base_scored["accuracy"] is not None:
            line += (
                f"；答案准确性 {scored['accuracy'] - base_scored['accuracy']:+.2f} 分、"
                f"完整性 {scored['completeness'] - base_scored['completeness']:+.2f} 分"
            )
        lines.append(line)

    # 分题型：找出增益最大与最小的题型
    best_name = max(
        (n for n in strategies if n != "vector"),
        key=lambda n: metric(n, "hit_rate@k") - metric("vector", "hit_rate@k"),
        default=None,
    )
    if best_name:
        types = [
            t
            for t in TYPE_ORDER
            if t in base["retrieval"]["by_type"] and t in results[best_name]["retrieval"]["by_type"]
        ]
        if types:
            gains = {
                t: results[best_name]["retrieval"]["by_type"][t]["hit_rate@k"]
                - base["retrieval"]["by_type"][t]["hit_rate@k"]
                for t in types
            }
            top = max(gains, key=gains.get)
            bottom = min(gains, key=gains.get)
            if gains[top] == gains[bottom]:
                # 全平时写"最大 +0.0pp / 最小 +0.0pp"是废话，直接说没差异
                lines.append(
                    f"- 分题型看，各方案在三种题型上的 Hit@{base['retrieval']['k']} 完全一致"
                    f"（均为 {_pct(metric(best_name, 'hit_rate@k'))}），本题集分不出题型间的差异"
                    "（原因见局限性：该指标已触顶）。"
                )
            else:
                lines.append(
                    f"- 分题型看，**{results[best_name]['label']}** 在「{top}」上增益最大"
                    f"（{_delta(gains[top], 0)}），在「{bottom}」上增益最小（{_delta(gains[bottom], 0)}）。"
                )

    total_judge_failed = sum(r["judge_failed"] for r in results.values())
    if total_judge_failed:
        lines.append(f"- 本轮共有 {total_judge_failed} 条答案打分失败（裁判输出不可解析或调用失败），已计入统计口径之外。")

    # Reranker 的 Hit@1 下降必须给出**实测**解释，不能靠猜。
    # 曾经的版本写的是「Reranker 把真正含答案的分块提到了前面，文档级指标看不见」——
    # 逐条核对后发现完全不是这么回事：被顶到第一位的往往是同主题的另一份文档
    # （例如问《差旅费用报销实施细则2025》，重排后 top1 变成《星尘计划差旅报销管理办法》）。
    # 这是重排在相似文档上的真实失误，不是指标口径的盲区，结论必须按实证写。
    note = _rerank_shift_note(results)
    if note:
        lines.append(note)

    # 诚实结论：赢了说赢，没赢说不赢。
    # 先比 Hit@k、打平再比 MRR。语料小的时候四个方案 Hit@k 全平是常态，
    # 只按 Hit@k 排序的话 max() 会在平局时返回列表里靠前的那个 ——
    # 那冠军就是 strategies 的书写顺序决定的，不是实验结论。
    def _rank_key(name: str) -> tuple[float, float]:
        return (metric(name, "hit_rate@k"), metric(name, "mrr"))

    winner = max(strategies, key=_rank_key)
    if _rank_key(winner) == _rank_key("vector"):
        lines.append(
            "\n**结论**：本次实验中，混合检索与 Reranker **没有**优于纯向量检索。"
            "可能的原因：语料规模小（见局限性），向量召回与 BM25 召回的候选高度重叠，"
            "RRF 融合带来的排序变化有限；中文短查询下 Reranker 的区分度也不足以改变 top-k 的成员构成。"
        )
    else:
        k = base["retrieval"]["k"]
        hit_gain = metric(winner, "hit_rate@k") - metric("vector", "hit_rate@k")
        mrr_gain = metric(winner, "mrr") - metric("vector", "mrr")
        if hit_gain > 0:
            lines.append(
                f"\n**结论**：本次实验中 **{results[winner]['label']}** 的检索效果最好，"
                f"Hit@{k} 相对纯向量检索提升 {_delta(hit_gain, 0)}，MRR 提升 {_delta(mrr_gain, 0)}。"
                "提升主要来自关键词召回对专有名词（文档名、制度名、金额条款）的精确匹配能力，"
                "向量检索在中文长句上容易把语义相近但事实不同的条款排到前面。"
            )
        else:
            lines.append(
                f"\n**结论**：本次实验中 **{results[winner]['label']}** 的检索效果最好，"
                f"但 Hit@{k} 与纯向量检索打平（均为 {_pct(metric(winner, 'hit_rate@k'))}），"
                f"优势在排序质量上：Hit@1 "
                f"{_delta(metric(winner, 'hit_rate@1') - metric('vector', 'hit_rate@1'), 0)}、"
                f"MRR {_delta(mrr_gain, 0)}。"
                "也就是说「文档都捞得到，但正确答案被排得更靠前」—— 在 top-k 已经足够大、"
                "命中率触顶的语料上，排序质量才是唯一还能拉开差距的维度。"
            )

    return "\n".join(lines)


def render_report(payload: dict) -> str:
    """生成 report.md。数字全部来自 payload，不写死。"""
    results = payload["strategies"]
    strategies = [s for s in payload["strategy_order"] if s in results]
    k = payload["top_k"]
    meta = payload["meta"]

    out: list[str] = []
    out.append("# RAG 检索方案对比评测报告\n")
    header = f"> 生成时间：{payload['generated_at']}　|　数据集：{payload['dataset_size']} 条 QA"
    if payload.get("rejudged_at"):
        header += f"　|　裁判重打分：{payload['rejudged_at']}"
    out.append(header + "\n")

    out.append("## 一、实验配置\n")
    out.append("| 项 | 值 |")
    out.append("|---|---|")
    out.append(f"| 向量后端 | `{payload['backend']}` |")
    out.append(f"| 索引规模 | {meta['chunk_count']} 个分块 / {meta['doc_count']} 份文档 |")
    out.append(f"| Embedding | `{payload['embedding_model']}` |")
    out.append(f"| 生成模型 | `{payload['llm_model']}` |")
    out.append(f"| 裁判模型 | `{payload['judge_model']}` |")
    out.append(f"| 检索 top-k | {k} |")
    out.append(f"| 数据集规模 | {payload['dataset_size']} 条（事实型 / 推理型 / 对比型） |")
    if payload.get("limit"):
        out.append(f"| 本次限制 | 仅前 {payload['limit']} 条 |")
    out.append("")

    out.append("## 二、四方案主表\n")
    out.append(f"| 方案 | Hit@1 | Hit@{k} | MRR | 准确性 | 相关性 | 完整性 |")
    out.append("|---|---|---|---|---|---|---|")
    for name in strategies:
        item = results[name]
        r = item["retrieval"]
        s = item["avg_scores"]
        out.append(
            f"| {item['label']} | {_pct(r['hit_rate@1'])} | {_pct(r['hit_rate@k'])} | {_fmt(r['mrr'])} "
            f"| {_fmt(s['accuracy'], 2)} | {_fmt(s['relevance'], 2)} | {_fmt(s['completeness'], 2)} |"
        )
    out.append("")
    out.append(
        f"> Hit@1 只看排在第一位的分块，Hit@{k} 看前 {k} 位；两者差距大说明"
        "\"捞到了但排得靠后\"，会直接影响 LLM 读到的上下文顺序。"
        "MRR 把名次折算成 1/rank 取均值，未命中记 0 并计入分母。\n"
    )

    out.append("## 三、分题型 Hit@%d\n" % k)
    header = "| 方案 | " + " | ".join(TYPE_ORDER) + " |"
    out.append(header)
    out.append("|---" * (len(TYPE_ORDER) + 1) + "|")
    for name in strategies:
        by_type = results[name]["retrieval"]["by_type"]
        cells = [
            f"{_pct(by_type[t]['hit_rate@k'])}（{by_type[t]['total']} 题）" if t in by_type else "—"
            for t in TYPE_ORDER
        ]
        out.append(f"| {results[name]['label']} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## 四、结论\n")
    out.append(payload["conclusion"] + "\n")

    out.append("## 五、局限性\n")
    # 四方案 Hit@k 全平时指名道姓地说出来 —— 这个指标已经没有区分度了，
    # 不写的话读者会拿"大家都是 100%"当"大家都一样好"。
    hit_k_values = {results[s]["retrieval"]["hit_rate@k"] for s in strategies}
    hit_at_k_ceiling = ""
    if len(hit_k_values) == 1:
        one = next(iter(hit_k_values))
        hit_at_k_ceiling = (
            f"- **Hit@{k} 已触顶、失去区分度**：四个方案的 Hit@{k} 都是 {_pct(one)}。"
            f"本语料只有 {meta['doc_count']} 份文档，top-{k} 几乎必然覆盖到正确文档，"
            "该指标区分不了方案；实际差异要看 Hit@1 与 MRR。\n"
        )
    # 完整性分若整体偏低，才提示"绝对值不可当真"。
    # 早期版本的裁判模板没有评分锚点，会因为「答案额外提供了来源信息」扣分
    # —— 那是在惩罚系统自带的引用溯源功能，不是答案真的不完整。
    # 模板修好后分数回到正常区间，这段就不应该再出现，所以做成条件输出。
    comps = [
        results[s]["avg_scores"]["completeness"]
        for s in strategies
        if results[s]["avg_scores"]["completeness"] is not None
    ]
    completeness_caveat = ""
    if comps and max(comps) < 3.0:
        completeness_caveat = (
            "- **完整性分的绝对值偏低，别当成「答案不完整」读**："
            f"四个方案的完整性均分都在 {min(comps):.1f}~{max(comps):.1f} 分区间，"
            "裁判很可能在对「答案比标准答案更长」或「附带了来源引用」扣分。"
            "该分数横向可比，但绝对值更多反映裁判的严格程度，而非答案真实完整度。\n"
        )

    out.append(
        LIMITATIONS.format(
            judge_model=payload["judge_model"],
            llm_model=payload["llm_model"],
            dataset_size=payload["dataset_size"],
            chunk_count=meta["chunk_count"],
            doc_count=meta["doc_count"],
            hit_at_k_ceiling=hit_at_k_ceiling,
            completeness_caveat=completeness_caveat,
        )
    )
    return "\n".join(out)


def _collect_meta() -> dict:
    """读取索引规模，写进报告免得过两周看不懂表是怎么来的。"""
    try:
        documents = vector_store.get_all_documents()
    except Exception as exc:
        logger.warning("无法读取索引规模：%s", exc)
        return {"chunk_count": 0, "doc_count": 0}
    names = {(doc.metadata or {}).get("file_name") for doc in documents}
    return {"chunk_count": len(documents), "doc_count": len(names - {None})}


def rejudge_existing(path: str, pairs: list[dict]) -> int:
    """只对已有结果重新打分，不重新检索、不重新生成答案。

    调裁判模板（JUDGE_TEMPLATE）之后必然要重跑一遍分数 —— 但答案本身没变，
    把 304 次生成重跑一遍纯属浪费。这里复用 results.json 里已有的 answer，
    只重跑打分，时间和 API 开销都减半，且新旧分数的差异完全来自模板改动。
    """
    payload = json.load(open(path, encoding="utf-8"))
    expected = {qa["id"]: qa["expected_answer"] for qa in pairs}

    total, failed = 0, 0
    for name, block in payload["strategies"].items():
        records = block.get("records") or []
        for i, record in enumerate(records, start=1):
            scored = judge_answer(
                record["question"], expected.get(record["id"], ""), record.get("answer", "")
            )
            if scored is None:
                failed += 1
            record["judge"] = scored
            total += 1
            if i % 20 == 0 or i == len(records):
                logger.info("[%s] 重打分进度 %d/%d（失败 %d 条）", name, i, len(records), failed)
        block["avg_scores"] = average_scores(records)
        block["judge_failed"] = sum(1 for r in records if not r.get("judge"))
        logger.info(
            "[%s] 重打分后均分 准确性=%s 相关性=%s 完整性=%s",
            name,
            _fmt(block["avg_scores"]["accuracy"], 2),
            _fmt(block["avg_scores"]["relevance"], 2),
            _fmt(block["avg_scores"]["completeness"], 2),
        )

    strategies = [s for s in payload["strategy_order"] if s in payload["strategies"]]
    payload["conclusion"] = build_conclusion(payload["strategies"], strategies)
    payload["judge_model"] = config.JUDGE_MODEL
    payload["rejudged_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    report_path = os.path.join(os.path.dirname(path), "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(payload))

    print(f"\n重打分完成：{total} 条，失败 {failed} 条")
    print(f"结果已更新：{path}")
    print(f"报告已更新：{report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口：python -m app.eval.run_eval。"""
    parser = argparse.ArgumentParser(prog="python -m app.eval.run_eval", description="RAG 评测")
    parser.add_argument("--validate", action="store_true", help="只校验数据集，不跑评测")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（先 5 条验证链路）")
    parser.add_argument("--strategies", default=None, help="逗号分隔，默认全部四种")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 打分（只出检索指标）")
    parser.add_argument("--dataset", default=None, help="数据集路径")
    parser.add_argument("--output", default=None, help="结果输出路径（默认 data/eval/results.json）")
    parser.add_argument(
        "--rejudge",
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        metavar="RESULTS_JSON",
        help="只对已有结果重新打分（改了裁判模板后用，不重新检索/生成答案）",
    )
    args = parser.parse_args(argv)

    config.setup_logging()

    pairs = load_qa_pairs(args.dataset)
    documents = vector_store.get_all_documents()

    # ---- 校验：无论后面跑不跑，都先把数据集验一遍 ----
    report = validate_qa_pairs(pairs, documents)
    print(f"数据集校验：通过率 {report['passed']}/{report['total']}")
    if report["failed"]:
        for item in report["failed"][:20]:
            print(f"  ❌ id={item['id']}　{item['reason']}")
        if len(report["failed"]) > 20:
            print(f"  …… 另有 {len(report['failed']) - 20} 条未显示")
        print("\n数据集未通过校验，评测终止（evidence 校验的意义就在于拦住这种题）。")
        return 1

    summary = summarize_qa_pairs(pairs)
    print(f"题型分布：{summary['by_type']}")
    print("文档覆盖：")
    for name, count in sorted(summary["by_source"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3} 条　{name}")

    if args.validate:
        return 0

    if args.rejudge:
        path = config.resolve_path(args.rejudge)
        if not os.path.exists(path):
            print(f"找不到结果文件：{path}")
            return 1
        return rejudge_existing(path, pairs)

    if args.limit:
        pairs = pairs[: args.limit]
        print(f"\n--limit {args.limit}：本次只跑前 {len(pairs)} 条。")

    strategies = (
        [s.strip() for s in args.strategies.split(",") if s.strip()]
        if args.strategies
        else list(retrieval.STRATEGIES)
    )
    for name in strategies:
        if name not in retrieval.STRATEGIES:
            print(f"未知策略：{name}（可选：{' / '.join(retrieval.STRATEGIES)}）")
            return 1

    k = config.EVAL_TOP_K
    results: dict[str, dict] = {}
    for name in strategies:
        results[name] = run_strategy(pairs, name, k, documents, judge=not args.no_judge)

    payload = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "llm_model": config.LLM_MODEL,
        "judge_model": "（已跳过）" if args.no_judge else config.JUDGE_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "backend": config.VECTOR_BACKEND,
        "top_k": k,
        "dataset_size": len(pairs),
        "limit": args.limit,
        "strategy_order": strategies,
        "meta": _collect_meta(),
        "strategies": results,
    }
    payload["conclusion"] = build_conclusion(results, strategies)

    output = args.output or config.resolve_path(DEFAULT_OUTPUT)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(os.path.dirname(output), "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(payload))

    print(f"\n结果已写入：{output}")
    print(f"报告已写入：{report_path}")
    print("\n主表：")
    report_text = render_report(payload)
    start = report_text.find("## 二")
    end = report_text.find("## 三")
    # 用下标切片而不是 split —— split("## 二")[1] 会把标题本身也切掉，
    # 打印出来只剩「、四方案主表」，看着像坏了。
    print(report_text[start:end if end > start else None].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
