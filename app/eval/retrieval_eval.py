"""检索指标评测：Hit Rate@1 / Hit Rate@k / MRR，并按题型分组统计。

命中判定为**文档级**：expected_source 出现在 top-k 结果的 file_name 中就算子命中。
不做分块级判定，是因为同一份文档的相邻分块内容高度重叠，"命中哪个块"区分度很低，
而"有没有捞到对的那份文档"才是这四套检索方案真正的差异所在。
"""

import logging

from langchain_core.documents import Document

from app import retrieval

logger = logging.getLogger(__name__)

# 与路线图 4.2 定义的指标一致
METRIC_KEYS = ("hit_rate@1", "hit_rate@k", "mrr")


def evaluate_retrieval(
    pairs: list[dict], strategy: str, k: int = 5, documents: list[Document] = None
) -> dict:
    """跑一种检索策略，返回检索指标与逐条明细。

    documents 参数保留为可选（当前实现走 retrieval.search_with_strategy，
    内部自会复用已缓存的索引），传了也不影响结果。
    """
    k = k or 5
    details: list[dict] = []
    top1_hits = 0
    topk_hits = 0
    reciprocal_sum = 0.0

    for i, pair in enumerate(pairs, start=1):
        question = pair["question"]
        expected = pair["expected_source"]

        try:
            results = retrieval.search_with_strategy(question, k=k, strategy=strategy)
        except Exception as exc:
            # 单条检索失败不能中断整轮评测，记为未命中并在明细里留痕
            # 用 %s 而不是 %d：id 允许是字符串（如 "q001"），用 %d 会在 except 块
            # 内部再抛 TypeError，反而让"单条失败不中断整轮"的承诺落空
            logger.warning("[%s] 第 %s 条检索失败：%s", strategy, pair.get("id"), exc)
            results = []

        ranked_sources = [
            (doc.metadata or {}).get("file_name") for doc, _score in results
        ]
        rank = 0
        for pos, name in enumerate(ranked_sources, start=1):
            if name == expected:
                rank = pos
                break

        hit = rank > 0
        if hit:
            topk_hits += 1
            reciprocal_sum += 1.0 / rank
            if rank == 1:
                top1_hits += 1
        # 未命中时 rank=0、1/rank 记 0 —— 必须留在分母里。
        # 跳过未命中项会让 MRR 虚高，这是最常见的一种（无意的）评测造假。

        details.append(
            {
                "id": pair.get("id"),
                "question": question,
                "type": pair.get("type"),
                "expected_source": expected,
                "hit": hit,
                "rank": rank,
                "top1_source": ranked_sources[0] if ranked_sources else None,
            }
        )

        if i % 10 == 0 or i == len(pairs):
            logger.info("[%s] 检索评测进度 %d/%d", strategy, i, len(pairs))

    total = len(pairs) or 1
    result = {
        "strategy": strategy,
        "total": len(pairs),
        "k": k,
        "hit_rate@1": top1_hits / total,
        "hit_rate@k": topk_hits / total,
        "mrr": reciprocal_sum / total,
        "by_type": _group_by_type(details),
        "details": details,
    }

    logger.info(
        "[%s] Hit@1=%.3f Hit@%d=%.3f MRR=%.3f",
        strategy,
        result["hit_rate@1"],
        k,
        result["hit_rate@k"],
        result["mrr"],
    )
    return result


def _group_by_type(details: list[dict]) -> dict[str, dict]:
    """按题型分组统计，用来看混合检索在哪种题型上增益最大。

    一个总分说明不了问题，分组之后"对比型提升最明显"这类结论才有依据。
    """
    grouped: dict[str, dict] = {}
    for item in details:
        bucket = grouped.setdefault(
            item.get("type") or "未知", {"total": 0, "hits": 0, "reciprocal_sum": 0.0}
        )
        bucket["total"] += 1
        if item["hit"]:
            bucket["hits"] += 1
            bucket["reciprocal_sum"] += 1.0 / item["rank"]

    return {
        name: {
            "total": bucket["total"],
            "hit_rate@k": bucket["hits"] / bucket["total"] if bucket["total"] else 0.0,
            "mrr": bucket["reciprocal_sum"] / bucket["total"] if bucket["total"] else 0.0,
        }
        for name, bucket in grouped.items()
    }
