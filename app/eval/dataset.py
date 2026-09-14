"""QA 数据集的加载与校验。

数据集最大的风险是**幻觉**：出题时凭常识写下一个文档里根本没有的数字，
评测跑出来一切正常，结论却是错的。唯一的防线是每条 QA 都带 `evidence`
（从原文逐字抄录的一句话），且 `validate_qa_pairs` 能 100% 验证它真实存在。
"""

import json
import logging
import os

from langchain_core.documents import Document

from app import config

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "data/eval/qa_pairs.json"

REQUIRED_FIELDS = ("id", "question", "expected_answer", "expected_source", "type", "evidence")
VALID_TYPES = ("事实型", "推理型", "对比型")


def load_qa_pairs(path: str = None) -> list[dict]:
    """加载 QA 数据集，返回条目列表。

    path 默认为 config.resolve_path('data/eval/qa_pairs.json')。
    文件不存在或 JSON 非法时抛异常 —— 评测没有数据集就没有意义，不该静默继续。
    """
    path = path or config.resolve_path(DEFAULT_DATASET)
    if not os.path.exists(path):
        raise FileNotFoundError(f"评测数据集不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if not isinstance(pairs, list):
        raise ValueError(f"数据集格式错误：顶层应为数组，实际为 {type(pairs).__name__}")

    logger.info("已加载 %d 条 QA（%s）", len(pairs), path)
    return pairs


def build_source_index(documents: list[Document]) -> dict[str, str]:
    """把分块按 file_name 归并成 {file_name: 全文}，供 evidence 校验用。

    用 file_name 而非 source：source 是归一化后的绝对路径，换个运行目录就变了，
    数据集里不可能写死；file_name 是稳定的逻辑文件名（已剥掉 .enc）。
    """
    index: dict[str, str] = {}
    for doc in documents:
        name = (doc.metadata or {}).get("file_name")
        if not name:
            continue
        index[name] = index.get(name, "") + "\n" + doc.page_content
    return index


def _normalize(text: str) -> str:
    """去掉空白后再比对。

    分块时正文里可能插入换行、缩进（标题增强会加 "[标题] " 前缀），
    逐字比对会被这些空白差异误伤，所以比对前统一压掉所有空白字符。
    """
    return "".join(text.split())


def validate_qa_pairs(pairs: list[dict], documents: list[Document]) -> dict:
    """校验每条 QA 的 evidence 是否真实出现在 expected_source 的分块正文中。

    返回 {"total": int, "passed": int, "failed": [{"id", "question", "reason"}]}。
    这是防止数据集里混进幻觉数据的唯一手段：evidence 是出题时从原文抄来的
    一句话，如果它在对应文档里都找不到，说明这条题是编的。

    同时也校验源文件名是否存在 —— 文件名写错（尤其是三份差旅文档之间写反）
    是另一个静默出错的重灾区。
    """
    source_index = build_source_index(documents)
    known = set(source_index)
    failed: list[dict] = []

    for pair in pairs:
        qid = pair.get("id")
        question = pair.get("question", "")

        missing = [field for field in REQUIRED_FIELDS if not pair.get(field)]
        if missing:
            failed.append(
                {"id": qid, "question": question, "reason": f"缺字段：{' / '.join(missing)}"}
            )
            continue

        if pair["type"] not in VALID_TYPES:
            failed.append(
                {
                    "id": qid,
                    "question": question,
                    "reason": f"type 非法：{pair['type']!r}（应为 {' / '.join(VALID_TYPES)}）",
                }
            )
            continue

        source = pair["expected_source"]
        if source not in known:
            failed.append(
                {
                    "id": qid,
                    "question": question,
                    "reason": f"expected_source 不在索引中：{source!r}"
                    "（注意文件名要与索引里的 file_name 完全一致，且不含 .enc）",
                }
            )
            continue

        if _normalize(pair["evidence"]) not in _normalize(source_index[source]):
            failed.append(
                {
                    "id": qid,
                    "question": question,
                    "reason": f"evidence 在 {source!r} 的正文中找不到（该题可能是编的）："
                    f"{pair['evidence'][:60]}…",
                }
            )
            continue

    return {"total": len(pairs), "passed": len(pairs) - len(failed), "failed": failed}


def summarize_qa_pairs(pairs: list[dict]) -> dict:
    """统计题型分布与每份文档的题目数，供自测清单第 2 条核对覆盖度。"""
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for pair in pairs:
        by_type[pair.get("type", "未知")] = by_type.get(pair.get("type", "未知"), 0) + 1
        source = pair.get("expected_source", "未知")
        by_source[source] = by_source.get(source, 0) + 1
    return {"total": len(pairs), "by_type": by_type, "by_source": by_source}
