"""LLM-as-judge 答案打分：准确性 / 相关性 / 完整性（各 0-5 分）。

裁判模型刻意用 `config.JUDGE_MODEL`（默认 qwen-max）而不是生成答案的
`config.LLM_MODEL`（qwen-plus）：让模型给自己的输出打分存在 self-enhancement
bias，换个模型当裁判更客观。judge 调用失败时退回 LLM_MODEL 重试一次。

LLM 输出的 JSON 极不稳定，本模块的容错比打分本身更值得看 —— 见 `_extract_json`。
"""

import json
import logging
import re

import dashscope

from app import config

logger = logging.getLogger(__name__)

JUDGE_TEMPLATE = """你是企业知识库问答系统的评测裁判。请对照"标准答案"给"待评答案"打分。

【问题】
{question}

【标准答案】
{expected_answer}

【待评答案】
{answer}

评分要求：
1. 先把标准答案拆成若干**关键事实点**（逐条列出）。
2. 逐点核对待评答案：命中记 1，缺失记 0，错误记 -1。
3. 再按三个维度打分，每维 0-5 整数：
   - accuracy（准确性）：有无事实错误，与标准答案冲突即为低分
   - relevance（相关性）：是否回应了问题，有没有跑题
   - completeness（完整性）：严格按下表锚定，**不要凭整体印象估分**
       5 = 标准答案的关键事实点全部覆盖
       4 = 覆盖大部分，只漏了 1 个次要点
       3 = 覆盖约一半
       2 = 只覆盖少部分
       1 = 几乎没覆盖
       0 = 完全没答或答非所问
4. **重要：待评答案末尾的来源引用（形如"（来源：xxx.docx，第 3 页，第 2-4 段）"）是本系统的标准输出，属于加分项。严禁因为答案"额外提供了来源信息"或"比标准答案更长"而扣任何一分。**
5. 只输出如下 JSON，不要任何解释、不要 Markdown 代码块包裹：
{{"key_points": ["要点1", "要点2"], "accuracy": 4, "relevance": 5, "completeness": 3, "reason": "一句话说明扣分原因"}}
"""

SCORE_FIELDS = ("accuracy", "relevance", "completeness")

_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出里尽力抠出 JSON 对象；失败返回 None。

    四道容错，少任何一道都可能让 50 条评测在中途崩掉：
      1. 剥掉 ```json / ``` 代码块包裹（模型极爱加）
      2. 正则截取第一个 { 到最后一个 } —— 前后常有"好的，我来评分："这类寒暄
      3. json.loads 失败不抛，返回 None 由调用方计入 failed
      4. 分数 clamp 到 [0,5] 并强转 int，字段缺失记 0
    """
    if not text:
        return None

    cleaned = _CODE_FENCE.sub("", text.strip())

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("裁判输出中找不到 JSON 对象：%r", text[:120])
        return None

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("裁判输出 JSON 解析失败（%s）：%r", exc, text[:120])
        return None

    if not isinstance(parsed, dict):
        logger.warning("裁判输出的 JSON 不是对象：%r", text[:120])
        return None

    return parsed


def _coerce_score(value) -> int:
    """把分数压到 [0,5] 的整数；缺失或非数字记 0，绝不抛异常。"""
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, score))


def _call_judge(model: str, prompt: str) -> str:
    """调用裁判模型，返回原始文本；失败抛异常由上层接住。"""
    response = dashscope.Generation.call(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=config.DASHSCOPE_API_KEY,
        result_format="message",
        temperature=0.0,  # 评测要可复现，不要采样随机性
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"裁判模型 {model} 返回 {response.status_code}：{getattr(response, 'message', '')}"
        )
    return response.output.choices[0].message.content or ""


def judge_answer(question: str, expected_answer: str, answer: str) -> dict | None:
    """调用 LLM 打分，返回 {"accuracy","relevance","completeness","key_points","reason"}。

    解析失败或调用异常时返回 None，由调用方统计 failed 数 —— 单条失败
    绝不能中断整轮评测（50 条跑到第 40 条崩掉，前面 39 条就白跑了）。
    """
    prompt = JUDGE_TEMPLATE.format(
        question=question, expected_answer=expected_answer, answer=answer or "（空回答）"
    )

    raw = None
    for model in (config.JUDGE_MODEL, config.LLM_MODEL):
        try:
            raw = _call_judge(model, prompt)
            break
        except Exception as exc:
            logger.warning("裁判模型 %s 调用失败：%s", model, exc)
    if raw is None:
        logger.error("裁判模型 %s 与回退模型 %s 均失败，本条记 failed。", config.JUDGE_MODEL, config.LLM_MODEL)
        return None

    parsed = _extract_json(raw)
    if parsed is None:
        return None

    return {
        "accuracy": _coerce_score(parsed.get("accuracy")),
        "relevance": _coerce_score(parsed.get("relevance")),
        "completeness": _coerce_score(parsed.get("completeness")),
        "key_points": parsed.get("key_points") or [],
        "reason": str(parsed.get("reason") or ""),
    }


def average_scores(records: list[dict]) -> dict:
    """对成功打分的记录求三维均分；没有成功记录时返回 None 值而不是 0。

    返回 0 会被误读成"答得很差"，而事实是"没测成"，两者必须区分开。
    """
    scored = [r for r in records if r.get("judge")]
    if not scored:
        return {field: None for field in SCORE_FIELDS}

    return {
        field: sum(r["judge"][field] for r in scored) / len(scored) for field in SCORE_FIELDS
    }
