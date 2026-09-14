"""FastAPI 接口 + Gradio 问答界面。"""

import json
import logging
import os
import shutil
import tempfile

import gradio as gr
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import config, ingestion, retrieval, vector_store

config.setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="企业知识库 RAG", version="1.0.0")


class AskRequest(BaseModel):
    question: str
    k: int | None = None
    # 可选的多轮历史 [{"role": "user"|"assistant", "content": "..."}]；
    # REST 无状态，历史由前端携带。不传即为单轮，行为与改动前一致。
    history: list[dict] | None = None


# --------------------------- FastAPI 路由 ---------------------------


@app.get("/api/health")
def health() -> dict:
    """健康检查，同时返回索引状态。"""
    return {
        "status": "ok",
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "vector_backend": config.VECTOR_BACKEND,
        "index_ready": vector_store.index_exists(),
        "index_path": config.INDEX_DIR if config.VECTOR_BACKEND == "faiss" else f"qdrant:{config.QDRANT_COLLECTION}",
    }


@app.post("/api/ingest")
async def api_ingest(file: UploadFile = File(...)) -> dict:
    """上传文档（.pdf / .docx）并入库。"""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ingestion.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 {' / '.join(ingestion.SUPPORTED_EXTENSIONS)} 文件。",
        )

    os.makedirs(config.DOCS_PATH, exist_ok=True)
    target = os.path.join(config.DOCS_PATH, os.path.basename(file.filename))
    existed_before = os.path.exists(target)

    # 先写临时文件，完整接收后再落位，避免半个文件污染知识库目录
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        # 必须先落位再入库：索引里的 source 记录的是最终路径，
        # 否则临时路径会写进索引，既导致重复上传无法去重，也留下失效引用
        shutil.move(tmp_path, target)
        try:
            chunks = ingestion.ingest_file(target)
        except Exception:
            # 入库失败则回滚，不留下未入库的文件
            if not existed_before and os.path.exists(target):
                os.remove(target)
            raise
    except Exception as exc:
        logger.exception("上传入库失败：%s", file.filename)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"入库失败：{exc}") from exc

    return {"status": "success", "chunks": chunks, "file": target}


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """提问并返回答案与来源（一次性返回）。"""
    try:
        return retrieval.answer_question(
            payload.question, k=payload.k, history=payload.history
        )
    except vector_store.IndexNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/ask/stream")
def api_ask_stream(payload: AskRequest):
    """提问并以 SSE 流式返回：每行 data: {...JSON...}，结束为 data: [DONE]。

    事件格式与 retrieval.stream_answer 一致（sources / delta / done / error）。
    用 curl 观察：curl -N -X POST localhost:8000/api/ask/stream \
        -H "Content-Type: application/json" -d '{"question":"..."}'
    """

    def event_stream():
        for event in retrieval.stream_answer(
            payload.question, k=payload.k, history=payload.history
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------- Gradio UI -----------------------------


def _sources_markdown(sources) -> str:
    """把来源渲染成带编号的 Markdown 列表（同文件同页同段只显示一次）。

    编号 [i] 来自 retrieval._to_sources 的 index，与回答正文的 (来源：[i])
    严格对齐。去重 key 加入段落号：Word 没有页码，所有块 page 都是 0，
    若只用 (file_name, page) 去重会把不同段落误判为重复只显示一条。
    """
    if not sources:
        return ""
    seen, lines = set(), []
    for s in sources:
        para_start = s.get("paragraph_start")
        # 去重 key：文件 + 页 + 段落起始；段落缺失（旧索引）时退化为 (文件, 页)
        key = (s.get("file_name"), s.get("page"), para_start)
        if key in seen:
            continue
        seen.add(key)

        page = s.get("page")
        # 位置文本：Word 无页码(page=0)显示"无页码"；PDF 显示"第 N 页"
        if page and page != 0:
            loc = f"第 {page} 页"
        else:
            loc = "无页码"
        para_end = s.get("paragraph_end")
        if para_start is not None and para_end is not None:
            para = (
                f"第 {para_start} 段"
                if para_start == para_end
                else f"第 {para_start}-{para_end} 段"
            )
            loc = f"{loc}，{para}"
        elif para_start is not None:
            loc = f"{loc}，第 {para_start} 段"

        idx = s.get("index")
        prefix = f"[{idx}] " if idx else ""
        lines.append(f"- {prefix}**{s.get('file_name')}** {loc}")
    return "\n\n---\n**参考来源**\n" + "\n".join(lines)


def ui_chat(user_msg, chat_display, history_state):
    """多轮对话回调（生成器，Gradio 会逐次 yield 渲染成打字机效果）。

    chat_display : 展示用（每条助手消息 = 头信息 + 答案 + 来源 Markdown）
    history_state: 喂给 LLM 的干净历史（只有一问一答，不含来源 Markdown）

    传给 stream_answer 的是【本轮之前】的历史，当前问题由 user_msg 单独传入，
    不重复塞进 history。
    """
    chat_display = list(chat_display or [])
    prior_history = list(history_state or [])

    if not user_msg or not user_msg.strip():
        yield chat_display, "", history_state
        return

    # 1) 先把用户消息上屏，并清空输入框
    chat_display = chat_display + [{"role": "user", "content": user_msg}]
    yield chat_display, "", history_state

    # 2) 助手侧先占位"正在检索"
    chat_display = chat_display + [{"role": "assistant", "content": "⏳ 正在检索知识库…"}]
    yield chat_display, "", history_state

    head, sources_md, answer = "⏳ 正在检索知识库…", "", ""
    for event in retrieval.stream_answer(user_msg, history=prior_history):
        etype = event.get("type")

        if etype == "sources":
            sources = event.get("sources") or []
            sources_md = _sources_markdown(sources)
            head = f"✅ 已检索到 {len(sources)} 个参考片段"
            content = f"{head}\n\n{sources_md}".rstrip()

        elif etype == "delta":
            answer += event.get("text", "")
            content = f"{head}\n\n{answer}{sources_md}"

        elif etype == "error":
            message = event.get("message", "处理失败。")
            # 已吐出的答案不回滚，只在后面补一句提示
            content = (
                f"{head}\n\n{answer}\n\n⚠️ {message}{sources_md}"
                if answer
                else f"⚠️ {message}"
            )
            # 重建列表而不是原地改 dict，否则 Gradio 检测不到变化
            chat_display = chat_display[:-1] + [
                {"role": "assistant", "content": content}
            ]
            yield chat_display, "", history_state
            return

        elif etype == "done":
            answer = event.get("answer", answer)
            content = f"{head}\n\n{answer}{sources_md}"

        else:
            continue

        chat_display = chat_display[:-1] + [{"role": "assistant", "content": content}]
        yield chat_display, "", history_state

    # 3) 本轮结束：把干净的一问一答并入历史（answer 不含来源 Markdown）
    history_state = prior_history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    # 按 MAX_HISTORY_TURNS 截断：窗口外的历史不会再进 prompt，留着只会让
    # gr.State 随对话轮数无限增长。MAX_HISTORY_TURNS=0 表示关闭历史，直接清空
    # （注意不能写 [-0:]，Python 会切片出整个列表）。
    limit = config.MAX_HISTORY_TURNS * 2
    history_state = history_state[-limit:] if limit > 0 else []
    yield chat_display, "", history_state


def ui_ingest(files):
    """Gradio 上传回调，支持一次选择多个文档（.pdf / .docx）。"""
    if not files:
        return "请先选择文档（.pdf / .docx）。"
    if isinstance(files, str):
        files = [files]

    os.makedirs(config.DOCS_PATH, exist_ok=True)
    messages, total = [], 0
    for path in files:
        name = os.path.basename(path)
        target = os.path.join(config.DOCS_PATH, name)
        try:
            # 同 API：先落到知识库目录，再入库，保证索引里的 source 是最终路径
            if os.path.abspath(path) != os.path.abspath(target):
                shutil.copy2(path, target)
            chunks = ingestion.ingest_file(target)
            total += chunks
            if chunks:
                messages.append(f"- ✅ {name}：新增 {chunks} 个分块")
            else:
                messages.append(f"- ⏭️ {name}：已入库，跳过")
        except Exception as exc:
            logger.exception("入库失败：%s", name)
            messages.append(f"- ❌ {name}：{exc}")
    return f"**入库完成**（共新增 {total} 个分块）\n" + "\n".join(messages)


def build_ui() -> gr.Blocks:
    # 开启队列：Gradio 只有在队列模式下才会把生成器函数的每次 yield 推给前端，
    # 否则会等生成器跑完再一次性渲染，打字机效果就没了。
    demo = gr.Blocks(title="企业知识库问答")
    demo.queue()
    with demo:
        gr.Markdown(
            "# 企业知识库问答\n"
            "上传 PDF / Word 文档建立知识库，然后基于文档内容提问，回答会自动附带来源与定位"
            "（PDF 显示页码，Word 文档显示段落号）。支持多轮追问，如「那交通费怎么算？」"
            "「它适用于哪些人？」。"
        )
        with gr.Tabs():
            with gr.Tab("知识问答"):
                # Gradio 6 的 Chatbot 消息格式固定为 list[dict]（无 type 参数），
                # 传 type="messages" 会直接 TypeError
                chatbot = gr.Chatbot(label="对话", height=520)
                msg = gr.Textbox(
                    label="你的问题",
                    placeholder="例如：外派人员的住宿费标准是什么？",
                    lines=2,
                )
                with gr.Row():
                    ask_btn = gr.Button("提问", variant="primary")
                    clear_btn = gr.Button("清空")
                # state 存"干净"历史（一问一答，不含来源 Markdown），喂给 LLM 用；
                # chatbot 里那份带来源编号，只用于展示
                state = gr.State([])
                ask_btn.click(
                    ui_chat, inputs=[msg, chatbot, state], outputs=[chatbot, msg, state]
                )
                msg.submit(
                    ui_chat, inputs=[msg, chatbot, state], outputs=[chatbot, msg, state]
                )
                clear_btn.click(
                    lambda: ([], "", []), outputs=[chatbot, msg, state]
                )

            with gr.Tab("文档入库"):
                gr.Markdown(
                    "上传 PDF / Word(.docx) 文档，系统会自动分块、向量化并写入向量索引。"
                )
                uploader = gr.File(
                    label="选择文档（可多选）",
                    file_count="multiple",
                    file_types=[".pdf", ".docx"],
                    type="filepath",
                )
                ingest_btn = gr.Button("开始入库", variant="primary")
                ingest_log = gr.Markdown()
                ingest_btn.click(ui_ingest, inputs=uploader, outputs=ingest_log)
    return demo


# 把 Gradio 挂到 FastAPI 的根路径；/api/* 路由已先行注册，不受影响
# 主题要传给 launch 而非 Blocks 构造函数（Gradio 6 起后者会告警）
app = gr.mount_gradio_app(app, build_ui(), path="/", theme=gr.themes.Soft())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
