"""FastAPI 接口 + Gradio 问答界面。"""

import logging
import os
import shutil
import tempfile

import gradio as gr
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import config, ingestion, retrieval, vector_store

config.setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="企业知识库 RAG", version="1.0.0")


class AskRequest(BaseModel):
    question: str
    k: int | None = None


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
    """提问并返回答案与来源。"""
    try:
        return retrieval.answer_question(payload.question, k=payload.k)
    except vector_store.IndexNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ----------------------------- Gradio UI -----------------------------


def _format_answer(result: dict) -> str:
    """把回答和来源渲染成 Markdown。"""
    parts = [result.get("answer", "")]
    sources = result.get("sources") or []
    if sources:
        seen, lines = set(), []
        for s in sources:
            key = (s.get("file_name"), s.get("page"))
            if key in seen:
                continue
            seen.add(key)
            # Word 没有页码（page=0），不显示“第 0 页”，只留文件名
            page = s.get("page")
            lines.append(
                f"- **{s.get('file_name')}**"
                + (f" 第 {page} 页" if page else "")
            )
        parts.append("\n\n---\n**参考来源**\n" + "\n".join(lines))
    return "\n".join(parts)


def ui_ask(question: str):
    """Gradio 提问回调。"""
    if not question or not question.strip():
        return "请输入问题。"
    try:
        return _format_answer(retrieval.answer_question(question))
    except vector_store.IndexNotReadyError as exc:
        return f"⚠️ {exc}"
    except Exception as exc:
        logger.exception("问答失败。")
        return f"⚠️ 处理失败：{exc}"


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
    with gr.Blocks(title="企业知识库问答", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 企业知识库问答\n"
            "上传 PDF / Word 文档建立知识库，然后基于文档内容提问，回答会自动附带来源与页码。"
        )
        with gr.Tabs():
            with gr.Tab("知识问答"):
                question = gr.Textbox(
                    label="你的问题",
                    placeholder="例如：公司的报销流程是怎样的？",
                    lines=2,
                )
                with gr.Row():
                    ask_btn = gr.Button("提问", variant="primary")
                    clear_btn = gr.Button("清空")
                answer = gr.Markdown(label="回答")
                ask_btn.click(ui_ask, inputs=question, outputs=answer)
                question.submit(ui_ask, inputs=question, outputs=answer)
                clear_btn.click(lambda: ("", ""), outputs=[question, answer])

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
app = gr.mount_gradio_app(app, build_ui(), path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
