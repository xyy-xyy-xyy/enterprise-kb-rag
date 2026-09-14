"""接口层：来源渲染与 /api/health。

注意本文件的 import 代价 —— `app/main.py` 在模块作用域执行 `build_ui()`，
所以 `import app.main` 会把整个 Gradio 界面建出来（数秒）。Gradio 的版本查询
已由 conftest 顶部的 GRADIO_ANALYTICS_ENABLED=False 关掉；此处不再重复处理。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import _sources_markdown, app


@pytest.fixture(scope="module")
def client():
    """整个模块共用一个 TestClient —— 每次实例化都会重建 Gradio 界面，太慢。"""
    with TestClient(app) as test_client:
        yield test_client


def _source(file_name="a.pdf", page=1, start=1, end=1, index=1):
    return {
        "index": index,
        "file_name": file_name,
        "page": page,
        "paragraph_start": start,
        "paragraph_end": end,
    }


# ============================ _sources_markdown ============================


def test_sources_markdown_is_empty_without_sources():
    """无命中时界面不该出现一个空的"参考来源"标题。"""
    assert _sources_markdown([]) == ""
    assert _sources_markdown(None) == ""


def test_sources_markdown_prefixes_the_reference_header():
    rendered = _sources_markdown([_source()])
    assert rendered.startswith("\n\n---\n**参考来源**\n")
    assert rendered.endswith("- [1] **a.pdf** 第 1 页，第 1 段")


def test_sources_markdown_keeps_the_index_from_to_sources():
    """编号必须用 _to_sources 给的 index，才能和正文里的 (来源：[2]) 对上。"""
    rendered = _sources_markdown(
        [_source(index=1, start=1, end=1), _source(index=2, start=3, end=4)]
    )
    assert "- [1] **a.pdf** 第 1 页，第 1 段" in rendered
    assert "- [2] **a.pdf** 第 1 页，第 3-4 段" in rendered


def test_sources_markdown_renders_word_as_pageless():
    """Word 的 page 恒为 0，必须显示"无页码"而不是"第 0 页"。"""
    rendered = _sources_markdown([_source(file_name="制度.docx", page=0, start=6, end=10)])
    assert "无页码，第 6-10 段" in rendered
    assert "第 0 页" not in rendered


def test_sources_markdown_collapses_single_paragraph_range():
    rendered = _sources_markdown([_source(page=5, start=5, end=5)])
    assert "第 5 页，第 5 段" in rendered
    assert "5-5" not in rendered


def test_sources_markdown_falls_back_to_page_only_without_paragraphs():
    rendered = _sources_markdown([_source(page=3, start=None, end=None)])
    assert "- [1] **a.pdf** 第 3 页" in rendered


def test_sources_markdown_dedupes_same_file_page_and_paragraph():
    """同一段被两路召回会进 sources 两次，界面上只该出现一行。

    去重 key 带段落号：Word 全是 page=0，只用 (文件, 页) 会把不同段落误判成重复。
    """
    rendered = _sources_markdown([_source(index=1), _source(index=2)])
    assert rendered.count("- [1] **a.pdf** 第 1 页，第 1 段") == 1
    assert "[2]" not in rendered


def test_sources_markdown_keeps_different_paragraphs_of_the_same_word_file():
    """这条正是上面的反例：同一个 docx、page 都是 0，但段落不同 → 两条都要显示。"""
    rendered = _sources_markdown(
        [_source(file_name="制度.docx", page=0, start=1, end=1, index=1),
         _source(file_name="制度.docx", page=0, start=5, end=7, index=2)]
    )
    assert "无页码，第 1 段" in rendered
    assert "无页码，第 5-7 段" in rendered


def test_sources_markdown_keeps_same_page_of_different_files():
    rendered = _sources_markdown(
        [_source(file_name="a.pdf", index=1), _source(file_name="b.pdf", index=2)]
    )
    assert "**a.pdf**" in rendered
    assert "**b.pdf**" in rendered


def test_sources_markdown_omits_index_prefix_when_index_is_absent():
    """旧调用方可能不传 index —— 此时不能渲染出 "[None] "。"""
    rendered = _sources_markdown([_source(index=None)])
    assert "[None]" not in rendered
    assert "- **a.pdf**" in rendered


# ============================ /api/health ============================


def test_health_returns_200_and_all_declared_fields(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert set(response.json()) == {
        "status",
        "llm_model",
        "embedding_model",
        "vector_backend",
        "encrypt_store",
        "sm3_dedup",
        "index_ready",
        "index_path",
    }


def test_health_reports_ok_regardless_of_index_state(client):
    """索引没建好不该让健康检查挂掉 —— 它正是用来报告"没建好"的。

    CI 里 VECTOR_BACKEND=faiss 且没有索引文件，index_ready 必为 false；
    本机连得上 Qdrant，index_ready 为 true。两者都必须返回 200。
    """
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["index_ready"], bool)


def test_health_reports_the_configured_models(client):
    from app import config

    body = client.get("/api/health").json()
    assert body["llm_model"] == config.LLM_MODEL
    assert body["embedding_model"] == config.EMBEDDING_MODEL
    assert body["vector_backend"] == config.VECTOR_BACKEND


def test_health_reports_the_security_switches(client):
    from app import config

    body = client.get("/api/health").json()
    assert body["encrypt_store"] == config.ENCRYPT_STORE
    assert body["sm3_dedup"] == config.SM3_DEDUP


def test_health_is_reachable_without_an_api_key(monkeypatch, client):
    """健康检查是探活用的，不能依赖任何厂商密钥 —— 否则容器起来了也报不健康。"""
    monkeypatch.setattr("app.config.DASHSCOPE_API_KEY", "")
    assert client.get("/api/health").status_code == 200


# ============================ REST 接口 ============================
#
# 这几条测的是对外契约：REST 是公开入口，请求体来自外部，
# 必须容忍畸形数据、必须用对的状态码回话。检索层被替换掉，不发任何真实请求。


def test_api_ask_returns_answer_and_sources(monkeypatch, client):
    from app import retrieval

    monkeypatch.setattr(
        retrieval,
        "answer_question",
        lambda question, k=None, history=None: {
            "answer": f"答案：{question}",
            "sources": [_source(index=1)],
        },
    )

    response = client.post("/api/ask", json={"question": "住宿费上限是多少？"})

    assert response.status_code == 200
    assert response.json() == {"answer": "答案：住宿费上限是多少？", "sources": [_source(index=1)]}


def test_api_ask_passes_question_k_and_history_through(monkeypatch, client):
    """REST 无状态，history 完全靠请求体携带 —— 漏传就等于多轮对话静默失效。"""
    from app import retrieval

    seen = {}

    def fake_answer(question, k=None, history=None):
        seen.update(question=question, k=k, history=history)
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(retrieval, "answer_question", fake_answer)
    history = [{"role": "user", "content": "住宿费上限"}, {"role": "assistant", "content": "800 元"}]

    client.post("/api/ask", json={"question": "那它超标呢？", "k": 3, "history": history})

    assert seen["question"] == "那它超标呢？"
    assert seen["k"] == 3
    assert seen["history"] == history


def test_api_ask_returns_503_when_index_is_not_ready(monkeypatch, client):
    """索引没建好要报 503（服务不可用），不是 500 —— 前端据此提示"请先入库"。"""
    from app import retrieval, vector_store

    def raise_not_ready(question, k=None, history=None):
        raise vector_store.IndexNotReadyError("索引不存在，请先运行 python -m app.ingestion")

    monkeypatch.setattr(retrieval, "answer_question", raise_not_ready)

    response = client.post("/api/ask", json={"question": "住宿费上限是多少？"})

    assert response.status_code == 503
    assert "索引不存在" in response.json()["detail"]


def test_api_ask_rejects_a_body_without_question(client):
    """pydantic 模型缺必填字段 → 422（FastAPI 的默认校验失败码）。"""
    assert client.post("/api/ask", json={}).status_code == 422


def test_api_ask_stream_emits_sse_frames_and_terminator(monkeypatch, client):
    """SSE 的帧格式（data: {...}\n\n）和结束标记 [DONE] 是前端解析的全部依据。"""
    from app import retrieval

    def fake_stream(question, k=None, history=None):
        yield {"type": "sources", "sources": [_source()]}
        yield {"type": "delta", "text": "每晚"}
        yield {"type": "done", "answer": "每晚 800 元。", "sources": [_source()]}

    monkeypatch.setattr(retrieval, "stream_answer", fake_stream)

    response = client.post("/api/ask/stream", json={"question": "住宿费上限是多少？"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert body.endswith("data: [DONE]\n\n"), "结束帧是前端判断流终止的唯一信号"
    assert '"type": "sources"' in body
    # 中文不能被转义成 \uXXXX，否则前端与 curl 观察都不可读
    assert "每晚" in body
    assert "\\u" not in body


def test_api_ask_stream_forwards_history(monkeypatch, client):
    from app import retrieval

    seen = {}

    def fake_stream(question, k=None, history=None):
        seen.update(history=history)
        yield {"type": "done", "answer": "ok", "sources": []}

    monkeypatch.setattr(retrieval, "stream_answer", fake_stream)
    history = [{"role": "user", "content": "甲"}]

    client.post("/api/ask/stream", json={"question": "乙", "history": history})

    assert seen["history"] == history


def test_api_ingest_rejects_unsupported_extension(client, tmp_path):
    response = client.post(
        "/api/ingest",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]
    assert ".docx" in response.json()["detail"]


def test_api_ingest_writes_the_file_and_returns_chunk_count(monkeypatch, client, tmp_path):
    """走一遍上传：落盘 → 加密 → 入库 → 返回分块数。检索层被替换，不发真实请求。"""
    from app import config, ingestion

    monkeypatch.setattr(config, "DOCS_PATH", str(tmp_path))
    monkeypatch.setattr(ingestion, "ingest_file", lambda path: 7)

    response = client.post(
        "/api/ingest",
        files={"file": ("new.docx", b"fake-docx-bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    # 对外返回的是逻辑路径（不带 .enc），要和索引里的 source 一致
    assert response.json() == {
        "status": "success",
        "chunks": 7,
        "file": str(tmp_path / "new.docx"),
    }

    if config.ENCRYPT_STORE:
        # 加密存储的落地保证：磁盘上只有密文，明文必须已被删除
        assert (tmp_path / "new.docx.enc").exists()
        assert not (tmp_path / "new.docx").exists(), "明文不能留在磁盘上"
    else:
        assert (tmp_path / "new.docx").exists()


def test_api_ingest_encrypts_what_it_ingested(monkeypatch, client, tmp_path):
    """入库的必须是加密后的路径，否则索引里存的是明文路径、加密形同虚设。"""
    from app import config, ingestion

    monkeypatch.setattr(config, "DOCS_PATH", str(tmp_path))
    if not config.ENCRYPT_STORE:
        pytest.skip("ENCRYPT_STORE 关闭，跳过加密路径用例。")

    seen = {}
    monkeypatch.setattr(ingestion, "ingest_file", lambda path: seen.setdefault("path", path) or 1)

    client.post("/api/ingest", files={"file": ("a.docx", b"x", "application/octet-stream")})

    assert seen["path"].endswith(".enc")


def test_api_ingest_rolls_back_the_uploaded_file_when_ingestion_fails(
    monkeypatch, client, tmp_path
):
    """入库失败必须回滚 —— 留半个文件在 data/docs/ 会让下次启动反复解析坏文档。"""
    from app import config, ingestion

    monkeypatch.setattr(config, "DOCS_PATH", str(tmp_path))

    def boom(path):
        raise RuntimeError("模拟入库失败")

    monkeypatch.setattr(ingestion, "ingest_file", boom)

    response = client.post(
        "/api/ingest",
        files={"file": ("bad.docx", b"fake", "application/octet-stream")},
    )

    assert response.status_code == 500
    assert "入库失败" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == [], "失败后目录必须是干净的（明文、密文、临时文件都不留）"


def test_api_ingest_preserves_the_previous_file_when_replacing_fails(
    monkeypatch, client, tmp_path
):
    """覆盖已有文件但入库失败时，原来那个文件的**存在性**不能一起被删掉。

    ⚠️ 注意这条只保证"文件还在"，不保证"内容还是旧的"：新上传的内容在
    `shutil.move` 那一步就已经把原文件覆盖了（加密存储下覆盖的是同名 .enc），
    回滚分支又因为 `existed_before=True` 而刻意不删。所以旧版本的内容实际已经丢失，
    而索引里仍是旧内容的向量 —— 磁盘与索引从此不一致。详见交付报告里的缺陷记录。
    """
    from app import config, ingestion

    monkeypatch.setattr(config, "DOCS_PATH", str(tmp_path))
    (tmp_path / "existing.docx").write_bytes(b"original")

    def boom(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(ingestion, "ingest_file", boom)

    response = client.post(
        "/api/ingest",
        files={"file": ("existing.docx", b"replacement", "application/octet-stream")},
    )

    assert response.status_code == 500
    # 文件仍在（无论是明文还是密文形态），目录里没有多余的临时文件
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining in (["existing.docx"], ["existing.docx.enc"]), remaining
