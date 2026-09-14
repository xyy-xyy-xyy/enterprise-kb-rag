"""向量索引层：后端分发、持久化、以及 Qdrant payload 的解析。

这一层最贵 —— 建索引要调 Embedding API、检索要连 Qdrant，所以本文件
**不建真索引、不连真服务**：Qdrant 客户端换成假对象，FAISS 的 vectorstore
换成只带 docstore 的桩。测的是真代码里的分支，不是"假装跑通了"。

值得单独盯住的是 Qdrant 的 payload 形状。它有两处"取错字段就静默失效"的坑：
`scroll` 不显式要 payload → `point.payload` 是 None → SM3 去重和重复入库检查
全部永远返回空集，而系统看起来一切正常。这里对 `scroll` 的调用参数逐条断言。
"""

import os
import sys

import pytest
from langchain_core.documents import Document

from app import config, vector_store

# ============================ 假对象 ============================


class _FakePayloadPoint:
    def __init__(self, payload):
        self.payload = payload


class _FakeCollection:
    def __init__(self, name):
        self.name = name


class _FakeCollections:
    def __init__(self, names):
        self.collections = [_FakeCollection(n) for n in names]


class _FakeQdrantClient:
    """只实现被测代码真正调用的四个方法，并记录调用参数。"""

    def __init__(self, collections=(), points=(), scroll_error=None, collection_error=None):
        self._collections = _FakeCollections(collections)
        self._points = list(points)
        self._scroll_error = scroll_error
        self._collection_error = collection_error
        self.scroll_calls = []
        self.deleted = []

    def get_collections(self):
        if self._collection_error:
            raise self._collection_error
        return self._collections

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        if self._scroll_error:
            raise self._scroll_error
        return self._points, None

    def delete_collection(self, collection_name):
        self.deleted.append(collection_name)


def _use_qdrant(monkeypatch, client, collection="enterprise_kb"):
    monkeypatch.setattr(config, "VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(config, "QDRANT_COLLECTION", collection)
    monkeypatch.setattr(vector_store, "_get_qdrant_client", lambda: client)
    return client


class _FakeIndex:
    def __init__(self, ntotal=0):
        self.ntotal = ntotal


class _FakeDocstore:
    def __init__(self, docs):
        self._dict = {str(i): d for i, d in enumerate(docs)}


class _FakeVectorStore:
    """FAISS 后端的桩：被测代码只用到 docstore / index.ntotal / 几个方法。"""

    def __init__(self, docs=(), ntotal=None, broken_docstore=False):
        if broken_docstore:
            self.docstore = object()  # 没有 _dict，模拟 langchain 内部结构变化
        else:
            self.docstore = _FakeDocstore(docs)
        self.index = _FakeIndex(ntotal if ntotal is not None else len(docs))
        self.added = []
        self.saved_to = None
        self.search_calls = []

    def add_documents(self, docs):
        self.added.extend(docs)

    def save_local(self, path):
        self.saved_to = path

    def similarity_search_with_score(self, query, k):
        self.search_calls.append((query, k))
        return [(Document(page_content="命中", metadata={"file_name": "a.pdf"}), 0.25)]


def _use_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path / "index"))
    return config.INDEX_DIR


# ============================ _collect_hashes ============================


def test_collect_hashes_maps_digest_to_file_name():
    metas = [
        {"sm3_hash": "a" * 64, "file_name": "制度.pdf"},
        {"sm3_hash": "b" * 64, "file_name": "办法.docx"},
    ]
    assert vector_store._collect_hashes(metas) == {"a" * 64: "制度.pdf", "b" * 64: "办法.docx"}


def test_collect_hashes_skips_chunks_without_a_digest():
    """老索引里的分块没有 sm3_hash —— 跳过即可，不能 KeyError 也不能塞个 None 进去。"""
    metas = [{"file_name": "老文档.pdf"}, {"sm3_hash": "", "file_name": "空串.pdf"}]
    assert vector_store._collect_hashes(metas) == {}


def test_collect_hashes_keeps_the_first_file_name_for_a_shared_digest():
    """同一文档的所有分块共享一个 SM3（摘要算的是整份文件），只留一个文件名即可。"""
    metas = [
        {"sm3_hash": "c" * 64, "file_name": "制度.pdf"},
        {"sm3_hash": "c" * 64, "file_name": "制度.pdf"},
        {"sm3_hash": "c" * 64, "file_name": "重名.pdf"},
    ]
    assert vector_store._collect_hashes(metas) == {"c" * 64: "制度.pdf"}


# ============================ 后端分发 ============================


def test_create_index_dispatches_to_faiss_or_qdrant(monkeypatch):
    calls = []
    monkeypatch.setattr(vector_store, "_faiss_create", lambda docs: calls.append("faiss"))
    monkeypatch.setattr(vector_store, "_qdrant_create", lambda docs: calls.append("qdrant"))
    docs = [Document(page_content="x", metadata={"file_name": "a.pdf"})]

    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    vector_store.create_index(docs)
    monkeypatch.setattr(config, "VECTOR_BACKEND", "qdrant")
    vector_store.create_index(docs)

    assert calls == ["faiss", "qdrant"]


def test_load_index_dispatches_to_faiss_or_qdrant(monkeypatch):
    calls = []
    monkeypatch.setattr(vector_store, "_faiss_load", lambda: calls.append("faiss"))
    monkeypatch.setattr(vector_store, "_qdrant_load", lambda: calls.append("qdrant"))

    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    vector_store.load_index()
    monkeypatch.setattr(config, "VECTOR_BACKEND", "qdrant")
    vector_store.load_index()

    assert calls == ["faiss", "qdrant"]


def test_get_indexed_hashes_ignores_the_vectorstore_argument_on_qdrant(monkeypatch):
    """docstring 承诺：`vectorstore` 参数只对 FAISS 生效，Qdrant 下必须忽略它。

    如果哪天有人"顺手"改成用传入的 vectorstore，Qdrant 下就会拿本进程里
    过期的 docstore 去判重，而且不会有任何报错。
    """
    client = _use_qdrant(
        monkeypatch, _FakeQdrantClient(points=[_FakePayloadPoint({"metadata": {"sm3_hash": "d" * 64, "file_name": "q.pdf"}})])
    )
    stray = _FakeVectorStore([Document(page_content="x", metadata={"sm3_hash": "e" * 64, "file_name": "wrong.pdf"})])

    assert vector_store.get_indexed_hashes(stray) == {"d" * 64: "q.pdf"}
    assert len(client.scroll_calls) == 1


def test_get_all_documents_dispatches_by_backend(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vector_store, "_faiss_get_all_documents", lambda: calls.append("faiss"))
    monkeypatch.setattr(vector_store, "_qdrant_get_all_documents", lambda: calls.append("qdrant"))

    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    vector_store.get_all_documents()
    monkeypatch.setattr(config, "VECTOR_BACKEND", "qdrant")
    vector_store.get_all_documents()

    assert calls == ["faiss", "qdrant"]


# ============================ 参数校验与 no-op ============================


def test_create_index_refuses_an_empty_document_list(monkeypatch):
    """空分块建索引会让后端抛出一堆看不懂的底层错误，这里必须先拦住。"""
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    with pytest.raises(ValueError, match="文档分块为空"):
        vector_store.create_index([])


def test_add_documents_is_a_noop_for_an_empty_list():
    """没有新分块时不该白跑一趟后端（不调用 add_documents，也不打印日志）。"""
    store = _FakeVectorStore()

    assert vector_store.add_documents(store, []) is store
    assert store.added == []


def test_add_documents_appends_and_reports_the_new_total(monkeypatch):
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    store = _FakeVectorStore(ntotal=3)
    docs = [Document(page_content="新条款", metadata={"file_name": "a.pdf"})]

    assert vector_store.add_documents(store, docs) is store
    assert len(store.added) == 1


def test_search_rejects_a_blank_query():
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="检索内容不能为空"):
            vector_store.search(bad)


def test_search_passes_the_query_and_k_through(monkeypatch):
    store = _FakeVectorStore()
    monkeypatch.setattr(vector_store, "load_index", lambda: store)

    results = vector_store.search("住宿费上限", k=7)

    assert store.search_calls == [("住宿费上限", 7)]
    assert results[0][0].metadata["file_name"] == "a.pdf"
    assert results[0][1] == pytest.approx(0.25)


def test_search_uses_the_default_k_when_none_is_given(monkeypatch):
    store = _FakeVectorStore()
    monkeypatch.setattr(vector_store, "load_index", lambda: store)

    vector_store.search("住宿费上限")

    assert store.search_calls == [("住宿费上限", 5)]


# ============================ FAISS：存在性 / 保存 / 清空 ============================


def test_faiss_index_exists_requires_both_files(monkeypatch, tmp_path):
    """只判断 .faiss 会在 index.pkl 丢失时报"索引已就绪"，随后加载直接崩。"""
    index_dir = _use_faiss(monkeypatch, tmp_path)
    os.makedirs(index_dir)
    faiss_file = os.path.join(index_dir, "index.faiss")
    pkl_file = os.path.join(index_dir, "index.pkl")

    assert vector_store.index_exists() is False

    open(faiss_file, "w").close()
    assert vector_store.index_exists() is False, "只有 index.faiss 不算就绪"

    open(pkl_file, "w").close()
    assert vector_store.index_exists() is True


def test_faiss_load_raises_a_ready_error_with_the_ingest_hint(monkeypatch, tmp_path):
    """错误信息要直接告诉用户敲哪条命令 —— 这是 IndexNotReadyError 存在的意义。"""
    index_dir = _use_faiss(monkeypatch, tmp_path)

    with pytest.raises(vector_store.IndexNotReadyError) as excinfo:
        vector_store.load_index()

    assert "python -m app.ingestion" in str(excinfo.value)
    assert index_dir in str(excinfo.value)


def test_save_index_faiss_creates_the_directory_and_saves(monkeypatch, tmp_path):
    index_dir = _use_faiss(monkeypatch, tmp_path)
    store = _FakeVectorStore()

    assert vector_store.save_index(store) == index_dir
    assert store.saved_to == index_dir
    assert os.path.isdir(index_dir)


def test_save_index_qdrant_writes_nothing_to_disk(monkeypatch, tmp_path):
    """Qdrant 是实时写入的，不该在本地留下任何索引目录。"""
    index_dir = str(tmp_path / "index")
    _use_qdrant(monkeypatch, _FakeQdrantClient(), collection="my_kb")
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    store = _FakeVectorStore()

    assert vector_store.save_index(store) == "my_kb"
    assert store.saved_to is None, "Qdrant 后端不该调用 save_local"
    assert not os.path.exists(index_dir)


def test_reset_index_faiss_removes_both_files(monkeypatch, tmp_path):
    index_dir = _use_faiss(monkeypatch, tmp_path)
    os.makedirs(index_dir)
    for name in ("index.faiss", "index.pkl"):
        open(os.path.join(index_dir, name), "w").close()

    vector_store.reset_index()

    assert sorted(os.listdir(index_dir)) == []


def test_reset_index_faiss_tolerates_a_missing_directory(monkeypatch, tmp_path):
    """重建流程会先 reset 再 create，索引还不存在时不能报错。"""
    _use_faiss(monkeypatch, tmp_path)
    vector_store.reset_index()  # 不抛异常即通过


def test_reset_index_qdrant_deletes_the_collection(monkeypatch):
    client = _use_qdrant(monkeypatch, _FakeQdrantClient(), collection="my_kb")

    vector_store.reset_index()

    assert client.deleted == ["my_kb"]


# ============================ FAISS：docstore 读取与降级 ============================


def test_faiss_get_sources_reads_from_the_docstore(monkeypatch):
    store = _FakeVectorStore(
        [
            Document(page_content="x", metadata={"source": "a.pdf"}),
            Document(page_content="y", metadata={"source": "b.pdf"}),
            Document(page_content="z", metadata={"source": "a.pdf"}),
            Document(page_content="w", metadata={}),  # 没有 source，丢弃
        ]
    )
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")

    assert vector_store.get_indexed_sources(store) == {"a.pdf", "b.pdf"}


def test_faiss_get_sources_degrades_to_empty_on_a_changed_internal_structure(monkeypatch):
    """langchain 内部结构变了就跳过去重检查，而不是让整次入库崩掉。

    代价是可能重复入库，但比"文档一个都进不去"轻。
    """
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")

    assert vector_store.get_indexed_sources(_FakeVectorStore(broken_docstore=True)) == set()


def test_faiss_get_all_documents_returns_every_chunk(monkeypatch):
    docs = [
        Document(page_content="甲", metadata={"file_name": "a.pdf"}),
        Document(page_content="乙", metadata={"file_name": "b.pdf"}),
    ]
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    monkeypatch.setattr(vector_store, "load_index", lambda: _FakeVectorStore(docs))

    assert [d.page_content for d in vector_store.get_all_documents()] == ["甲", "乙"]


def test_faiss_get_all_documents_degrades_to_no_candidates(monkeypatch):
    """读不到 docstore 时返回空列表 —— BM25 少一路候选，但检索本身还能跑。"""
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    monkeypatch.setattr(
        vector_store, "load_index", lambda: _FakeVectorStore(broken_docstore=True)
    )

    assert vector_store.get_all_documents() == []


def test_faiss_get_hashes_prefers_the_passed_vectorstore(monkeypatch):
    """传了 vectorstore 就不该再去 load_index（那会把整个索引读一遍）。"""
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")

    def explode():
        raise AssertionError("不该调用 load_index")

    monkeypatch.setattr(vector_store, "load_index", explode)
    store = _FakeVectorStore(
        [Document(page_content="x", metadata={"sm3_hash": "f" * 64, "file_name": "a.pdf"})]
    )

    assert vector_store.get_indexed_hashes(store) == {"f" * 64: "a.pdf"}


def test_faiss_get_hashes_degrades_to_empty_on_a_changed_internal_structure(monkeypatch):
    monkeypatch.setattr(config, "VECTOR_BACKEND", "faiss")
    assert vector_store.get_indexed_hashes(_FakeVectorStore(broken_docstore=True)) == {}


# ============================ Qdrant：collection 与加载 ============================


def test_qdrant_collection_exists_matches_the_configured_name(monkeypatch):
    _use_qdrant(monkeypatch, _FakeQdrantClient(collections=["other", "my_kb"]), collection="my_kb")
    assert vector_store.index_exists() is True

    _use_qdrant(monkeypatch, _FakeQdrantClient(collections=["other"]), collection="my_kb")
    assert vector_store.index_exists() is False


def test_qdrant_collection_exists_returns_false_when_the_server_is_down(monkeypatch):
    """连不上 Qdrant 要报"索引未就绪"，而不是抛连接异常把整个服务带崩。"""
    _use_qdrant(monkeypatch, _FakeQdrantClient(collection_error=ConnectionError("拒绝连接")))

    assert vector_store.index_exists() is False


def test_qdrant_load_raises_a_ready_error_naming_the_collection(monkeypatch):
    _use_qdrant(monkeypatch, _FakeQdrantClient(collections=[]), collection="my_kb")

    with pytest.raises(vector_store.IndexNotReadyError) as excinfo:
        vector_store.load_index()

    assert "my_kb" in str(excinfo.value)
    assert "python -m app.ingestion" in str(excinfo.value)


def test_missing_langchain_qdrant_reports_an_actionable_error(monkeypatch):
    """缺依赖时必须报"照做就行"的错，而不是裸的 ModuleNotFoundError。

    阶段五 CI 栽过一次：requirements.txt 漏声明 langchain-qdrant，本机（装了这个包）
    全绿，全新环境里用例却报"索引未就绪"，根因被藏住。这条把提示文字钉死，
    下次真缺依赖时一眼能看出该执行什么命令。

    模拟手法：`sys.modules[name] = None` 会让 `from langchain_qdrant import ...`
    抛 ImportError —— 等同于环境里没装这个包。
    """
    monkeypatch.setitem(sys.modules, "langchain_qdrant", None)

    with pytest.raises(RuntimeError) as excinfo:
        vector_store._qdrant_vector_store_class()

    message = str(excinfo.value)
    assert "langchain-qdrant" in message
    assert "pip install -r requirements.txt" in message


# ============================ Qdrant：payload 解析 ============================


def test_qdrant_get_sources_parses_the_nested_metadata_payload(monkeypatch):
    """LangChain 写入的结构是 {page_content, metadata:{source}}，source 在第二层。"""
    client = _use_qdrant(
        monkeypatch,
        _FakeQdrantClient(
            points=[
                _FakePayloadPoint({"metadata": {"source": "a.pdf"}}),
                _FakePayloadPoint({"metadata": {"source": "b.pdf"}}),
                _FakePayloadPoint({"metadata": {"source": "a.pdf"}}),
                _FakePayloadPoint({"metadata": {}}),  # 无 source
                _FakePayloadPoint(None),  # payload 整个是 None
            ]
        ),
    )

    assert vector_store.get_indexed_sources(None) == {"a.pdf", "b.pdf"}
    assert client.scroll_calls[0]["with_payload"] == ["metadata.source"]
    assert client.scroll_calls[0]["with_vectors"] is False


def test_qdrant_get_sources_returns_empty_set_when_scroll_fails(monkeypatch):
    _use_qdrant(monkeypatch, _FakeQdrantClient(scroll_error=TimeoutError("超时")))

    assert vector_store.get_indexed_sources(None) == set()


def test_qdrant_get_hashes_requests_only_the_two_needed_payload_fields(monkeypatch):
    """必须显式点名这两个字段：不传 with_payload 时 point.payload 是 None，
    去重会永远返回空集 —— 表面上毫无异常，代价是重复文档被反复入库。
    """
    client = _use_qdrant(
        monkeypatch,
        _FakeQdrantClient(
            points=[
                _FakePayloadPoint({"metadata": {"sm3_hash": "1" * 64, "file_name": "a.pdf"}}),
                _FakePayloadPoint({"metadata": {"sm3_hash": "1" * 64, "file_name": "a.pdf"}}),
                _FakePayloadPoint({"metadata": {"file_name": "无摘要.pdf"}}),
                _FakePayloadPoint({}),
                _FakePayloadPoint(None),
            ]
        ),
    )

    assert vector_store.get_indexed_hashes() == {"1" * 64: "a.pdf"}
    assert client.scroll_calls[0]["with_payload"] == [
        "metadata.sm3_hash",
        "metadata.file_name",
    ]
    assert client.scroll_calls[0]["with_vectors"] is False


def test_qdrant_get_hashes_returns_empty_on_a_scroll_failure(monkeypatch):
    _use_qdrant(monkeypatch, _FakeQdrantClient(scroll_error=OSError("网络断了")))

    assert vector_store.get_indexed_hashes() == {}


def test_qdrant_get_all_documents_skips_points_without_content(monkeypatch):
    """payload 里没有 page_content 的点（脏数据/手工写入）不能变成空正文分块，
    否则 BM25 语料里会混进一堆空串，还会被当成"命中"返回。
    """
    _use_qdrant(
        monkeypatch,
        _FakeQdrantClient(
            collections=["my_kb"],
            points=[
                _FakePayloadPoint({"page_content": "甲", "metadata": {"file_name": "a.pdf"}}),
                _FakePayloadPoint({"page_content": "", "metadata": {"file_name": "empty.pdf"}}),
                _FakePayloadPoint({"metadata": {"file_name": "no-content.pdf"}}),
                _FakePayloadPoint({"page_content": "乙"}),  # 无 metadata → 空 dict
            ],
        ),
        collection="my_kb",
    )

    docs = vector_store.get_all_documents()

    assert [d.page_content for d in docs] == ["甲", "乙"]
    assert docs[0].metadata == {"file_name": "a.pdf"}
    assert docs[1].metadata == {}


def test_qdrant_get_all_documents_raises_when_the_collection_is_missing(monkeypatch):
    _use_qdrant(monkeypatch, _FakeQdrantClient(collections=[]), collection="my_kb")

    with pytest.raises(vector_store.IndexNotReadyError, match="my_kb"):
        vector_store.get_all_documents()


def test_qdrant_reset_swallows_a_delete_failure(monkeypatch):
    """重建流程里 reset 失败不该中断 —— collection 本就不存在时最常发生。"""

    class _FailingClient(_FakeQdrantClient):
        def delete_collection(self, collection_name):
            raise RuntimeError("collection 不存在")

    _use_qdrant(monkeypatch, _FailingClient())

    vector_store.reset_index()  # 不抛异常即通过


# ============================ 嵌入模型实例复用 ============================


def test_get_embeddings_reuses_a_single_instance(monkeypatch):
    """每建一次实例都会重读密钥并新建连接池；批处理里反复调用会拖慢整轮入库。"""
    created = []

    class _FakeEmbeddings:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(vector_store, "DashScopeEmbeddings", _FakeEmbeddings)
    monkeypatch.setattr(vector_store, "_embeddings", None)
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "text-embedding-v2")

    first = vector_store.get_embeddings()
    second = vector_store.get_embeddings()

    assert first is second
    assert len(created) == 1
    assert created[0] == {"model": "text-embedding-v2", "dashscope_api_key": "test-key"}
