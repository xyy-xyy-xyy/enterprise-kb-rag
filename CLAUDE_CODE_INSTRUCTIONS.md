# Claude Code 指令：企业知识库 RAG 系统

## 项目概述

构建一个基于 RAG（检索增强生成）的企业知识库问答系统。用户上传 PDF 文档，系统将其分块、向量化并存储到 FAISS，用户提问时检索相关片段，调用大模型生成带引用来源的回答。

## 技术栈

- **向量数据库**：FAISS（本地文件型，不需要 Docker）
- **LLM**：通义千问 qwen-plus（通过阿里云 DashScope API 调用）
- **Embedding**：DashScope text-embedding-v2
- **文档解析**：PyMuPDF（fitz）
- **框架**：LangChain 1.4.0 + langchain-community 0.4.2
- **API**：FastAPI
- **Web UI**：Gradio（简易问答界面）
- **Python**：3.13，虚拟环境在 `venv/`

## 已就绪的环境

所有依赖已安装在 `venv/` 中，包括：
- `faiss-cpu==1.15.0`
- `langchain==1.4.0` / `langchain-community==0.4.2`
- `langchain-text-splitters==1.1.2`
- `dashscope==1.27.4`
- `fastapi==0.141.1` / `gradio==6.26.0`
- `pymupdf==1.28.2`

配置文件 `.env` 已创建：
```
DASHSCOPE_API_KEY=sk-ws-H...（已填好真实 key）
FAISS_INDEX_PATH=data/faiss_index
LLM_MODEL=qwen-plus
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## 已验证的导入路径（直接用，不会报错）

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss  # faiss-cpu 1.15.0
```

## 目标项目结构

```
enterprise-kb-rag/
├── app/
│   ├── __init__.py
│   ├── config.py          # 读取 .env，集中管理配置
│   ├── vector_store.py    # FAISS 索引的创建/保存/加载/检索
│   ├── ingestion.py       # PDF → 分块 → 向量化 → 存 FAISS
│   ├── retrieval.py       # 检索 → 拼 prompt → 调 LLM → 返回答案+来源
│   └── main.py            # FastAPI 入口 + Gradio UI
├── data/
│   ├── docs/              # 放知识库 PDF 文档（用户手动放）
│   └── faiss_index/       # FAISS 索引文件（自动生成）
├── .env                   # 已就绪
├── .gitignore             # 已就绪
├── requirements.txt       # 已就绪
└── CLAUDE_CODE_INSTRUCTIONS.md  # 本文件
```

## 各模块详细规格

### 1. app/config.py

集中读取 .env 配置，供其他模块导入。

```python
from dotenv import load_dotenv
import os

load_dotenv()  # 读取项目根目录的 .env

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBEDDING_MODEL = "text-embedding-v2"  # DashScope 默认嵌入模型
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DOCS_DIR = "data/docs"
```

### 2. app/vector_store.py

封装 FAISS 的三个核心操作：建索引、存索引、加载索引。

关键函数：
- `get_embeddings()` → 返回 `DashScopeEmbeddings(model=EMBEDDING_MODEL, dashscope_api_key=DASHSCOPE_API_KEY)`
- `create_index(documents)` → 用 `FAISS.from_documents(documents, embeddings)` 创建索引，返回 vectorstore
- `save_index(vectorstore)` → 调用 `vectorstore.save_local(FAISS_INDEX_PATH)`
- `load_index()` → 调用 `FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)`，返回 vectorstore
- `search(query, k=5)` → 加载索引后调用 `vectorstore.similarity_search_with_score(query, k=k)`，返回 `[(Document, score)]`

实现注意：
- `allow_dangerous_deserialization=True` 是 langchain 安全机制要求的参数，FAISS 加载本地索引必须加
- FAISS 的 `save_local` 会在指定目录下生成 `index.faiss` 和 `index.pkl` 两个文件

### 3. app/ingestion.py

PDF 文档入库流程：加载 → 分块 → 向量化 → 存 FAISS。

关键函数：
- `ingest_pdf(file_path)` → 对单个 PDF 执行完整入库流程
- `ingest_directory(dir_path)` → 遍历 `data/docs/` 下所有 .pdf 文件批量入库

流程细节：
1. `PyMuPDFLoader(file_path).load()` 加载 PDF，得到 `List[Document]`（每个 Document 是一页）
2. `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)` 分块
3. 给每个分块添加 metadata：`{"source": file_path, "page": page_number}`
4. 调用 `vector_store.create_index(chunks)` 创建索引
5. 如果已有索引存在，用 `vectorstore.add_documents(new_chunks)` 追加；否则新建
6. 调用 `vector_store.save_index(vectorstore)` 持久化

提供命令行入口：
```python
if __name__ == "__main__":
    # 扫描 data/docs/ 下所有 PDF，建索引
    ingest_directory("data/docs")
    print("文档入库完成")
```

### 4. app/retrieval.py

问答检索流程：向量检索 → 拼上下文 → 调 LLM → 返回答案+来源。

关键函数：
- `get_llm()` → 返回 `ChatTongyi(model=LLM_MODEL, dashscope_api_key=DASHSCOPE_API_KEY)`
- `answer_question(question, k=5)` → 返回 `{"answer": str, "sources": List[dict]}`

流程细节：
1. `vector_store.search(question, k=5)` 检索 top-5 相关片段
2. 把片段文本拼成 context 字符串
3. 构造 prompt：

```
你是一个企业知识库助手。请根据以下参考资料回答用户问题。
如果资料中没有相关信息，请如实说明"知识库中未找到相关内容"。
回答时请引用来源文档和页码。

参考资料：
[1] 来源：{source}, 第{page}页
内容：{chunk_text}
...

用户问题：{question}
```

4. `llm.invoke([HumanMessage(content=prompt)])` 调用大模型
5. 返回 `{"answer": response.content, "sources": [{"source": doc.metadata["source"], "page": doc.metadata["page"], "snippet": doc.page_content[:100]} for doc, score in results]}`

### 5. app/main.py

FastAPI API + Gradio 界面，两种访问方式。

FastAPI 路由：
- `POST /api/ingest` → 接收上传的 PDF 文件，调用 `ingestion.ingest_pdf()` 入库，返回 `{"status": "success", "chunks": count}`
- `POST /api/ask` → 接收 `{"question": "..."}`，调用 `retrieval.answer_question()`，返回 `{"answer": "...", "sources": [...]}`
- `GET /api/health` → 健康检查

Gradio 界面（用 `gr.mount_gradio_app` 挂载到 FastAPI，或用 `app = gr.routes`）：
- 一个文本输入框：提问
- 一个输出区：显示答案 + 来源引用
- 可选：文件上传组件用于上传 PDF

启动方式：
```bash
# 方式一：FastAPI + Gradio
uvicorn app.main:app --reload --port 8000

# 方式二：先入库再问答
python -m app.ingestion   # 扫描 data/docs/ 建/更新索引
```

## 实现要求

1. **不要硬编码 API Key**，全部从 config.py 读取
2. **错误处理**：索引不存在时给清晰提示（"请先运行 python -m app.ingestion 入库"）；API 调用失败时返回友好错误信息
3. **日志**：用 `logging` 替代 print，关键步骤（入库、检索、LLM 调用）要有日志
4. **data/docs/ 和 data/faiss_index/ 目录**：代码中用 `os.makedirs(exist_ok=True)` 自动创建，不报错
5. **FAISS 索引追加**：如果索引已存在，新文档应该追加到已有索引，不是覆盖重建
6. **Gradio UI 样式**：标题"企业知识库问答"，简洁现代风格

## 验收标准

- [ ] `python -m app.ingestion` 能扫描 data/docs/ 下 PDF 并建索引
- [ ] `uvicorn app.main:app` 启动后，POST /api/ask 能返回带来源的答案
- [ ] Gradio 界面能输入问题、显示答案和来源
- [ ] 没有 API Key 硬编码
- [ ] .gitignore 已忽略 .env 和 data/faiss_index/

## 开发顺序建议

1. `config.py`（最简单，5 分钟）
2. `vector_store.py`（FAISS 增删查，10 分钟）
3. `ingestion.py`（PyMuPDF + 分块 + 建索引，10 分钟）
4. `retrieval.py`（检索 + prompt + LLM 调用，10 分钟）
5. `main.py`（FastAPI 路由 + Gradio UI，10 分钟）
6. 往 data/docs/ 放一份 PDF，跑 `python -m app.ingestion` 建索引
7. 启动服务，提问，验证答案带来源引用
