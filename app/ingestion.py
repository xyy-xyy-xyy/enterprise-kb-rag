"""文档入库：加载 → 分块 → 向量化 → 存向量索引（FAISS 或 Qdrant）。

支持 PDF 与 Word(.docx) 两种格式，解析后统一成同一套 metadata 结构
（source / file_name / page），便于后续混合检索。
"""

import logging
import os

import docx
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config, vector_store

logger = logging.getLogger(__name__)

# 支持入库的文件扩展名（均小写）
SUPPORTED_EXTENSIONS = (".pdf", ".docx")


def normalize_source(path: str) -> str:
    """归一化文档来源路径，保证同一文件在不同写法下能正确去重。"""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """构造切分器，供各格式解析函数共用，保证切分参数一致。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def _split_pdf(file_path: str) -> list[Document]:
    """加载 PDF 并切分为带 metadata 的分块。"""
    logger.info("正在解析 PDF：%s", file_path)
    pages = PyMuPDFLoader(file_path).load()
    if not pages:
        logger.warning("PDF 无可用文本内容：%s", file_path)
        return []

    chunks = _build_splitter().split_documents(pages)

    file_name = os.path.basename(file_path)
    for chunk in chunks:
        # PyMuPDFLoader 的 page 从 0 开始，这里转成人类可读的 1-based 页码
        page = chunk.metadata.get("page", 0) + 1
        chunk.metadata = {
            "source": normalize_source(file_path),
            "file_name": file_name,
            "page": page,
        }

    logger.info("%s → %d 页 → %d 个分块", file_name, len(pages), len(chunks))
    return chunks


def _split_docx(file_path: str) -> list[Document]:
    """加载 Word(.docx) 并切分为带 metadata 的分块。

    python-docx 只能读取段落与表格中的文本；文档里的图片、文本框、SmartArt
    读不到，这属于正常情况，不视为错误。docx 没有页码概念，page 统一填 0，
    保持与 PDF 一致的 metadata 结构以便混合检索。
    """
    logger.info("正在解析 Word：%s", file_path)
    document = docx.Document(file_path)

    # 段落文本（跳过空行，避免切分出大量空白块）
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    # 表格按行取值，用制表符连接单元格，尽量保留行列关系
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))

    full_text = "\n".join(parts)
    if not full_text:
        logger.warning("Word 无可用文本内容：%s", file_path)
        return []

    # 全文视为一个整体文档，切分参数与 PDF 完全一致
    chunks = _build_splitter().split_documents(
        [Document(page_content=full_text, metadata={})]
    )

    file_name = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata = {
            "source": normalize_source(file_path),
            "file_name": file_name,
            "page": 0,
        }

    logger.info("%s → %d 个分块", file_name, len(chunks))
    return chunks


def ingest_file(file_path: str) -> int:
    """对单个文档（.pdf / .docx）执行完整入库流程，返回写入的分块数。

    索引已存在时追加而非覆盖；若该文件已入库则跳过（返回 0）。
    需要重新解析同一文件请用 `ingest_directory(rebuild=True)` 重建索引。
    """
    config.setup_logging()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        split_func = _split_pdf
    elif ext == ".docx":
        split_func = _split_docx
    else:
        raise ValueError(
            f"不支持的文件类型：{ext or file_path}（仅支持 {' / '.join(SUPPORTED_EXTENSIONS)}）"
        )

    if vector_store.index_exists():
        store = vector_store.load_index()
        if normalize_source(file_path) in vector_store.get_indexed_sources(store):
            logger.info("已入库，跳过：%s", file_path)
            return 0
    else:
        logger.info("未检测到已有索引，将新建。")
        store = None

    chunks = split_func(file_path)
    if not chunks:
        return 0

    if store is None:
        store = vector_store.create_index(chunks)
    else:
        store = vector_store.add_documents(store, chunks)

    vector_store.save_index(store)
    logger.info("入库完成：%s（%d 个分块）", os.path.basename(file_path), len(chunks))
    return len(chunks)


def ingest_pdf(file_path: str) -> int:
    """兼容旧调用方：等价于 ingest_file，现同时支持 .pdf 与 .docx。"""
    return ingest_file(file_path)


def reset_index() -> None:
    """删除索引（用于重建）。FAISS 删文件，Qdrant 删 collection。"""
    vector_store.reset_index()


def ingest_directory(dir_path: str = None, rebuild: bool = False) -> dict:
    """遍历目录下所有 .pdf / .docx 文件批量入库，返回统计信息。

    rebuild=True 时先清空索引再全量重建（会重新解析所有文档）。
    """
    config.setup_logging()
    dir_path = dir_path or config.DOCS_PATH
    os.makedirs(dir_path, exist_ok=True)

    if rebuild:
        logger.info("rebuild=True，将清空索引并全量重建。")
        reset_index()

    doc_files = sorted(
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if name.lower().endswith(SUPPORTED_EXTENSIONS)
    )
    if not doc_files:
        logger.warning("目录 %s 下没有找到文档（.pdf/.docx），请放入文档后重试。", dir_path)
        return {"files": 0, "chunks": 0, "skipped": 0}

    logger.info("在 %s 下发现 %d 个文档。", dir_path, len(doc_files))
    total_chunks, skipped = 0, 0
    for path in doc_files:
        try:
            written = ingest_file(path)
        except Exception:
            # 单个文件失败不应中断整批入库
            logger.exception("入库失败，已跳过：%s", path)
            continue
        if written == 0:
            skipped += 1
        total_chunks += written

    logger.info(
        "批量入库结束：%d 个文件，新增 %d 个分块，跳过 %d 个。",
        len(doc_files),
        total_chunks,
        skipped,
    )
    return {"files": len(doc_files), "chunks": total_chunks, "skipped": skipped}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="扫描文档（.pdf/.docx）并建立向量索引")
    parser.add_argument("--docs", default=config.DOCS_PATH, help="文档所在目录")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="清空已有索引并全量重建（默认只追加新文档）",
    )
    args = parser.parse_args()

    # 扫描 data/docs/ 下所有文档，建索引
    stats = ingest_directory(args.docs, rebuild=args.rebuild)
    print(
        f"文档入库完成：{stats['files']} 个文件，"
        f"新增 {stats['chunks']} 个分块，跳过 {stats['skipped']} 个"
    )
