"""文档入库：加载 → 分块 → 向量化 → 存向量索引（FAISS 或 Qdrant）。

支持 PDF 与 Word(.docx) 两种格式，解析后统一成同一套 metadata 结构
（source / file_name / page / chunk_id），便于后续混合检索。

分块时做中文标题增强：识别章节标题后拼到其下每个 chunk 的正文前，
形如 "[标题] 第三章 报销流程\n正文…"，提升关键词与向量召回的可定位性。
"""

import logging
import os
import re

import docx
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config, vector_store

logger = logging.getLogger(__name__)

# 支持入库的文件扩展名（均小写）
SUPPORTED_EXTENSIONS = (".pdf", ".docx")

# 标题启发式：中文章节号 / 多级数字编号 / 中文序号列举
_HEADING_PATTERNS = (
    re.compile(r"^第\s*[0-9一二三四五六七八九十百零]+\s*[章节篇条部分]"),
    re.compile(r"^\d+(\.\d+)+[\s、.．]"),
    re.compile(r"^[一二三四五六七八九十]+\s*[、.．]\s*\S"),
)
# 标题行长度上限，超长的多半是正文而非标题
_MAX_HEADING_LEN = 40
# 以这些标点结尾的加粗短行通常是一句正文，不作为标题
_SENTENCE_ENDINGS = ("。", "！", "？", "；", ".", "!", "?", ";")
# 短行判定上限：加粗且不超过该长度才算标题
_MAX_BOLD_HEADING_LEN = 20

# PyMuPDF span.flags 第 4 位表示粗体
_PYMUPDF_BOLD_FLAG = 1 << 4


def normalize_source(path: str) -> str:
    """归一化文档来源路径，保证同一文件在不同写法下能正确去重。"""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """构造切分器，供各格式解析函数共用，保证切分参数一致。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def _is_heading(text: str, bold: bool = False) -> bool:
    """判断一行文本是否像标题：章节号/数字编号开头，或加粗的短行。"""
    line = text.strip()
    if not line or len(line) > _MAX_HEADING_LEN:
        return False
    if any(pattern.match(line) for pattern in _HEADING_PATTERNS):
        return True
    # 加粗短行也算标题，但以句末标点结尾的通常是一句正文（如“本办法由人力资源部负责解释。”）
    return (
        bold
        and len(line) <= _MAX_BOLD_HEADING_LEN
        and not line.endswith(_SENTENCE_ENDINGS)
    )


def _group_sections(blocks: list[tuple[str, bool]]) -> list[tuple[str | None, str]]:
    """把 [(文本, 是否标题)] 归并成 [(所属标题, 正文)] 分节。

    标题下没有正文时（如连续两个标题、或文末标题），把标题本身作为独立一节，
    避免标题文本在切分时被丢掉。
    """
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal heading
        if buffer:
            sections.append((heading, "\n".join(buffer)))
            buffer.clear()
        elif heading is not None:
            sections.append((None, heading))
            heading = None

    for text, is_heading in blocks:
        if is_heading:
            flush()
            heading = text
        else:
            buffer.append(text)
    flush()
    return sections


def _section_to_chunks(
    splitter: RecursiveCharacterTextSplitter, heading: str | None, text: str
) -> list[str]:
    """切分一节正文，并把标题拼到该节每个分块前面。

    形如 "[标题] 第三章 报销流程\\n正文…"；没有标题时原样返回。
    """
    pieces = [piece for piece in splitter.split_text(text) if piece.strip()]
    if not heading:
        return pieces
    return [f"[标题] {heading}\n{piece}" for piece in pieces]


def _finalize_chunks(chunks: list[Document], file_path: str) -> list[Document]:
    """统一补 metadata：source / file_name / page / chunk_id。

    chunk_id 形如 "{file_name}_p{page}_{i}"，i 为文档内块序号，
    供混合检索的 RRF 融合作为唯一 key（同一页的多个分块必须能区分开）。
    """
    file_name = os.path.basename(file_path)
    source = normalize_source(file_path)
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        chunk.metadata = {
            "source": source,
            "file_name": file_name,
            "page": page,
            "chunk_id": f"{file_name}_p{page}_{i}",
        }
    return chunks


def _pdf_page_blocks(page) -> list[tuple[str, bool]]:
    """按 block 顺序读出一页的 [(文本, 是否标题)]。"""
    blocks = []
    # type=0 为文本块，1 为图片块（图片无文字，跳过即可）
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue

        lines, bold_chars, total_chars = [], 0, 0
        for line in block.get("lines", []):
            line_text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            if line_text:
                lines.append(line_text)
            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                total_chars += len(span_text)
                if span.get("flags", 0) & _PYMUPDF_BOLD_FLAG:
                    bold_chars += len(span_text)

        text = "\n".join(lines).strip()
        if not text:
            continue

        # 整块大部分字符加粗时视为加粗块
        is_bold = total_chars > 0 and bold_chars / total_chars >= 0.6
        blocks.append((text, _is_heading(text, bold=is_bold)))
    return blocks


def _docx_paragraph_is_bold(paragraph) -> bool:
    """段落是否整体加粗（Word 里常用作小标题）。"""
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(run.bold for run in runs)


def _docx_is_heading(paragraph) -> bool:
    """判断 Word 段落是否为标题：套用标题样式，或加粗短行，或章节号开头。"""
    style_name = paragraph.style.name if paragraph.style is not None else ""
    if "Heading" in style_name or "标题" in style_name:
        return True
    return _is_heading(paragraph.text, bold=_docx_paragraph_is_bold(paragraph))


def _docx_blocks(document) -> list[tuple[str, bool]]:
    """按段落顺序读出 Word 正文的 [(文本, 是否标题)]，表格行附在最后。"""
    blocks = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append((text, _docx_is_heading(paragraph)))

    # 表格行用制表符连接以保留行列关系；合并单元格可能重复，轻微重复可接受
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                blocks.append(("\t".join(cells), False))
    return blocks


def _split_pdf(file_path: str) -> list[Document]:
    """加载 PDF 并切分为带 metadata 的分块（页码 1-based，含标题增强）。"""
    logger.info("正在解析 PDF：%s", file_path)
    splitter = _build_splitter()
    chunks: list[Document] = []

    with pymupdf.open(file_path) as pdf:
        page_count = pdf.page_count
        for page_number, page in enumerate(pdf, start=1):
            for heading, text in _group_sections(_pdf_page_blocks(page)):
                for content in _section_to_chunks(splitter, heading, text):
                    chunks.append(
                        Document(page_content=content, metadata={"page": page_number})
                    )

    if not chunks:
        logger.warning("PDF 无可用文本内容：%s", file_path)
        return []

    _finalize_chunks(chunks, file_path)
    logger.info(
        "%s → %d 页 → %d 个分块", os.path.basename(file_path), page_count, len(chunks)
    )
    return chunks


def _split_docx(file_path: str) -> list[Document]:
    """加载 Word(.docx) 并切分为带 metadata 的分块（含标题增强）。

    python-docx 只能读取段落与表格中的文本；文档里的图片、文本框、SmartArt
    读不到，这属于正常情况，不视为错误。docx 没有页码概念，page 统一填 0，
    保持与 PDF 一致的 metadata 结构以便混合检索。
    """
    logger.info("正在解析 Word：%s", file_path)
    document = docx.Document(file_path)
    splitter = _build_splitter()
    chunks: list[Document] = []

    for heading, text in _group_sections(_docx_blocks(document)):
        for content in _section_to_chunks(splitter, heading, text):
            chunks.append(Document(page_content=content, metadata={"page": 0}))

    if not chunks:
        logger.warning("Word 无可用文本内容：%s", file_path)
        return []

    _finalize_chunks(chunks, file_path)
    logger.info("%s → %d 个分块", os.path.basename(file_path), len(chunks))
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
