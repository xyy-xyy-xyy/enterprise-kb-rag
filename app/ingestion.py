"""文档入库：加载 → 分块 → 向量化 → 存向量索引（FAISS 或 Qdrant）。

支持 PDF 与 Word(.docx) 两种格式，解析后统一成同一套 metadata 结构
（source / file_name / page / chunk_id / paragraph_start / paragraph_end / sm3_hash），
便于后续混合检索与回答里的精确定位（PDF 用页码，Word 用段落号）。

分块时做中文标题增强：识别章节标题后拼到其下每个 chunk 的正文前，
形如 "[标题] 第三章 报销流程\n正文…"，提升关键词与向量召回的可定位性。

文档可以密文（`.enc`）落盘：解析前在内存里解密，全程不落临时文件。
`file_name` / `source` 一律用剥掉 `.enc` 的逻辑名，否则明文版与密文版会被判成两个文档。
"""

import io
import logging
import os
import re

import docx
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config, crypto, vector_store

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


def _pdf_page_blocks(page) -> list[tuple[str, bool, int]]:
    """按 block 顺序读出一页的 [(文本, 是否标题, 页内块号)]。

    页内块号 = 该页中文本块（type=0）的出现顺序，从 1 开始，
    用于回答里"第 N 段"的精确定位；图片块(type!=0)不占号。
    """
    blocks = []
    # type=0 为文本块，1 为图片块（图片无文字，跳过即可）
    block_no = 0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        block_no += 1

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
        blocks.append((text, _is_heading(text, bold=is_bold), block_no))
    return blocks


def _group_sections(
    blocks: list[tuple[str, bool, int]],
) -> list[tuple[str | None, str, int, int]]:
    """把 [(文本, 是否标题, 位置号)] 归并成 [(所属标题, 正文, 起始位置, 结束位置)] 分节。

    位置号 = PDF 页内文本块号 / Word 段落号（见 _pdf_page_blocks / _docx_blocks）。
    起始/结束位置取该节覆盖的最小/最大位置号（标题本身的位置也计入），
    供后续把"第 N-M 段"的精确定位透传到每个分块的 metadata。

    标题下没有正文时（如连续两个标题、或文末标题），把标题本身作为独立一节、
    其内容就是标题文本、位置为该标题所在位置，避免标题文本在切分时被丢掉。
    """
    sections: list[tuple[str | None, str, int, int]] = []
    heading: str | None = None
    heading_pos: int | None = None
    buffer: list[str] = []
    positions: list[int] = []

    def flush() -> None:
        nonlocal heading, heading_pos
        if buffer:
            sections.append((heading, "\n".join(buffer), positions[0], positions[-1]))
            buffer.clear()
            positions.clear()
        elif heading is not None:
            pos = heading_pos if heading_pos is not None else 0
            # heading=None：独立标题段不额外加标题前缀，内容即标题文本
            sections.append((None, heading, pos, pos))
            heading = None
            heading_pos = None

    for text, is_heading, pos in blocks:
        if is_heading:
            flush()
            heading = text
            heading_pos = pos
        else:
            buffer.append(text)
            positions.append(pos)
    flush()
    return sections


def _section_to_chunks(
    splitter: RecursiveCharacterTextSplitter,
    heading: str | None,
    text: str,
    para_start: int,
    para_end: int,
) -> list[tuple[str, int, int]]:
    """切分一节正文，并把标题拼到该节每个分块前面。

    形如 "[标题] 第三章 报销流程\n正文…"；没有标题时原样返回。
    一个 section 被切分成多个 chunk 时，这些 chunk 共享该 section 的段落范围
    (para_start, para_end)，作为回答里"第 N-M 段"定位的依据。
    返回 [(分块文本, 起始位置, 结束位置)]。
    """
    pieces = [piece for piece in splitter.split_text(text) if piece.strip()]
    if not heading:
        return [(piece, para_start, para_end) for piece in pieces]
    return [
        (f"[标题] {heading}\n{piece}", para_start, para_end) for piece in pieces
    ]


def _finalize_chunks(
    chunks: list[Document], file_path: str, sm3_hash: str | None = None
) -> list[Document]:
    """统一补 metadata：source / file_name / page / chunk_id / 段落定位 / sm3_hash。

    chunk_id 形如 "{file_name}_p{page}_{i}"，i 为文档内块序号，
    供混合检索的 RRF 融合作为唯一 key（同一页的多个分块必须能区分开）。
    sm3_hash 只进 metadata，**不要拼进 chunk_id**：chunk_id 格式一变，
    RRF 融合的 key 全部错位。

    注意本函数是整块重写 metadata，任何新字段都必须在这里补，否则会被静默抹掉。

    paragraph_start / paragraph_end 由 _split_pdf / _split_docx 写入；
    旧索引重建前可能缺失（None），不要抛异常，缺失即不写。
    """
    file_name = crypto.logical_name(file_path)
    source = crypto.logical_source(file_path)
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        metadata = {
            "source": source,
            "file_name": file_name,
            "page": page,
            "chunk_id": f"{file_name}_p{page}_{i}",
            "paragraph_start": chunk.metadata.get("paragraph_start"),
            "paragraph_end": chunk.metadata.get("paragraph_end"),
        }
        # sm3_hash 为 None 时不写，保持与旧索引一致的结构
        if sm3_hash:
            metadata["sm3_hash"] = sm3_hash
        chunk.metadata = metadata
    return chunks


def read_document_bytes(file_path: str) -> tuple[bytes, str, str]:
    """读取文档明文，返回 (明文字节, 逻辑文件名, 逻辑 source)。

    `.enc` 走内存解密，明文直接读；调用方不需要关心磁盘上是哪种形态。
    """
    if crypto.is_encrypted(file_path):
        key = crypto.load_or_create_key()
        data = crypto.decrypt_file_to_bytes(file_path, key)
    else:
        with open(file_path, "rb") as f:
            data = f.read()
    return data, crypto.logical_name(file_path), crypto.logical_source(file_path)


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


def _docx_blocks(document) -> list[tuple[str, bool, int]]:
    """按段落顺序读出 Word 正文的 [(文本, 是否标题, 段落号)]，表格行附在最后。

    段落号 = document.paragraphs 的真实下标 + 1（从 1 开始），空段落也占号，
    这样用户在 Word 里数段落能对上；表格行接在正文段落后继续排号（全局唯一）。
    """
    blocks = []
    paragraphs = document.paragraphs
    for idx, paragraph in enumerate(paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            blocks.append((text, _docx_is_heading(paragraph), idx))

    # 表格行用制表符连接以保留行列关系；合并单元格可能重复，轻微重复可接受
    # 段号从正文段数之后接着排，保证全文段落号唯一、可定位
    table_row_no = len(paragraphs)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                table_row_no += 1
                blocks.append(("\t".join(cells), False, table_row_no))
    return blocks


def _split_pdf(file_path: str) -> list[Document]:
    """加载 PDF 并切分为带 metadata 的分块（页码 1-based，含标题增强）。"""
    data, _, _ = read_document_bytes(file_path)
    return _split_pdf_bytes(data, file_path)


def _split_pdf_bytes(data: bytes, file_path: str) -> list[Document]:
    """从内存字节解析 PDF。

    `file_path` 只用于生成 metadata（逻辑名 / source），不再用于打开文件 ——
    密文文档在内存里解密后直接喂给解析器，明文全程不落盘。
    """
    logger.info("正在解析 PDF：%s", file_path)
    splitter = _build_splitter()
    chunks: list[Document] = []

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        page_count = pdf.page_count
        for page_number, page in enumerate(pdf, start=1):
            for heading, text, p_start, p_end in _group_sections(_pdf_page_blocks(page)):
                for content, c_start, c_end in _section_to_chunks(
                    splitter, heading, text, p_start, p_end
                ):
                    chunks.append(
                        Document(
                            page_content=content,
                            metadata={
                                "page": page_number,
                                "paragraph_start": c_start,
                                "paragraph_end": c_end,
                            },
                        )
                    )

    if not chunks:
        logger.warning("PDF 无可用文本内容：%s", file_path)
        return []

    _finalize_chunks(chunks, file_path, sm3_hash=crypto.sm3_hex(data))
    logger.info(
        "%s → %d 页 → %d 个分块", crypto.logical_name(file_path), page_count, len(chunks)
    )
    return chunks


def _split_docx(file_path: str) -> list[Document]:
    """加载 Word(.docx) 并切分为带 metadata 的分块（含标题增强）。"""
    data, _, _ = read_document_bytes(file_path)
    return _split_docx_bytes(data, file_path)


def _split_docx_bytes(data: bytes, file_path: str) -> list[Document]:
    """从内存字节解析 Word(.docx)。

    python-docx 只能读取段落与表格中的文本；文档里的图片、文本框、SmartArt
    读不到，这属于正常情况，不视为错误。docx 没有页码概念，page 统一填 0，
    保持与 PDF 一致的 metadata 结构以便混合检索。

    同 `_split_pdf_bytes`，`file_path` 只用于生成 metadata。
    """
    logger.info("正在解析 Word：%s", file_path)
    document = docx.Document(io.BytesIO(data))
    splitter = _build_splitter()
    chunks: list[Document] = []

    for heading, text, p_start, p_end in _group_sections(_docx_blocks(document)):
        for content, c_start, c_end in _section_to_chunks(
            splitter, heading, text, p_start, p_end
        ):
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "page": 0,
                        "paragraph_start": c_start,
                        "paragraph_end": c_end,
                    },
                )
            )

    if not chunks:
        logger.warning("Word 无可用文本内容：%s", file_path)
        return []

    _finalize_chunks(chunks, file_path, sm3_hash=crypto.sm3_hex(data))
    logger.info("%s → %d 个分块", crypto.logical_name(file_path), len(chunks))
    return chunks


def _resolve_stored_path(file_path: str) -> str:
    """把逻辑路径解析成磁盘上的实际路径。

    加密存储下磁盘上是 `xxx.pdf.enc`，而调用方（API / UI / 遍历）手里通常是
    `xxx.pdf`。这里自动定位，调用方就不必关心存储形态。
    """
    if os.path.exists(file_path):
        return file_path
    encrypted = file_path + crypto.ENC_SUFFIX
    if os.path.exists(encrypted):
        return encrypted
    return file_path  # 都不存在：交给后面的 FileNotFoundError 报清楚


def ingest_file(file_path: str) -> int:
    """对单个文档（.pdf / .docx，或其 `.enc` 密文）执行完整入库流程，返回写入的分块数。

    索引已存在时追加而非覆盖。两种跳过要分清楚：
      - `已入库，跳过`：source（归一化路径）命中，同一路径重复入库
      - `内容重复已存在(<文件名>)，跳过`：SM3 命中，换个文件名传同一份内容也会被拦下
    需要重新解析同一文件请用 `ingest_directory(rebuild=True)` 重建索引。
    """
    config.setup_logging()

    stored_path = _resolve_stored_path(file_path)
    if not os.path.exists(stored_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    # 扩展名按逻辑名判断：密文路径的 splitext 结果是 ".enc"
    logical = crypto.logical_name(stored_path)
    ext = os.path.splitext(logical)[1].lower()
    if ext == ".pdf":
        split_func = _split_pdf_bytes
    elif ext == ".docx":
        split_func = _split_docx_bytes
    else:
        raise ValueError(
            f"不支持的文件类型：{ext or file_path}（仅支持 {' / '.join(SUPPORTED_EXTENSIONS)}）"
        )

    data, file_name, source = read_document_bytes(stored_path)
    sm3_hash = crypto.sm3_hex(data)

    if vector_store.index_exists():
        store = vector_store.load_index()
        if config.SM3_DEDUP:
            # 命中即说明内容已存在（哪怕换了文件名）；store 只对 FAISS 生效
            duplicate = vector_store.get_indexed_hashes(store).get(sm3_hash)
            if duplicate:
                logger.info(
                    "内容重复已存在(%s)，跳过：%s", duplicate, file_name
                )
                return 0
        if source in vector_store.get_indexed_sources(store):
            logger.info("已入库，跳过：%s", file_name)
            return 0
    else:
        logger.info("未检测到已有索引，将新建。")
        store = None

    chunks = split_func(data, stored_path)
    if not chunks:
        return 0

    if store is None:
        store = vector_store.create_index(chunks)
    else:
        store = vector_store.add_documents(store, chunks)

    vector_store.save_index(store)
    logger.info("入库完成：%s（%d 个分块）", file_name, len(chunks))
    return len(chunks)


def ingest_pdf(file_path: str) -> int:
    """兼容旧调用方：等价于 ingest_file，现同时支持 .pdf 与 .docx。"""
    return ingest_file(file_path)


def reset_index() -> None:
    """删除索引（用于重建）。FAISS 删文件，Qdrant 删 collection。"""
    vector_store.reset_index()


def _scan_document_files(dir_path: str) -> list[str]:
    """扫描目录下的文档：明文与密文都收，同名时优先密文。

    返回按逻辑名排序的实际路径列表（密文路径带 `.enc`）。
    """
    names = sorted(os.listdir(dir_path))
    encrypted = {
        name[: -len(crypto.ENC_SUFFIX)]: name
        for name in names
        if name.lower().endswith(crypto.ENC_SUFFIX)
        and name[: -len(crypto.ENC_SUFFIX)].lower().endswith(SUPPORTED_EXTENSIONS)
    }
    chosen = dict(encrypted)
    for name in names:
        if not name.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        if name in chosen:
            logger.warning(
                "明文与密文同时存在，本次使用密文：%s%s（明文 %s 已被忽略）",
                name,
                crypto.ENC_SUFFIX,
                name,
            )
            continue
        chosen[name] = name

    return [os.path.join(dir_path, name) for name in sorted(chosen.values())]


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

    doc_files = _scan_document_files(dir_path)
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
    # 国密相关的维护命令转发给 app.crypto，统一在那边实现
    parser.add_argument(
        "--encrypt-all", action="store_true", help="data/docs/ 明文 → .enc（转发给 app.crypto）"
    )
    parser.add_argument(
        "--verify-all", action="store_true", help="解密重算 SM3 并与索引比对（转发给 app.crypto）"
    )
    args = parser.parse_args()

    if args.encrypt_all:
        raise SystemExit(crypto.main(["--encrypt-all"]))
    if args.verify_all:
        raise SystemExit(crypto.main(["--verify-all"]))

    # 扫描 data/docs/ 下所有文档，建索引
    stats = ingest_directory(args.docs, rebuild=args.rebuild)
    print(
        f"文档入库完成：{stats['files']} 个文件，"
        f"新增 {stats['chunks']} 个分块，跳过 {stats['skipped']} 个"
    )
