"""入库层：分块、元数据、段落定位、密文读取。

语料全部来自 conftest 动态生成 —— CI 里 `data/docs/` 是空的（全被 gitignore）。
"""

import os

import docx
import pymupdf
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import crypto, ingestion


# ============================ _is_heading ============================


@pytest.mark.parametrize(
    "text",
    [
        "第一章 总则",
        "第 3 章 报销流程",  # 章号与"章"之间有空格也要认
        "第一节 适用范围",
        "1.2 住宿标准",  # 多级数字编号
        "三、报销时限",  # 中文序号 + 顿号
        "一. 适用范围",  # 中文序号 + 点号
    ],
)
def test_is_heading_recognizes_chapter_and_numbered_titles(text):
    assert ingestion._is_heading(text) is True


def test_is_heading_rejects_long_line_even_with_chapter_prefix():
    """超长行多半是正文，章节号开头也不认 —— 否则整段正文会被当标题拼进每个分块。"""
    long_line = "第一章 " + "正" * 60
    assert len(long_line) > ingestion._MAX_HEADING_LEN
    assert ingestion._is_heading(long_line) is False


def test_is_heading_treats_bold_short_line_as_title():
    assert ingestion._is_heading("总则", bold=True) is True


def test_is_heading_rejects_bold_line_ending_with_sentence_punctuation():
    """加粗但以句末标点结尾的短行是一句正文（如"本办法由人力资源部负责解释。"）。"""
    assert ingestion._is_heading("本办法由人力资源部负责解释。", bold=True) is False


def test_is_heading_rejects_bold_line_over_length_limit():
    text = "年" * (ingestion._MAX_BOLD_HEADING_LEN + 1)
    assert ingestion._is_heading(text, bold=True) is False


def test_is_heading_rejects_plain_body_and_blank():
    assert ingestion._is_heading("一线城市住宿费上限为每晚 800 元。") is False
    assert ingestion._is_heading("") is False
    assert ingestion._is_heading("   ") is False


# ============================ _group_sections ============================


def test_group_sections_attaches_body_to_its_heading():
    blocks = [("第一章 总则", True, 1), ("正文一", False, 2), ("正文二", False, 3)]
    assert ingestion._group_sections(blocks) == [("第一章 总则", "正文一\n正文二", 2, 3)]


def test_group_sections_keeps_heading_without_body_as_its_own_section():
    """连续两个标题时，前一个没有正文 —— 它必须自成一段，否则标题文字直接丢失。

    且该段的 heading 必须是 None：内容已经是标题本身，再加 "[标题] " 前缀会重复。
    """
    blocks = [("第一章 总则", True, 1), ("第二章 附则", True, 2), ("正文", False, 3)]
    assert ingestion._group_sections(blocks) == [
        (None, "第一章 总则", 1, 1),
        ("第二章 附则", "正文", 3, 3),
    ]


def test_group_sections_flushes_trailing_heading_at_end_of_document():
    blocks = [("正文", False, 5), ("第三章 附则", True, 6)]
    assert ingestion._group_sections(blocks) == [
        (None, "正文", 5, 5),
        (None, "第三章 附则", 6, 6),
    ]


def test_group_sections_returns_empty_for_no_blocks():
    assert ingestion._group_sections([]) == []


# ============================ _section_to_chunks ============================


def test_section_to_chunks_prefixes_heading_to_every_chunk():
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    chunks = ingestion._section_to_chunks(splitter, "第一章 总则", "甲甲甲甲甲甲\n乙乙乙乙乙乙", 2, 3)
    assert len(chunks) == 2
    assert chunks[0][0].startswith("[标题] 第一章 总则\n")
    assert chunks[1][0].startswith("[标题] 第一章 总则\n")


def test_section_to_chunks_without_heading_leaves_text_untouched():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = ingestion._section_to_chunks(splitter, None, "一线城市住宿费上限为每晚 800 元。", 4, 4)
    assert chunks == [("一线城市住宿费上限为每晚 800 元。", 4, 4)]


def test_section_to_chunks_shares_paragraph_range_across_pieces():
    """同一 section 切成多块时，这些块共享段落范围 —— 定位到"第 5-7 段"而不是各说各的。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8, chunk_overlap=0)
    chunks = ingestion._section_to_chunks(splitter, None, "甲" * 30, 5, 7)
    assert len(chunks) > 1
    assert {(start, end) for _, start, end in chunks} == {(5, 7)}


def test_section_to_chunks_drops_blank_pieces():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    assert ingestion._section_to_chunks(splitter, None, "   \n  ", 1, 1) == []


# ============================ _finalize_chunks ============================


def test_finalize_chunks_builds_unique_chunk_ids():
    chunks = [
        ingestion.Document(page_content="甲", metadata={"page": 1, "paragraph_start": 1}),
        ingestion.Document(page_content="乙", metadata={"page": 1, "paragraph_start": 2}),
        ingestion.Document(page_content="丙", metadata={"page": 2, "paragraph_start": 3}),
    ]
    result = ingestion._finalize_chunks(chunks, "sample.pdf")
    ids = [c.metadata["chunk_id"] for c in result]
    assert ids == ["sample.pdf_p1_0", "sample.pdf_p1_1", "sample.pdf_p2_2"]
    # 同一页的多个分块必须能区分开，否则 RRF 融合会把它们并成一块
    assert len(set(ids)) == len(ids)


def test_finalize_chunks_strips_enc_suffix_from_file_name_and_source():
    """密文路径的 file_name 必须是逻辑名，否则同一文档的明文版与密文版会被判成两个。"""
    chunks = [ingestion.Document(page_content="甲", metadata={"page": 1})]
    result = ingestion._finalize_chunks(chunks, os.path.join("data", "docs", "a.pdf.enc"))
    assert result[0].metadata["file_name"] == "a.pdf"
    assert result[0].metadata["source"].endswith("a.pdf")


def test_finalize_chunks_rewrites_metadata_wholesale():
    """本函数整块重写 metadata —— 未声明的字段会被静默丢掉，这个行为必须被固定住。"""
    chunks = [
        ingestion.Document(
            page_content="甲",
            metadata={"page": 1, "paragraph_start": 2, "paragraph_end": 4, "临时字段": "x"},
        )
    ]
    meta = ingestion._finalize_chunks(chunks, "sample.pdf")[0].metadata
    assert "临时字段" not in meta
    assert meta["paragraph_start"] == 2
    assert meta["paragraph_end"] == 4
    assert set(meta) == {
        "source",
        "file_name",
        "page",
        "chunk_id",
        "paragraph_start",
        "paragraph_end",
    }


def test_finalize_chunks_omits_sm3_hash_when_absent_and_includes_it_when_present():
    chunks = [ingestion.Document(page_content="甲", metadata={"page": 1})]
    assert "sm3_hash" not in ingestion._finalize_chunks(chunks, "a.pdf")[0].metadata
    assert (
        ingestion._finalize_chunks(chunks, "a.pdf", sm3_hash="deadbeef")[0].metadata["sm3_hash"]
        == "deadbeef"
    )


def test_finalize_chunks_keeps_sm3_hash_out_of_chunk_id():
    """sm3_hash 只进 metadata。一旦拼进 chunk_id，RRF 融合的 key 会全部错位。"""
    chunks = [ingestion.Document(page_content="甲", metadata={"page": 1})]
    meta = ingestion._finalize_chunks(chunks, "a.pdf", sm3_hash="deadbeef")[0].metadata
    assert "deadbeef" not in meta["chunk_id"]


# ============================ normalize_source ============================


def test_normalize_source_makes_relative_and_absolute_paths_equal():
    relative = os.path.join("data", "docs", "a.pdf")
    assert ingestion.normalize_source(relative) == ingestion.normalize_source(
        os.path.abspath(relative)
    )


def test_normalize_source_keeps_different_files_distinct():
    assert ingestion.normalize_source("data/docs/a.pdf") != ingestion.normalize_source(
        "data/docs/b.pdf"
    )


# ============================ PDF 解析 ============================


def test_split_pdf_bytes_uses_one_based_page_numbers(sample_pdf_bytes):
    chunks = ingestion._split_pdf_bytes(sample_pdf_bytes, "sample.pdf")
    assert [c.metadata["page"] for c in chunks] == [1, 3]


def test_split_pdf_bytes_skips_blank_page_entirely(sample_pdf_bytes):
    """第 2 页是空白页：不产生分块，也不能把第 3 页的内容错标成第 2 页。"""
    chunks = ingestion._split_pdf_bytes(sample_pdf_bytes, "sample.pdf")
    assert 2 not in [c.metadata["page"] for c in chunks]


def test_split_pdf_bytes_prepends_heading_and_sets_paragraph_range(sample_pdf_bytes):
    chunks = ingestion._split_pdf_bytes(sample_pdf_bytes, "sample.pdf")
    first = chunks[0]
    assert first.page_content.startswith("[标题] 第一章 住宿标准\n")
    assert "一线城市住宿费上限为每晚 800 元。" in first.page_content
    assert (first.metadata["paragraph_start"], first.metadata["paragraph_end"]) == (2, 2)


def test_split_pdf_bytes_stamps_sm3_of_source_bytes(sample_pdf_bytes):
    chunks = ingestion._split_pdf_bytes(sample_pdf_bytes, "sample.pdf")
    assert chunks[0].metadata["sm3_hash"] == crypto.sm3_hex(sample_pdf_bytes)


def test_split_pdf_bytes_returns_empty_for_blank_document():
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    assert ingestion._split_pdf_bytes(data, "blank.pdf") == []


# ============================ Word 解析 ============================


def test_split_docx_bytes_marks_word_chunks_as_pageless(sample_docx_bytes):
    chunks = ingestion._split_docx_bytes(sample_docx_bytes, "sample.docx")
    assert [c.metadata["page"] for c in chunks] == [0, 0]


def test_split_docx_bytes_blank_paragraph_still_consumes_a_paragraph_number(sample_docx_bytes):
    """空段落不产生内容块，但**照样占号** —— 用户在 Word 里数段落才能对上。

    语料里段落号 3 是空段落，所以标题在 4、正文从 5 开始；
    若空段落不占号，这里会变成 1-3 而不是 2-2 / 5-7。
    """
    chunks = ingestion._split_docx_bytes(sample_docx_bytes, "sample.docx")
    assert (chunks[0].metadata["paragraph_start"], chunks[0].metadata["paragraph_end"]) == (2, 2)
    assert (chunks[1].metadata["paragraph_start"], chunks[1].metadata["paragraph_end"]) == (5, 7)


def test_split_docx_bytes_appends_table_rows_after_body_paragraphs(sample_docx_bytes):
    """表格行的段号从正文段数之后接着排（语料里是 6、7），保证全文段号唯一。"""
    chunks = ingestion._split_docx_bytes(sample_docx_bytes, "sample.docx")
    table_chunk = chunks[-1]
    assert "项目\t标准" in table_chunk.page_content
    assert "年假\t5 天" in table_chunk.page_content


def test_split_docx_bytes_recognizes_heading_style_and_chapter_prefix(sample_docx_bytes):
    """标题识别有两条路：Word 的 Heading 样式、以及"第 X 章"这种章节号前缀。"""
    chunks = ingestion._split_docx_bytes(sample_docx_bytes, "sample.docx")
    assert chunks[0].page_content.startswith("[标题] 员工考勤与请假管理办法\n")
    assert chunks[1].page_content.startswith("[标题] 第一章 年假\n")


def test_docx_blocks_number_skips_blank_but_keeps_index(tmp_path):
    """直接验 _docx_blocks：空段落不出现在结果里，但它占掉的号必须留下痕迹。"""
    document = docx.Document()
    document.add_paragraph("甲")
    document.add_paragraph("")
    document.add_paragraph("乙")
    blocks = ingestion._docx_blocks(document)
    assert blocks == [("甲", False, 1), ("乙", False, 3)]


# ============================ 密文读取 ============================


def test_read_document_bytes_returns_file_bytes_for_plaintext(tmp_path):
    path = tmp_path / "plain.docx"
    path.write_bytes(b"\x89PNG-fake-bytes")
    data, name, source = ingestion.read_document_bytes(str(path))
    assert data == b"\x89PNG-fake-bytes"
    assert name == "plain.docx"


def test_read_document_bytes_decrypts_enc_file_to_identical_bytes(
    tmp_path, monkeypatch, sample_sm4_key
):
    """加密 → 读取 的往返必须逐字节一致，否则所有密文文档都解析不出来。"""
    monkeypatch.setattr(crypto.config, "SM4_KEY", sample_sm4_key)

    original = b"PDF-or-DOCX-payload-\x00\x01\x02"
    plain = tmp_path / "secret.pdf"
    plain.write_bytes(original)

    enc_path = crypto.encrypt_file(str(plain), crypto.load_or_create_key())
    assert enc_path.endswith(".enc")

    data, name, source = ingestion.read_document_bytes(enc_path)
    assert data == original
    # 逻辑名必须剥掉 .enc，否则密文版与明文版会被当成两份文档
    assert name == "secret.pdf"
    assert not source.endswith(".enc")


def test_read_document_bytes_rejects_ciphertext_with_a_broken_length(
    tmp_path, monkeypatch, sample_sm4_key
):
    """长度不是 16 的整数倍 → 结构都不对，必须直接拒绝。"""
    monkeypatch.setattr(crypto.config, "SM4_KEY", sample_sm4_key)
    plain = tmp_path / "secret.pdf"
    plain.write_bytes(b"payload" * 10)
    crypto.encrypt_file(str(plain), crypto.load_or_create_key())

    enc = tmp_path / "secret.pdf.enc"
    enc.write_bytes(enc.read_bytes()[:-1])  # 砍掉一个字节

    with pytest.raises(ValueError):
        ingestion.read_document_bytes(str(enc))


def test_read_document_bytes_raises_when_the_ciphertext_padding_is_broken(
    tmp_path, monkeypatch, sample_sm4_key
):
    """密文尾部被改动、导致 PKCS#7 填充不合法时，必须报错而不是吐出乱码喂给解析器。

    ⚠️ 这里**逐个试**篡改值，而不是直接翻一位就断言报错 —— 因为 SM4-CBC 只有
    PKCS#7 填充、没有 MAC，翻掉最后一个字节后能不能被发现，取决于解密出的最后
    一块是否恰好构成合法填充（约 255/256 会失败，约 1/256 会**静默通过**并返回
    一段乱码明文）。直接翻一位的写法有 ~0.4% 的随机翻车率，在 CI 上就是偶发红灯。
    先试出一个确实会失败的值，断言才是确定的。

    另见交付报告：整块篡改（非末尾块）**不会**被解密层发现，`--verify-all`
    才是覆盖这一情形的完整性手段。
    """
    monkeypatch.setattr(crypto.config, "SM4_KEY", sample_sm4_key)
    plain = tmp_path / "secret.pdf"
    plain.write_bytes(b"payload" * 10)
    enc_path = crypto.encrypt_file(str(plain), crypto.load_or_create_key())

    enc = tmp_path / "secret.pdf.enc"
    good = bytearray(enc.read_bytes())

    for candidate in range(256):
        if candidate == good[-1]:
            continue
        blob = bytearray(good)
        blob[-1] = candidate
        enc.write_bytes(bytes(blob))
        try:
            ingestion.read_document_bytes(enc_path)
        except Exception:
            break
    else:
        pytest.fail("256 种尾部篡改无一被发现 —— PKCS#7 校验形同虚设")

    with pytest.raises(Exception):
        ingestion.read_document_bytes(enc_path)


def test_read_document_bytes_cannot_detect_tampering_in_a_non_final_block(
    tmp_path, monkeypatch, sample_sm4_key
):
    """钉住一个安全事实：SM4-CBC 无 MAC，改动**非末尾块**的密文不会被发现。

    CBC 的性质：改 C_i 会把 P_i 整块搅乱，并只让 P_{i+1} 对应位置翻一位。
    所以只动最后一个密文块**之前**的块、且避开 P_末块 的填充区，填充校验就依然
    通过，解密"成功"并返回一段乱码明文。密文完整性因此不能指望解密层，
    必须靠 SM3 摘要 —— `python -m app.crypto --verify-all` 存在的意义就在这里。

    明文 70 字节 → 填充到 80 字节 = 5 个块，密文 = IV(16) + 5×16 = 96 字节。
    索引 64 是第 4 块的首字节，即末块的前一块；末块（P_5）的填充在尾部 10 字节，
    不受影响。
    """
    monkeypatch.setattr(crypto.config, "SM4_KEY", sample_sm4_key)
    original = b"payload" * 10
    plain = tmp_path / "secret.pdf"
    plain.write_bytes(original)
    enc_path = crypto.encrypt_file(str(plain), crypto.load_or_create_key())

    enc = tmp_path / "secret.pdf.enc"
    blob = bytearray(enc.read_bytes())
    assert len(blob) == 96, "块布局变了的话下面的偏移量就不再成立"
    blob[64] ^= 0xFF
    enc.write_bytes(bytes(blob))

    data, _, _ = ingestion.read_document_bytes(enc_path)  # 不报错 —— 这正是问题

    assert len(data) == len(original), "填充仍合法，长度不该变"
    assert data[:48] == original[:48], "前 3 块不受影响"
    assert data[48:64] != original[48:64], "第 4 块被整块搅乱"
    assert data[65:] == original[65:], "第 5 块只有第 1 个字节翻转"


# ============================ _scan_document_files ============================


def test_scan_document_files_collects_plain_and_encrypted_sorted(tmp_path):
    for name in ["alpha.pdf", "beta.docx.enc", "log.txt"]:
        (tmp_path / name).write_bytes(b"x")

    found = [os.path.basename(p) for p in ingestion._scan_document_files(str(tmp_path))]
    assert found == ["alpha.pdf", "beta.docx.enc"]


def test_scan_document_files_prefers_ciphertext_when_both_exist(tmp_path):
    """明文与密文并存时用密文（明文可能是上次加密留下的中间态）。"""
    (tmp_path / "gamma.pdf").write_bytes(b"plain")
    (tmp_path / "gamma.pdf.enc").write_bytes(b"enc")

    found = [os.path.basename(p) for p in ingestion._scan_document_files(str(tmp_path))]
    assert found == ["gamma.pdf.enc"]


def test_scan_document_files_ignores_enc_with_unsupported_inner_extension(tmp_path):
    """`notes.txt.enc` 的内层扩展名不是 .pdf/.docx，不该被当成文档收进来。"""
    (tmp_path / "notes.txt.enc").write_bytes(b"enc")
    assert ingestion._scan_document_files(str(tmp_path)) == []
