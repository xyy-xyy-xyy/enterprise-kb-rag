"""国密 CLI 子命令：README 里对外承诺的那几个命令，端到端跑一遍。

`tests/test_crypto.py` 覆盖的是库函数（SM3/SM4 原语、加解密、密钥解析）；
这里补的是**命令本身**——用户实际敲的是 `python -m app.crypto --encrypt-all`，
里面的批量逻辑、跳过分支、`.bak` 退休语义、退出码，库函数测试一个都覆盖不到。

全部在 tmp_path 里跑：`config.DOCS_PATH` / `config.SM4_KEY_PATH` 都被改到临时目录，
绝不碰 `data/docs/` 与 `data/.sm4_key`。
"""

import os

import pytest

from app import config, crypto, ingestion, vector_store


@pytest.fixture
def cli_env(tmp_path, monkeypatch, sample_sm4_key):
    """把 CLI 的工作目录和密钥都指向临时目录。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(config, "DOCS_PATH", str(docs))
    monkeypatch.setattr(config, "SM4_KEY_PATH", str(tmp_path / ".sm4_key"))
    monkeypatch.setattr(config, "SM4_KEY", sample_sm4_key)
    return docs


def _write(docs, name, content=b"payload"):
    path = docs / name
    path.write_bytes(content)
    return path


# ============================ --selftest ============================


def test_selftest_returns_zero():
    """SM3 官方向量 + SM4 往返 + IV 随机性三个断言都由它跑，必须是绿的。"""
    assert crypto._cmd_selftest() == 0


def test_selftest_writes_nothing_and_needs_no_key(cli_env, capsys):
    """自测是只读的：不生成密钥、不改文档，可以随时在任何环境跑。"""
    crypto._cmd_selftest()
    assert list(cli_env.iterdir()) == []
    assert not os.path.exists(config.SM4_KEY_PATH)
    assert "SM3" in capsys.readouterr().out


# ============================ --gen-key ============================


def test_gen_key_writes_a_usable_key_file(cli_env, monkeypatch, capsys):
    """生成的密钥要能被 load_or_create_key 读回来，且是 32 位 hex（16 字节）。"""
    monkeypatch.setattr(config, "SM4_KEY", "")  # 模拟"没有环境变量"，走密钥文件分支

    assert crypto._cmd_gen_key() == 0

    assert os.path.exists(config.SM4_KEY_PATH)
    key_hex = open(config.SM4_KEY_PATH, encoding="utf-8").read().strip()
    assert len(key_hex) == 32
    assert int(key_hex, 16)  # 必须是合法 hex
    assert key_hex in capsys.readouterr().out
    assert crypto.load_or_create_key() == bytes.fromhex(key_hex)


def test_gen_key_warns_before_overwriting_an_existing_key_file(cli_env, monkeypatch, capsys):
    """覆盖已有密钥会让所有存量 .enc 永久解不开 —— 必须提前警告。"""
    monkeypatch.setattr(config, "SM4_KEY", "")
    crypto._cmd_gen_key()
    capsys.readouterr()

    crypto._cmd_gen_key()  # 第二次调用，密钥文件已存在

    assert "已存在，将被覆盖" in capsys.readouterr().out


# ============================ --encrypt-all / --decrypt-all ============================


def test_encrypt_all_encrypts_documents_and_retires_plaintext_to_bak(cli_env, capsys):
    """核心契约：产出 .enc、原明文改名 .bak 保留（绝不删除），返回 0。"""
    _write(cli_env, "a.pdf", b"pdf-bytes")
    _write(cli_env, "b.docx", b"docx-bytes")

    assert crypto._cmd_encrypt_all() == 0

    assert (cli_env / "a.pdf.enc").exists()
    assert (cli_env / "b.docx.enc").exists()
    assert (cli_env / "a.pdf.bak").read_bytes() == b"pdf-bytes", "原明文必须留 .bak"
    assert not (cli_env / "a.pdf").exists(), "明文不能留在原位"

    out = capsys.readouterr().out
    assert "2 个成功" in out


def test_encrypt_all_round_trips_through_decrypt_all(cli_env):
    """加密再解密必须逐字节还原 —— 这是用户"文件还能拿回来吗"的底线。"""
    original = "中文内容 with mixed ASCII\n".encode("utf-8")
    _write(cli_env, "a.pdf", original)

    crypto._cmd_encrypt_all()
    assert crypto._cmd_decrypt_all() == 0

    assert (cli_env / "a.pdf").read_bytes() == original, "解密后必须与原文逐字节一致"


def test_encrypt_all_is_a_noop_on_a_directory_without_documents(cli_env, capsys):
    assert crypto._cmd_encrypt_all() == 0
    assert "没有需要加密的明文文档" in capsys.readouterr().out


def test_encrypt_all_is_idempotent_and_retires_the_leftover_plaintext(cli_env, capsys):
    """明文与密文并存且内容一致时，把多余的明文退休成 .bak，而不是报错或重加密。"""
    _write(cli_env, "a.pdf", b"same-bytes")
    crypto._cmd_encrypt_all()
    # 手工把明文放回来（模拟上次加密中途被打断，留下了一份多余明文）
    _write(cli_env, "a.pdf", b"same-bytes")
    capsys.readouterr()

    assert crypto._cmd_encrypt_all() == 0

    out = capsys.readouterr().out
    assert "内容一致，明文已改为 .bak" in out
    assert not (cli_env / "a.pdf").exists()
    assert (cli_env / "a.pdf.bak").read_bytes() == b"same-bytes"


def test_encrypt_all_refuses_to_touch_plaintext_that_differs_from_the_ciphertext(cli_env, capsys):
    """内容不一致说明密文是旧的 —— 绝不能覆盖明文，只能告警交人工，并计入 skipped。"""
    _write(cli_env, "a.pdf.enc", b"stale-ciphertext")
    _write(cli_env, "a.pdf", b"newer-content")

    assert crypto._cmd_encrypt_all() == 0  # 跳过不算失败

    out = capsys.readouterr().out
    assert "内容不一致" in out
    assert "1 个跳过" in out
    assert (cli_env / "a.pdf").read_bytes() == b"newer-content", "明文必须原样保留"


def test_encrypt_all_continues_after_a_single_file_fails(cli_env, monkeypatch, capsys):
    """单个文件坏掉不能中断整批，且必须以退出码 1 报告失败。"""
    _write(cli_env, "good.pdf", b"fine")
    _write(cli_env, "bad.pdf", b"will-fail")

    real_encrypt = crypto.encrypt_file

    def flaky_encrypt(src, key, dest_path=None):
        if os.path.basename(src) == "bad.pdf":
            raise OSError("模拟写入失败")
        return real_encrypt(src, key, dest_path)

    monkeypatch.setattr(crypto, "encrypt_file", flaky_encrypt)
    code = crypto._cmd_encrypt_all()

    out = capsys.readouterr().out
    assert code == 1, "有文件失败时退出码必须是 1"
    assert (cli_env / "good.pdf.enc").exists(), "失败的那个不该影响其他文件"
    assert "1 个失败" in out


def test_decrypt_all_restores_documents_and_keeps_the_ciphertext_as_bak(cli_env, capsys):
    _write(cli_env, "a.pdf", b"content")
    crypto._cmd_encrypt_all()
    capsys.readouterr()

    assert crypto._cmd_decrypt_all() == 0

    assert (cli_env / "a.pdf").read_bytes() == b"content"
    assert (cli_env / "a.pdf.enc.bak").exists(), "密文必须留 .bak，否则误删就再也拿不回来"
    assert "1 个成功" in capsys.readouterr().out


def test_decrypt_all_is_a_noop_when_there_is_nothing_encrypted(cli_env, capsys):
    _write(cli_env, "plain.pdf", b"x")
    assert crypto._cmd_decrypt_all() == 0
    assert (cli_env / "plain.pdf").read_bytes() == b"x"
    assert "没有需要解密的密文文档" in capsys.readouterr().out


# ============================ --verify-all ============================


def test_verify_all_reports_success_for_intact_documents(cli_env, monkeypatch, capsys):
    """索引里有摘要且对得上 → 逐条 ✅，退出码 0。"""
    _write(cli_env, "a.pdf", b"payload")
    crypto._cmd_encrypt_all()
    capsys.readouterr()

    digest = crypto.sm3_hex(b"payload")
    monkeypatch.setattr(
        vector_store, "get_indexed_hashes", lambda vectorstore=None: {digest: "a.pdf"}
    )

    code = crypto._cmd_verify_all()

    out = capsys.readouterr().out
    assert code == 0
    assert "✅ a.pdf.enc" in out
    assert "1 个与索引一致" in out


def test_verify_all_detects_a_corrupted_ciphertext(cli_env, monkeypatch, capsys):
    """密文被改一个字 → 必须报异常、退出码 1，而不是安静地当成正常。

    这是完整性巡检的全部意义所在：密文被改坏只表现为"解密失败"，不报就没人知道。
    """
    _write(cli_env, "a.pdf", b"payload-content")
    crypto._cmd_encrypt_all()
    capsys.readouterr()

    monkeypatch.setattr(
        vector_store, "get_indexed_hashes", lambda vectorstore=None: {"deadbeef": "other.pdf"}
    )
    enc = cli_env / "a.pdf.enc"
    blob = bytearray(enc.read_bytes())
    blob[-1] ^= 0xFF
    enc.write_bytes(bytes(blob))

    code = crypto._cmd_verify_all()

    out = capsys.readouterr().out
    assert code == 1, "发现异常时退出码必须是 1，否则 CI/脚本无法据此告警"
    assert "❌ a.pdf.enc" in out
    assert "无法读取/解密" in out


def test_verify_all_detects_a_document_whose_content_changed(cli_env, monkeypatch, capsys):
    """能解密但 SM3 对不上 → 报"索引里找不到该内容"（文档被改动或尚未入库）。"""
    _write(cli_env, "a.pdf", b"payload")
    crypto._cmd_encrypt_all()
    capsys.readouterr()

    monkeypatch.setattr(
        vector_store, "get_indexed_hashes", lambda vectorstore=None: {"0" * 64: "old.pdf"}
    )

    code = crypto._cmd_verify_all()

    out = capsys.readouterr().out
    assert code == 1
    assert "在索引里找不到该内容" in out
    assert "1 个异常" in out


def test_verify_all_reports_a_corrupted_document_on_a_legacy_index(
    cli_env, monkeypatch, capsys
):
    """老索引（无 sm3_hash 记录）下，损坏的密文**必须被报出来**。

    曾踩过的坑（已修，`app/crypto.py` 的 legacy 分支）：解密失败的文件在循环里
    已经被收进 `problems`（`problems.append(f"❌ {label}：无法读取/解密 —— {exc}")`），
    但老索引分支把 `problems` **整个丢掉** —— 既不打印，也把异常数硬编码成 0，
    还无条件 `return 0`。实测：2 个文档、其中 1 个密文被改坏，命令输出
    `巡检完成：2 个文件，1 个可正常解密，0 个异常。` 并返回 0 —— 一份"全部正常"的
    假报告。而 `--verify-all` 恰恰是用户用来确认密文有没有被改坏的工具：
    它在最该报警的时刻保持了沉默。

    现在的正确行为：老索引只是"无法做内容比对"，不等于"可以忽略读不出来的文件"。
    异常照常计数、照常打印、退出码照常为 1。
    """
    _write(cli_env, "intact.pdf", b"good content")
    _write(cli_env, "corrupt.pdf", b"bad content")
    crypto._cmd_encrypt_all()
    capsys.readouterr()

    # 空索引 = 老索引场景
    monkeypatch.setattr(vector_store, "get_indexed_hashes", lambda vectorstore=None: {})
    enc = cli_env / "corrupt.pdf.enc"
    blob = bytearray(enc.read_bytes())
    blob[-1] ^= 0xFF
    enc.write_bytes(bytes(blob))

    code = crypto._cmd_verify_all()
    out = capsys.readouterr().out

    assert code == 1, "有文档解不开却返回 0"
    assert "corrupt.pdf.enc" in out, "损坏的文档在输出里完全没有出现"




# ============================ main() 分发 ============================


def test_main_dispatches_selftest(cli_env):
    assert crypto.main(["--selftest"]) == 0


def test_main_dispatches_encrypt_all(cli_env):
    _write(cli_env, "a.pdf", b"x")
    assert crypto.main(["--encrypt-all"]) == 0
    assert (cli_env / "a.pdf.enc").exists()


def test_main_requires_exactly_one_command(cli_env):
    """不传命令必须报错退出，而不是默默什么都不做（mutually_exclusive_group required）。"""
    with pytest.raises(SystemExit) as excinfo:
        crypto.main([])
    assert excinfo.value.code != 0


def test_main_rejects_two_commands_at_once(cli_env):
    with pytest.raises(SystemExit) as excinfo:
        crypto.main(["--selftest", "--gen-key"])
    assert excinfo.value.code != 0


# ============================ 与入库层的接缝 ============================


def test_encrypted_documents_are_still_discovered_by_the_ingestion_scanner(cli_env):
    """加密之后入库扫描必须仍能看见这些文档，且逻辑名不带 .enc。

    否则一旦执行加密，知识库就"空"了 —— 这是国密层最致命的接线错误。
    """
    _write(cli_env, "a.pdf", b"pdf-bytes")
    _write(cli_env, "b.docx", b"docx-bytes")
    crypto._cmd_encrypt_all()

    files = ingestion._scan_document_files(str(cli_env))

    assert [os.path.basename(p) for p in files] == ["a.pdf.enc", "b.docx.enc"]
    assert [crypto.logical_name(p) for p in files] == ["a.pdf", "b.docx"]
