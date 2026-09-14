"""国密安全层单元测试。

纯本地：不需要 DashScope API key，也不需要 FAISS/Qdrant 索引。
运行：`python -m pytest tests/test_crypto.py -q`（需在项目根目录执行）
"""

import os

import pytest

from app import crypto

# SM3 官方测试向量（GB/T 32905-2016 附录 A）
SM3_ABC = "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"


@pytest.fixture
def key() -> bytes:
    """每个用例一把新密钥，避免用例之间互相影响。"""
    return crypto.generate_sm4_key()


class TestSM3:
    def test_official_vector(self):
        assert crypto.sm3_hex(b"abc") == SM3_ABC

    def test_empty_input(self):
        # 空输入也应有合法摘要，不能抛异常
        digest = crypto.sm3_hex(b"")
        assert len(digest) == 64

    def test_output_is_64_hex_chars(self):
        digest = crypto.sm3_hex("中文内容 with ASCII".encode("utf-8"))
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_different_input_differs(self):
        assert crypto.sm3_hex(b"a") != crypto.sm3_hex(b"b")

    def test_same_input_is_stable(self):
        data = "同样内容".encode("utf-8")
        assert crypto.sm3_hex(data) == crypto.sm3_hex(data)


class TestKey:
    def test_generate_key_length(self):
        assert len(crypto.generate_sm4_key()) == crypto.KEY_SIZE

    def test_two_keys_differ(self):
        assert crypto.generate_sm4_key() != crypto.generate_sm4_key()

    @pytest.mark.parametrize(
        "text,expected_valid",
        [
            ("", False),
            ("   ", False),
            ("nothex!!", False),
            ("ab" * 16, True),  # 32 位 hex = 16 字节
            ("AB" * 16, True),  # 大写也接受
            ("ab" * 15, False),  # 30 位 hex = 15 字节，长度不对
            ("ab" * 17, False),  # 34 位 hex = 17 字节，长度不对
        ],
    )
    def test_parse_key(self, text, expected_valid):
        assert (crypto._parse_key(text) is not None) is expected_valid


class TestSM4:
    @pytest.mark.parametrize(
        "plaintext",
        [
            b"",  # 空
            b"a",  # 单字节
            b"\x00" * 16,  # 恰好一个分组，触发整块填充
            "国密 SM4 加密测试：中文与符号「」".encode("utf-8"),
            os.urandom(1000),  # 超长随机数据
        ],
        ids=["empty", "single", "exact-block", "chinese", "random-1kb"],
    )
    def test_roundtrip(self, key, plaintext):
        assert crypto.decrypt_sm4(crypto.encrypt_sm4(plaintext, key), key) == plaintext

    def test_same_plaintext_differs_each_time(self, key):
        """最重要的一条：IV 随机，两次加密同一明文必须得到不同密文。

        若这里失败，说明 IV 被固定或用了 ECB —— 等于没加密。
        """
        first = crypto.encrypt_sm4(b"same plaintext", key)
        second = crypto.encrypt_sm4(b"same plaintext", key)
        assert first != second
        assert first[: crypto.IV_SIZE] != second[: crypto.IV_SIZE]
        # 但两者都能正确解回原文
        assert crypto.decrypt_sm4(first, key) == b"same plaintext"
        assert crypto.decrypt_sm4(second, key) == b"same plaintext"

    @pytest.mark.parametrize("size", [0, 1, 15, 16, 17, 100])
    def test_ciphertext_length(self, key, size):
        """密文长度 = IV(16) + 明文向上取整到 16 的倍数。"""
        blob = crypto.encrypt_sm4(b"x" * size, key)
        blocks = size // 16 + 1  # PKCS7 至少补一组
        assert len(blob) == crypto.IV_SIZE + blocks * crypto.KEY_SIZE

    def test_tampered_ciphertext_not_silently_wrong(self, key):
        """篡改密文首字节：要么抛异常，要么解出与原文不同的内容，不能静默返回原文。"""
        original = b"hello world, this is a longer payload" * 3
        blob = bytearray(crypto.encrypt_sm4(original, key))
        blob[crypto.IV_SIZE] ^= 0xFF  # 改密文第一个字节

        try:
            recovered = crypto.decrypt_sm4(bytes(blob), key)
        except ValueError:
            return  # 填充校验失败，已检出
        assert recovered != original

    def test_tampered_iv_not_silently_wrong(self, key):
        original = b"payload" * 8
        blob = bytearray(crypto.encrypt_sm4(original, key))
        blob[0] ^= 0xFF  # 改 IV 第一个字节
        try:
            recovered = crypto.decrypt_sm4(bytes(blob), key)
        except ValueError:
            return
        assert recovered != original

    def test_truncated_ciphertext_raises(self, key):
        blob = crypto.encrypt_sm4(b"x" * 32, key)
        with pytest.raises(ValueError):
            crypto.decrypt_sm4(blob[:20], key)

    def test_wrong_key_does_not_recover_plaintext(self, key):
        """换密钥：CBC 填充校验有约 1/256 概率碰巧通过，故只断言"解不出原文"。"""
        original = b"secret payload" * 4
        blob = crypto.encrypt_sm4(original, key)
        try:
            assert crypto.decrypt_sm4(blob, crypto.generate_sm4_key()) != original
        except ValueError:
            pass


class TestLogicalPath:
    def test_is_encrypted(self):
        assert crypto.is_encrypted("a.pdf.enc")
        assert crypto.is_encrypted("A.PDF.ENC")
        assert not crypto.is_encrypted("a.pdf")

    def test_logical_name_strips_enc(self):
        assert crypto.logical_name("/tmp/飞行社指南.pdf.enc") == "飞行社指南.pdf"
        assert crypto.logical_name("/tmp/飞行社指南.pdf") == "飞行社指南.pdf"

    def test_logical_source_strips_enc(self, tmp_path):
        """逻辑 source 必须剥掉 .enc，否则明文版与密文版会被当成两个文档。"""
        encrypted = crypto.logical_source(str(tmp_path / "a.pdf.enc"))
        plaintext = crypto.logical_source(str(tmp_path / "a.pdf"))
        assert encrypted == plaintext
        assert not encrypted.endswith(crypto.ENC_SUFFIX)

    def test_logical_source_is_absolute(self, tmp_path):
        assert os.path.isabs(crypto.logical_source(str(tmp_path / "a.pdf.enc")))


class TestFileHelpers:
    def test_encrypt_then_decrypt_file(self, tmp_path, key):
        payload = "文档正文内容".encode("utf-8") * 50
        src = tmp_path / "doc.pdf"
        src.write_bytes(payload)

        enc_path = crypto.encrypt_file(str(src), key)
        assert enc_path.endswith(crypto.ENC_SUFFIX)
        assert open(enc_path, "rb").read() != payload  # 磁盘上确实不是明文
        assert crypto.decrypt_file_to_bytes(enc_path, key) == payload

    def test_encrypt_file_keeps_original(self, tmp_path, key):
        """encrypt_file 本身不删明文（删不删由调用方决定）。"""
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"data")
        crypto.encrypt_file(str(src), key)
        assert src.exists()

    def test_default_dest_suffix(self, tmp_path, key):
        src = tmp_path / "doc.docx"
        src.write_bytes(b"data")
        assert crypto.encrypt_file(str(src), key) == str(src) + crypto.ENC_SUFFIX
