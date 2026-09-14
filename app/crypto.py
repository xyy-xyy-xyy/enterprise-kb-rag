"""国密安全层：SM4-CBC 加解密 + SM3 摘要 + 密钥管理。

只用 `cryptography`（已在 requirements.txt），不引入 gmssl：本机实测
`algorithms.SM4` 与 `hashlib.new("sm3", ...)` 均可用，且 SM3 结果与官方测试向量一致。

一份 SM3 摘要同时承担两件事：
  - 内容去重：相同内容换个文件名上传也能识别（比按路径去重更强）
  - 完整性校验：解密后重算摘要与索引比对，可发现文档被改动过

密钥优先级：`.env` 的 `SM4_KEY` > 密钥文件 > 自动生成（打 WARN）。
"""

import hashlib
import logging
import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from app import config

logger = logging.getLogger(__name__)

# SM4 的分组长度与密钥长度都是 16 字节
KEY_SIZE = 16
IV_SIZE = 16

# 密文文件后缀：磁盘上是 xxx.pdf.enc，逻辑名仍是 xxx.pdf
ENC_SUFFIX = ".enc"

# 加密前把原明文改名保留，绝不删除：中途失败也不能丢原始文档
_BAK_SUFFIX = ".bak"

_PKCS7 = PKCS7(128)  # 128 bit = 16 字节分组


# ============================ SM3 摘要 ============================


def sm3_hex(data: bytes) -> str:
    """计算 SM3 摘要，返回 64 位十六进制字符串。

    必须对**明文**计算：密文带随机 IV，同一份文档每次加密结果都不同，
    对密文算摘要没有去重意义。所以调用方要在解密之后、解析之前调用。
    """
    return hashlib.new("sm3", data).hexdigest()


# ============================ 密钥管理 ============================


def generate_sm4_key() -> bytes:
    """生成 16 字节随机密钥。"""
    return os.urandom(KEY_SIZE)


def _parse_key(text: str) -> bytes | None:
    """把十六进制密钥文本解析成 16 字节；不合法返回 None（只告警，不回显内容）。

    注意不要把密钥本身写进日志。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        key = bytes.fromhex(cleaned)
    except ValueError:
        logger.warning("SM4 密钥不是合法的十六进制字符串，已忽略该来源。")
        return None
    if len(key) != KEY_SIZE:
        logger.warning(
            "SM4 密钥长度应为 %d 字节（%d 位 hex），实际 %d 字节，已忽略该来源。",
            KEY_SIZE,
            KEY_SIZE * 2,
            len(key),
        )
        return None
    return key


def _write_key_file(path: str, key: bytes) -> None:
    """把密钥写成十六进制文本，并尽量收紧文件权限。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(key.hex())
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows 下 chmod 基本无效，不作为错误
        logger.debug("无法设置密钥文件权限（Windows 下属正常）。")


def load_or_create_key() -> bytes:
    """按优先级取 SM4 密钥：环境变量 > 密钥文件 > 自动生成并落盘。

    都不存在时不硬报错：本机已有明文文档在跑，缺密钥直接抛异常会打断演示。
    但也绝不静默通过 —— 自动生成必须打 WARN 提醒这是开发模式。
    """
    key = _parse_key(config.SM4_KEY)
    if key:
        logger.debug("使用环境变量 SM4_KEY 提供的密钥。")
        return key

    key_path = config.SM4_KEY_PATH
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            key = _parse_key(f.read())
        if key:
            logger.debug("使用密钥文件 %s 中的密钥。", key_path)
            return key
        logger.warning("密钥文件 %s 内容不可用，将重新生成。", key_path)

    key = generate_sm4_key()
    _write_key_file(key_path, key)
    logger.warning(
        "开发模式：未配置 SM4_KEY，已自动生成开发密钥并写入 %s。"
        "生产环境请通过环境变量注入，且不要把密钥提交进仓库。",
        key_path,
    )
    return key


# ============================ SM4 加解密 ============================


def encrypt_sm4(plaintext: bytes, key: bytes) -> bytes:
    """SM4-CBC 加密，返回 `IV(16 字节) || 密文`。

    IV 每次调用都重新随机生成。**固定 IV 或 ECB 会让相同明文产出相同密文，
    等于没加密**，而且代码照样跑得通、不报错，是最容易被忽略的坑。
    """
    iv = os.urandom(IV_SIZE)
    padder = _PKCS7.padder()
    padded = padder.update(plaintext) + padder.finalize()
    encryptor = Cipher(algorithms.SM4(key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return iv + ciphertext


def decrypt_sm4(blob: bytes, key: bytes) -> bytes:
    """还原 `encrypt_sm4` 的结果；密文被截断、密钥不对或填充被破坏时抛 ValueError。"""
    if len(blob) < IV_SIZE + KEY_SIZE or (len(blob) - IV_SIZE) % KEY_SIZE:
        raise ValueError("密文长度不合法，文件可能已损坏或被截断。")

    iv, ciphertext = blob[:IV_SIZE], blob[IV_SIZE:]
    decryptor = Cipher(algorithms.SM4(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = _PKCS7.unpadder()
    try:
        return unpadder.update(padded) + unpadder.finalize()
    except ValueError as exc:
        # 填充校验失败：密钥不匹配或密文被改过（SM4-CBC 自身不带 MAC，靠这层兜底）
        raise ValueError("解密失败：密钥不匹配或密文已被篡改。") from exc


def _read_bytes(path: str) -> bytes:
    """读取文件原始字节。"""
    with open(path, "rb") as f:
        return f.read()


def encrypt_file(src_path: str, key: bytes, dest_path: str = None) -> str:
    """把明文文件加密成 `.enc`，返回密文路径。**不删除原明文**（由调用方决定）。"""
    dest_path = dest_path or src_path + ENC_SUFFIX
    blob = encrypt_sm4(_read_bytes(src_path), key)
    with open(dest_path, "wb") as f:
        f.write(blob)
    logger.info(
        "已加密：%s → %s（%d 字节）",
        os.path.basename(src_path),
        os.path.basename(dest_path),
        len(blob),
    )
    return dest_path


def decrypt_file_to_bytes(enc_path: str, key: bytes) -> bytes:
    """解密文件并**只返回内存 bytes**。

    刻意不提供"解密到临时文件"的接口：明文一旦落盘就可能被恢复，
    解析统一走内存（`pymupdf.open(stream=...)` / `docx.Document(BytesIO(...))`）。
    """
    return decrypt_sm4(_read_bytes(enc_path), key)


# ============================ 逻辑路径 ============================


def is_encrypted(path: str) -> bool:
    """路径是否指向密文文件。"""
    return path.lower().endswith(ENC_SUFFIX)


def _strip_enc(path: str) -> str:
    """剥掉 `.enc` 后缀（大小写不敏感）。"""
    return path[: -len(ENC_SUFFIX)] if is_encrypted(path) else path


def logical_name(path: str) -> str:
    """逻辑文件名：剥掉 `.enc`。

    必须是逻辑名：否则同一份文档的明文版与密文版会被判成两个文档，
    界面上也会显示 `xxx.pdf.enc`，引用溯源全乱。
    """
    return os.path.basename(_strip_enc(path))


def logical_source(path: str) -> str:
    """逻辑 source：剥掉 `.enc` 之后再归一化。

    与 `logical_name` 同理 —— source 带上 `.enc` 会导致重复入库。
    """
    # 延迟导入：ingestion 依赖本模块，模块级导入会形成循环
    from app.ingestion import normalize_source

    return normalize_source(_strip_enc(path))


# ============================ 命令行工具 ============================

_SM3_ABC_VECTOR = "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"


def _check(label: str, passed: bool, detail: str = "") -> bool:
    """打印一条自测结果，返回是否通过。"""
    print(f"[{'PASS' if passed else 'FAIL'}] {label}{detail}")
    return passed


def _cmd_selftest() -> int:
    """SM3 官方向量 + SM4 往返 + IV 随机性断言。"""
    results = []

    actual = sm3_hex(b"abc")
    results.append(
        _check("SM3('abc') 匹配官方测试向量", actual == _SM3_ABC_VECTOR, f" → {actual}")
    )

    digest = sm3_hex("中文内容".encode("utf-8"))
    results.append(
        _check("SM3 输出为 64 位 hex", len(digest) == 64 and all(c in "0123456789abcdef" for c in digest))
    )

    key = generate_sm4_key()
    samples = [b"", b"a", "国密 SM4 加密测试".encode("utf-8"), os.urandom(1000)]
    roundtrip = all(decrypt_sm4(encrypt_sm4(s, key), key) == s for s in samples)
    results.append(_check("SM4-CBC 加解密往返（空/单字节/中文/1KB 随机）", roundtrip))

    blob1, blob2 = encrypt_sm4(b"same plaintext", key), encrypt_sm4(b"same plaintext", key)
    results.append(
        _check(
            "同一明文两次加密密文不同（IV 随机）",
            blob1 != blob2 and blob1[:IV_SIZE] != blob2[:IV_SIZE],
        )
    )

    expected_len = IV_SIZE + KEY_SIZE * ((len(b"x" * 100) // KEY_SIZE) + 1)
    results.append(
        _check(
            "密文长度 = IV + 明文向上取整到 16 的倍数",
            len(encrypt_sm4(b"x" * 100, key)) == expected_len,
        )
    )

    # 换密钥后：CBC 的填充校验有约 1/256 概率碰巧通过，所以不能断言"必定抛异常"，
    # 只能断言"要么抛异常，要么解出的不是原文"
    original = b"secret payload for wrong-key check"
    try:
        recovered = decrypt_sm4(encrypt_sm4(original, key), generate_sm4_key())
        wrong_key_ok = recovered != original
    except ValueError:
        wrong_key_ok = True
    results.append(_check("换密钥无法还原原文（抛异常或解出乱码）", wrong_key_ok))

    results.append(
        _check(
            "logical_name/logical_source 能剥掉 .enc",
            logical_name("/tmp/飞行社指南.pdf.enc") == "飞行社指南.pdf"
            and is_encrypted("a.pdf.enc")
            and not is_encrypted("a.pdf"),
        )
    )

    passed, total = sum(results), len(results)
    print(f"\n自测结果：{passed}/{total} 通过")
    return 0 if passed == total else 1


def _cmd_gen_key() -> int:
    """生成密钥写入密钥文件并打印 hex（覆盖已有文件前会提示）。"""
    key_path = config.SM4_KEY_PATH
    if os.path.exists(key_path) and not config.SM4_KEY:
        print(f"⚠️ {key_path} 已存在，将被覆盖。")
    key = generate_sm4_key()
    _write_key_file(key_path, key)
    print(f"已写入 {key_path}")
    print(f"SM4_KEY={key.hex()}")
    print("请勿把该值提交进仓库；生产环境请通过环境变量注入。")
    return 0


def _retire_plaintext(src: str) -> str:
    """把明文改名成 `.bak` 保留（绝不删除），返回备份路径。"""
    bak = src + _BAK_SUFFIX
    os.replace(src, bak)
    return bak


def _list_plaintext_files(dir_path: str) -> list[str]:
    """列出目录下的**全部**明文文档，包括与密文同名共存的那些。

    不能用 `ingestion._scan_document_files`：那是"有效语料"视图，同名密文存在时
    会把明文过滤掉；而 `--encrypt-all` 恰恰要看到它，否则会留下一份没人注意的明文。
    """
    from app import ingestion  # 延迟导入：避免循环依赖

    return sorted(
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if name.lower().endswith(ingestion.SUPPORTED_EXTENSIONS)
    )


def _cmd_encrypt_all() -> int:
    """把 `data/docs/` 下的明文文档加密成 `.enc`，原明文改名 `.bak` 保留。"""
    key = load_or_create_key()
    plaintext_files = _list_plaintext_files(config.DOCS_PATH)
    if not plaintext_files:
        print("没有需要加密的明文文档。")
        return 0

    succeeded, skipped, failed = 0, 0, 0
    for src in plaintext_files:
        enc = src + ENC_SUFFIX
        if os.path.exists(enc):
            # 明文与密文并存：内容一致就把多余的明文退休成 .bak，收尾；
            # 内容不一致说明密文是旧的，绝不能动明文，只告警交人工处理
            try:
                same = decrypt_file_to_bytes(enc, key) == _read_bytes(src)
            except Exception:
                same = False
            if same:
                _retire_plaintext(src)
                succeeded += 1
                print(f"🔒 {os.path.basename(src)}：密文已存在且内容一致，明文已改为 .bak")
            else:
                logger.warning(
                    "明文 %s 与已有密文内容不一致，未做任何改动，请人工确认后处理。", src
                )
                print(
                    f"⚠️  {os.path.basename(src)}：与已有密文内容不一致，"
                    "明文保持原样，请人工确认（本次未加密该文件）"
                )
                skipped += 1
            continue
        try:
            encrypt_file(src, key)
        except Exception as exc:  # 单个文件失败不中断整批
            logger.exception("加密失败：%s", src)
            print(f"❌ {os.path.basename(src)}：{exc}")
            failed += 1
            continue
        _retire_plaintext(src)
        succeeded += 1
        print(f"✅ {os.path.basename(src)} → {os.path.basename(enc)}（原明文保留为 .bak）")

    print(
        f"\n加密完成：{succeeded} 个成功，{skipped} 个跳过（需人工确认），{failed} 个失败。"
    )
    return 1 if failed else 0


def _cmd_decrypt_all() -> int:
    """把 `data/docs/` 下的 `.enc` 解密回明文，密文改名 `.bak` 保留。"""
    from app import ingestion  # 延迟导入：避免循环依赖

    key = load_or_create_key()
    files = [p for p in ingestion._scan_document_files(config.DOCS_PATH) if is_encrypted(p)]
    if not files:
        print("没有需要解密的密文文档。")
        return 0

    succeeded, skipped, failed = 0, 0, 0
    for enc in files:
        dest = _strip_enc(enc)
        if os.path.exists(dest):
            print(f"⏭️  跳过（明文已存在）：{os.path.basename(dest)}")
            skipped += 1
            continue
        try:
            data = decrypt_file_to_bytes(enc, key)
            with open(dest, "wb") as f:
                f.write(data)
        except Exception as exc:
            logger.exception("解密失败：%s", enc)
            print(f"❌ {os.path.basename(enc)}：{exc}")
            failed += 1
            continue
        os.replace(enc, enc + _BAK_SUFFIX)
        succeeded += 1
        print(f"✅ {os.path.basename(enc)} → {os.path.basename(dest)}（密文保留为 .bak）")

    print(f"\n解密完成：{succeeded} 个成功，{skipped} 个跳过，{failed} 个失败。")
    return 1 if failed else 0


def _cmd_verify_all() -> int:
    """逐个解密并重算 SM3，与索引里的摘要比对，打印不一致清单。

    密文被改坏只表现为"解密失败"，不会崩溃 —— 这一条本身就是被测项。
    """
    from app import ingestion, vector_store  # 延迟导入：避免循环依赖

    key = load_or_create_key()
    try:
        indexed = vector_store.get_indexed_hashes()
    except Exception as exc:
        print(f"⚠️ 无法读取索引（{exc}），本次只能校验文件可解密性。")
        indexed = {}

    files = ingestion._scan_document_files(config.DOCS_PATH)
    if not files:
        print(f"{config.DOCS_PATH} 下没有找到文档。")
        return 0

    # 索引里一条 sm3_hash 都没有，说明它是加国密之前建的（老索引不带该字段）。
    # 此时"比对不上"是必然的，不是文档被改坏 —— 必须把这两种情况分开说，
    # 否则用户看到满屏 ⚠️ 会以为自己的文档全坏了。
    legacy_index = not indexed
    if legacy_index:
        print(
            "ℹ️  当前索引中没有任何 sm3_hash 记录，说明该索引建立于国密层之前。\n"
            "    本次只能校验文件能否正常解密，无法做内容完整性比对。\n"
            "    需要比对时请重建索引（python -m app.ingestion --rebuild，"
            "会重新向量化全部文档、耗时且产生 API 费用）。\n"
        )

    ok_count, problems = 0, []
    for path in files:
        name = os.path.basename(path)
        label = name if is_encrypted(path) else f"{name}（明文）"
        try:
            if is_encrypted(path):
                data = decrypt_file_to_bytes(path, key)
            else:
                with open(path, "rb") as f:
                    data = f.read()
        except Exception as exc:
            problems.append(f"❌ {label}：无法读取/解密 —— {exc}")
            continue

        digest = sm3_hex(data)
        owner = indexed.get(digest)
        if owner:
            ok_count += 1
            print(f"✅ {label}：SM3 {digest[:16]}… 与索引一致（{owner}）")
        elif legacy_index:
            # 老索引：能正常解密就算通过，不做内容比对，也不计入异常
            ok_count += 1
            print(f"✅ {label}：可正常解密（SM3 {digest[:16]}…，老索引无摘要可比对）")
        else:
            problems.append(
                f"⚠️  {label}：SM3 {digest[:16]}… 在索引里找不到该内容"
                "（文档可能被改动，或尚未入库）"
            )

    if legacy_index:
        # ⚠️ 异常数不能硬编码成 0：解密失败的文件在循环里已经进了 problems，
        # 写死 0 会让 --verify-all 在文件损坏时报告"全部正常"——而它存在的意义
        # 恰恰是发现损坏。异常条目也要逐条打印，否则用户看不到是哪个文件坏了。
        print(f"\n巡检完成：{len(files)} 个文件，{ok_count} 个可正常解密，{len(problems)} 个异常。")
        for line in problems:
            print(line)
        print("（老索引无 sm3_hash 记录，本次未做内容完整性比对）")
        return 1 if problems else 0

    print(f"\n巡检完成：{len(files)} 个文件，{ok_count} 个与索引一致，{len(problems)} 个异常。")
    for line in problems:
        print(line)
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口：python -m app.crypto <命令>。"""
    import argparse

    config.setup_logging()
    parser = argparse.ArgumentParser(
        prog="python -m app.crypto", description="国密安全层工具（SM4 加解密 / SM3 校验）"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true", help="跑 SM3/SM4 自测")
    group.add_argument("--gen-key", action="store_true", help="生成 SM4 密钥并写入密钥文件")
    group.add_argument(
        "--encrypt-all", action="store_true", help="data/docs/ 明文 → .enc（明文保留为 .bak）"
    )
    group.add_argument(
        "--decrypt-all", action="store_true", help="data/docs/ .enc → 明文（密文保留为 .bak）"
    )
    group.add_argument("--verify-all", action="store_true", help="解密重算 SM3 并与索引比对")
    args = parser.parse_args(argv)

    if args.selftest:
        return _cmd_selftest()
    if args.gen_key:
        return _cmd_gen_key()
    if args.encrypt_all:
        return _cmd_encrypt_all()
    if args.decrypt_all:
        return _cmd_decrypt_all()
    return _cmd_verify_all()


if __name__ == "__main__":
    raise SystemExit(main())
