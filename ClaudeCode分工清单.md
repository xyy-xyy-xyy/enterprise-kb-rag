# Claude Code 任务分工清单

> 用途：明确 enterprise-kb-rag 这个项目里，哪些活可以甩给 Claude Code 干，
> 哪些必须留在 WorkBuddy / 自己手里。每个可委托任务都附带「直接复制给 Claude Code 的提示词」。

---

## 一、怎么判断一件事能不能给 Claude Code

| 适合给 Claude Code | 不适合给 Claude Code |
|---|---|
| 有详细伪代码、边界清晰 | 架构选型 / 路线图规划 |
| 纯代码实现、改固定几个文件 | 需要外部账号 / 私有 API Key 才能验证 |
| 写测试、写样板、补 CRUD | 找素材（如演示 PDF）、面试话术包装 |
| 已有的明确算法（Reranker / BM25） | 需要你拍板的"为什么"（如为什么用 SM3 不用 MD5） |

**一句话**：Claude Code 是「你画好施工图，它来砌墙」。凡是还要你做决定的，别丢给它。

---

## 二、推荐分工（对照 `路线图进度.md`）

### ✅ 适合交给 Claude Code

| 路线图 | 任务 | 给 CC 的注意点 |
|---|---|---|
| 2.1 | 流式输出 | 伪代码完整，改 `retrieval.py` + `main.py`，低风险 |
| 2.2 | BM25 混合检索 | ⚠️ **必须附带下方两个避坑**，否则它会复刻 bug |
| 2.3 | BGE-Reranker | 模型 2GB，提示它「先走 DashScope `gte-rerank` API，别急着下本地模型」 |
| 2.4 | 多轮对话 | 用 `ConversationBufferMemory` 或手动拼 history |
| 2.5 | 查询改写 | 调 LLM 把口语问题改成关键词查询 |
| 2.8 | 文档管理 API | GET 列表 + DELETE 删除 + 重建索引 + Gradio Tab |
| 3.1~3.5 | 国密安全层 | `crypto.py` + `ingestion.py` 改造，规格清晰，CC 能独立实现 |
| 4.2~4.5 | 评测脚本 | `app/eval/` 目录，Hit Rate / MRR / LLM-judge |
| 5.3 | 单元测试 | pytest，覆盖率目标 ≥70% |
| 5.4 | CI/CD | `.github/workflows/ci.yml` |
| 5.5 | Docker 部署 | `Dockerfile` + `docker-compose.yml`（Qdrant 服务 + app 服务） |

### ❌ 留在 WorkBuddy / 自己干

- **路线图规划、架构选型**：这是你的脑子活，WorkBuddy 帮你调研对比，你拍板。
- **补演示 PDF**：需要你自己的素材（现在 `data/docs/` 只有 1~2 份，评测前必须补到 5~8 份）。
- **面试讲解稿 / 简历措辞**：得基于你真实做的过程写，CC 编不出来。
- **「为什么用 SM3 不用 MD5」这类解释性决策**：你先定方向，再让 CC 实现。
- **需要你私有 API Key 才能跑通验证的事**：CC 在你机器上能跑，但别把 Key 贴进对话。

---

## 三、给 Claude Code 的通用背景提示词（每次开头都贴这段）

```
你正在协助开发一个企业知识库 RAG 项目 enterprise-kb-rag。
技术栈：Python 3.13 + LangChain 1.4 + FastAPI + Gradio 6。
向量库是【双后端】：FAISS（本地文件）和 Qdrant（Docker 服务），
通过 .env 的 VECTOR_BACKEND=faiss|qdrant 切换，代码在 app/vector_store.py 已抽象好。
Embedding 用 DashScope text-embedding-v2，大模型用通义千问 qwen-plus（ChatTongyi）。
PDF 解析用 PyMuPDF，分块用 RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)。
代码约定：
- 所有相对路径以项目根目录为基准（见 app/config.py 的 resolve_path）。
- 新增函数要保持和现有模块一致的日志与异常处理风格。
- 不要破坏已有的双后端抽象，新检索逻辑要同时兼容 FAISS 和 Qdrant。
改动完告诉我改了哪些文件、怎么本地验证。
```

---

## 四、两个必须写进提示词的「坑」（否则 Claude Code 会复刻 bug）

**坑 1：BM25 中文分词**
路线图 2.2 伪代码里 `doc.page_content.split()` 和 `query.split()` 是按空格切的，
中文没有空格，会切成一坨。必须让 CC 用 `jieba` 分词：
```
BM25 语料构建和查询分词都要用 jieba.lcut()，不要直接用 str.split()。
（先 pip install jieba）
```

**坑 2：RRF 融合的 key 会撞车**
路线图 2.2 用 `source + page` 当融合 key。但同一页 PDF 会被切成多个 chunk，
它们的 source+page 完全相同，分数会累加到一个 key 上，最后只捞回一个 chunk。
必须让 CC 给每个 chunk 发唯一 id：
```
入库时为每个 chunk 生成 chunk_id = f"{file_name}_{page}_{序号}" 存入 metadata，
RRF 融合用 chunk_id 当 key，不要用 source+page。
```

---

## 五、每个任务的「复制即用」提示词

### 任务 2.2 混合检索（带避坑）
```
参考 路线图进度.md 的 2.2 节，在 app/vector_store.py 新增 hybrid_search(query, k)，
实现向量检索 + BM25 关键词检索，用 RRF（常数 60）融合。
要求：
1. BM25 语料和查询都用 jieba.lcut() 分词，不要 str.split()。
2. 融合 key 用 metadata 里的 chunk_id（每个 chunk 唯一），不要用 source+page。
3. 在 app/retrieval.py 的 answer_question 里把 vector_store.search 换成 hybrid_search，
   由 .env 的 HYBRANK_SEARCH 开关控制（新增 config.HYBRID_SEARCH）。
4. 保持 FAISS 和 Qdrant 双后端都可用。
```

### 任务 3.1~3.4 国密层（含去重）
```
参考 路线图进度.md 阶段三，新建 app/crypto.py（gmssl 库），
实现 encrypt_sm4 / decrypt_sm4 / hash_sm3 / generate_key。
然后在 app/ingestion.py 的入库流程里：
1. 算文件内容的 SM3 哈希存进 metadata["content_hash"]（用于完整性校验）；
2. 用 content_hash 做【内容去重】——同一份文件改名重传也应被识别为重复（
   现在的去重是按归一化路径 normalize_source，改名会绕过，要改成按 SM3 内容哈希判断）。
3. SM4 文档加密存储（.enc 后缀）放到 3.2，先不做也可以，但去重 + SM3 校验先完成。
```

### 任务 4.2~4.4 评测
```
参考 路线图进度.md 阶段四，新建 app/eval/ 目录：
retrieval_eval.py 算 Hit Rate 和 MRR（k=5），answer_eval.py 用 ChatTongyi 当 judge 打分，
run_eval.py 跑 4 种方案（纯向量 / 纯 BM25 / 混合 / 混合+Reranker）对比并出 Markdown 表格。
评测数据集先用 data/eval/qa_pairs.json（需先有 50 个 QA 对，这部分我自己造）。
```

---

## 六、协作节奏建议

1. **WorkBuddy 先定方向 + 写架构契约 + 调研对标**（已完成 GitHub 调研、路线图评审、双后端实测）。
2. **Claude Code 按上面的提示词批量实现阶段二 / 三 / 四的代码**。
3. **你自己在每个任务完成后跑一遍验证**（启动服务 → 提问 → 看结果），把报错贴回 WorkBuddy 或 Claude Code。
4. **WorkBuddy 最后做面试包装、README 维护、推 GitHub**。
