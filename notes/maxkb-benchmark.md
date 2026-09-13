# MaxKB 对标分析：企业级 RAG 的成熟做法

> 调研时间：2026-09-11 ｜ 对象：[1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB)（22.7K star，GPL-3.0）
> 源码已 clone 至 `C:\Users\24162\WorkBuddy\2026-09-09-18-30-07\MaxKB-reference`（--depth 1）
> 结论用途：验证并修正 dev-agent 的 RAG 优化清单

---

## 一、MaxKB 技术栈（与我的选型建议一致）

| 层 | MaxKB | dev-agent |
|---|---|---|
| 后端 | Python / Django | FastAPI |
| LLM 框架 | LangChain | LangGraph（LangChain 生态） |
| 存储 | **PostgreSQL + pgvector** | SQLite + Chroma |
| 前端 | Vue.js | Streamlit |

> 选型验证：我之前建议"重做版直接上 Postgres + pgvector"，MaxKB 22K star 的生产系统正是这个组合——**向量与业务元数据同库**，省一个组件。

---

## 二、MaxKB RAG 管线的 5 个关键设计

### 1. 分段：标题树 + parent_chain 前缀（不是固定窗口）

`apps/common/utils/split_model.py`：

- 按 Markdown 标题层级（`#` ~ `######`）把文档解析成**树**
- 扁平化时保留 `parent_chain`，最终 chunk 内容 = **标题路径 + 正文**：
  ```python
  'content': ",".join([p['content'] for p in obj['parent_chain']]) + content
  ```
- 同一父级下的多个标题会**合并成一个块**（`titles_to_paragraph`），标题本身也是可检索单元
- 用 jieba 提取 `keywords` 字段单独存储
- **代码块 mask**：把 ``` 围栏内内容替换成等长空格，防止代码里的 `#` 被误判为 Markdown 标题

### 2. 检索：三模式 + 分数相加融合（不是 RRF）

`SearchMode`: `embedding` / `keywords` / `blend`

blend 模式的核心 SQL（`apps/knowledge/sql/blend_search.sql`）：

```sql
comprehensive_score = (1 - 余弦距离) + ts_rank_cd(search_vector, plainto_tsquery('simple', query), 32)
...
WHERE comprehensive_score > 阈值
ORDER BY comprehensive_score DESC
```

要点：
- **归一化后相加**（余弦相似度 0~1 + PG 全文检索归一化分 0~1），不是 RRF 排名融合
- 全文检索走 **PostgreSQL 原生 `ts_rank_cd`**（`32` = rank/(rank+1) 归一化），不是 Python 里跑 BM25
- **候选池 `LIMIT LEAST(top_k * 10, 500)`** —— 先捞 10 倍候选再排序
- **`WHERE comprehensive_score > 阈值`** 硬过滤

### 3. 相似度阈值 + 高相似直接返回（跳过 LLM）

- 知识库参数：`top_n`（引用分段数）、`similarity`（相识度阈值）
- `HitHandlingMethod`：`optimization`（模型优化）/ `directly_return`（直接返回）
- 文档级配置 `directly_return_similarity` 默认 **0.9**：命中超过 0.9 的段落**直接返回，不调 LLM**
  → 企业场景的三重收益：快（无 LLM 延迟）、省（无 token 成本）、零幻觉

### 4. 关联问题生成（PROBLEM 向量）

- Embedding 表的 `SourceType` 有三类：`PROBLEM`（问题）/ `PARAGRAPH`（段落）/ `TITLE`（标题）
- `apps/knowledge/task/generate.py` 的 `generate_problem_by_paragraph`：让 LLM **为每个段落生成若干个相关问题**，单独嵌入入库
- 检索时"用户问题 ↔ 段落问题"匹配，比"问题 ↔ 陈述句"匹配更准——**这是召回率的杀手锏**

### 5. 文档解析：统一转 Markdown，表格保留结构

docx 解析（`doc_split_handle.py`）把段落和表格都转成 Markdown（`paragraph_to_md` / `table_to_md`），再进入分段流程——表格不会因为按字符切断而丢结构。

---

## 三、对标差距表

| MaxKB 的做法 | dev-agent 现状 | 差距 | 优化优先级 |
|---|---|:---:|:---:|
| 标题树 + parent_chain 前缀 | 固定 500 字窗口，无结构 | 🔴 大 | 高（批次 2） |
| 相似度阈值过滤 | **完全没有** | 🔴 大 | 高（批次 2） |
| 高相似直接返回（跳 LLM） | 无 | 🟡 中 | 中（新增） |
| 分数相加融合 + 大候选池 | RRF + top_k*2，且 BM25 权重=0 | 🔴 大 | 高（批次 2） |
| PG 原生全文检索 | Python BM25（当前被关闭） | 🟡 中 | 中 |
| 关联问题生成 | **完全没有** | 🔴 大 | 高（新增，性价比之王） |
| 表格转 Markdown | 按字符切断 | 🟡 中 | 中 |
| 关键词字段 | 无 | 🟢 小 | 低 |
| 代码块 mask | 无 | 🟢 小 | 低 |

---

## 四、本次对标新增的两项（原清单没有）

### ★ 关联问题生成（建议提到批次 2 首位）

对每个 chunk 让 LLM 生成 2-3 个"这个段落能回答什么问题"，单独建向量索引；检索时同时查段落向量和问题向量，取并集。
- **为什么值得做**：MaxKB 把它作为核心能力，说明是经过生产验证的；且它是**纯增量**——不动现有索引结构，只是多存一类向量
- **成本**：入库时多一次 LLM 调用（可批量、可只对重要文档做）；检索时多一次向量查询
- **验证**：Recall@5 应有明显提升

### ★ 高相似直接返回（省成本 + 零幻觉）

`similarity > 0.9` 时跳过生成，直接把段落原文返回（或简单包装）。
- 实现简单（一个 if 分支），但对"企业场景成本治理"是很有说服力的叙事
- 与现有的语义缓存思路一致，属于同一类"能不调 LLM 就不调"的优化

---

## 五、更新后的 RAG 优化优先级

| 批次 | 事项 | 来源 |
|:---:|------|------|
| 0 | 建固定测试集 + Recall@K / MRR | 前提 |
| 1 | bge query 指令前缀 · 关 BM25 白算 · .dockerignore · 清个人路径 | 低成本速赢 |
| 2 | **关联问题生成** ← 新增 | 对标 MaxKB |
| 2 | 数据清洗（断行合并 / 页眉页脚 / 归一化） | 现状分析 |
| 2 | 混合检索修复：改**分数相加融合** + 扩大候选池到 top_k*10 | 对标 MaxKB |
| 2 | 相似度阈值 + **高相似直接返回** ← 新增 | 对标 MaxKB |
| 2 | reranker（对大候选池重排） | 通用最佳实践 |
| 3 | 分块升级：标题树 + parent_chain 前缀（需重建索引） | 对标 MaxKB |
| 4 | 安全与 API（认证 / work_dir 收权 / 超时重试） | 企业级 P0 |
| 5 | 工程骨架（配置 / 规范化 / 分层 / CI） | 企业级 P1 |

---

## 六、一个反直觉的发现

MaxKB **没有用 RRF**，而是用"归一化分数相加"。RRF 的好处是不需要关心两路分数的量纲，但代价是**丢掉了绝对分数信息**——而绝对分数正是阈值过滤的基础。

dev-agent 现在用 RRF（`k=60`），如果将来要加"相似度阈值拒答"，RRF 的分数（1/(60+rank) ≈ 0.016）没有绝对意义，阈值无从设定。**所以：加阈值和用 RRF 是冲突的**，改成分数相加融合更合理（顺带对齐 MaxKB 的生产验证做法）。
