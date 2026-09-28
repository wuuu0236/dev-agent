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
