# DataLens 工程审查（开发者视角）

审查日期：2026-09-14
审查范围：仓库全量（README / src / pages / scripts / tests / 部署配置）
审查标准：一个开发者接手这个项目时，会怎么评价它

---

## 结论先行

**功能是真的，迭代是真的，缺的是最后一道"收敛"工序。**

这个项目已经走完了"把东西做出来"的阶段，但没有走"把它做成一个别人能接手的东西"的阶段。具体表现为三类问题：

1. **文档描述的项目 ≠ 代码里的项目**——README 在说一件不存在的事
2. **生产代码里留着学习过程的痕迹**——教程式注释、版本编号、教学 print
3. **工程规范基本为零**——无 lint/format 配置，核心链路用 print 不用 logging

前两类是"信任问题"：面试官看完 README 再看代码，会立刻降低对这个仓库的信任。
第三类是"专业度问题"：不致命，但会被认为没在真实工程里工作过。

**优先级判断：修文档 > 清痕迹 > 补规范 > 补测试。** 因为前三类的修复成本是分钟级，而收益是"第一印象"级别的。

---

## 一、文档与代码脱节（最伤信任）

### 1.1 README 三处宣称"线上版本是 LangGraph"——不成立

| 位置 | 原文 |
|---|---|
| README:11 | 「当前线上版本为 LangGraph StateGraph」 |
| README:19 | 「v4 LangGraph StateGraph（当前线上版本）。早期版本已归档」 |
| README:31 | 「## 架构（v4 LangGraph StateGraph）」 |

**实际链路**：

- README 自己挂的在线演示链接是 Streamlit 应用
- Streamlit 问答页 `pages/3_智能问答.py:6` → `from src.rag_agent import stream_rag_query`
- `src/rag_agent.py:239 stream_rag_query()` → `HybridRetriever.search()` → 直接生成
- **全程没有 LangGraph 参与**

LangGraph 只在 `src/api/server.py` 的 HTTP API 路径上。也就是说：**线上跑的是 v1 时代的纯 RAG 管道，README 却宣称线上是 v4。**
而且这份 README 自己在第 45 行的"两条入口，一个大脑"一节里已经说清了"Web 走纯 RAG"——**同一个文件里前后矛盾**。

### 1.2 README 宣称"早期版本已归档"——仓库里没有

- README:19 「早期版本已归档，演进记录见 git 历史」
- 实际：全仓检索无任何 v1/v2/v3 文件，也没有 archive 目录
- `src/agent/` 目录下**只有一个文件** `dev_agent_langgraph.py`

"归档"给人的预期是文件还在，只是不启用。实际是删掉了。"见 git 历史"也算一种归档，但和"已归档"字面意思不符，容易被认为是虚张声势。

### 1.3 README 的项目结构漏报 7 个模块

README:243-262 列出 `src/` 下的 12 项，实际 `src/` 根目录有 **15 个 py 文件** + 3 个子包。

漏报的模块（都是核心功能）：

| 漏报模块 | 实际作用 |
|---|---|
| `cleaner.py` | 四步数据清洗（页眉页脚/硬换行/垃圾块） |
| `reranker.py` | 交叉编码精排，默认开启，作用最大的改动 |
| `evaluation_ragas.py` | Ragas 四维评估 |
| `evaluation.py` | 手写 LLM Judge 对照实现 |
| `embeddings.py` | 嵌入层（分批/重试/查询侧编码） |
| `config.py` | 全局配置中枢 |
| `chunker.py` | 被合并写在 "parser.py / chunker.py" 一行里 |

**后果**：README 是别人理解项目的第一入口。漏报精排模块，等于把你最拿得出手的改动藏起来了。

### 1.4 结构描述把模块位置写错

- README:246 `tools/  # 文件工具 / 安全审查 / 混合检索`
- 实际混合检索在 `src/hybrid_retriever.py`，不在 `tools/`

---

## 二、生产代码里留着"学习痕迹"

### 2.1 `src/agent/dev_agent_langgraph.py` —— 它是生产入口，但代码是学习笔记

这个文件是 **`src/api/server.py` 唯一 import 的 Agent**，即 HTTP API 的生产实现。但它的注释是写给"正在学 LangGraph 的自己"看的：

```python
# 第 1-11 行（文件头）
"""
dev_agent_langgraph.py —— LangGraph 版 Agent

和 v3 功能完全一样（ReAct：思考 → 调工具 → 思考 → 回答）
区别：用 StateGraph（图）代替 while（循环）

三个核心概念：
  1. State  — 在图中流动的数据（类比快递包裹）
  2. Node   — 处理 State 的函数（类比流水线工人）
  3. Edge   — 节点之间的连线（类比岔路口指示牌）
"""
```

正文里同样：

```
第 187 行   #  tool_calls 也直接返回 LangChain 格式——转换层删掉，少一层出错面。
第 203 行   # 返回的 AIMessage 自带 LangChain 格式的 tool_calls（id / name / args）
第 217 行   # 对应 while 循环：for tc in msg.tool_calls: func(**args)
第 252 行   # 对应 while 循环：if msg.tool_calls: ... else: ...
第 416 行   print("和 v3 功能一样，但用 StateGraph 代替 while 循环")
```

**三个具体问题**：

1. **"类比快递包裹 / 类比流水线工人"** 是教学措辞。生产代码的注释应该解释"为什么这么写"，而不是"这个概念是什么"。
2. **反复引用 "v3"、"while 循环"**——这些对象在仓库里已经不存在了。注释在引用不存在的东西，后来者会去找 v3，然后找不到。
3. **第 416 行的 print 是自述口径**——它说明写这个文件时的目的就是"对比 v3 和 v4"，而不是"提供一个 API 实现"。

这不是要否定这些注释的价值——它们对学习是好的。问题是**它们的位置不对**：应该在 `notes/` 里，不应该在生产入口文件里。

### 2.2 版本编号（v1/v2/v3/v4）只存在于文档和注释里

代码里没有任何版本概念——没有 `v1/` `v2/` 目录，没有版本常量，没有 changelog 文件。v1-v4 只是 README 和注释里的叙事。

对开发者而言，"我们做到 v4 了"是一个需要被代码结构支撑的说法（比如多个实现并存、或 git tag）。现在这个说法没有代码基础，会显得是包装。

---

## 三、命名与实质不符

| 名称 | 实际是什么 | 问题 |
|---|---|---|
| `src/rag_agent.py` | 确定性管道：检索 → 拼上下文 → 生成 | 名字含 "agent" 但没有自主决策。在一个主打 Agent 的项目里，把 RAG 管道叫 Agent，会让人怀疑你分不清 RAG 与 Agent 的区别 |
| `HybridRetriever` | 单路向量检索（`BM25_WEIGHT=0`） | 名字和实际行为不符（此项你已知，决定开启混合检索） |
| README 结构里的 `tools/ # ... 混合检索` | 混合检索在 `src/hybrid_retriever.py` | 位置写错 |

命名是最低成本的诚实。`rag_agent.py` 改名为 `rag_pipeline.py` 或 `rag_qa.py` 只需要一次全局替换——但它消除了一个会在面试里被追问的点。

---

## 四、结构与重复

### 4.1 同一个检索能力，两套 API 风格

| 形式 | 位置 | 返回 |
|---|---|---|
| 类 `HybridRetriever.search()` | hybrid_retriever.py:82 | `list[dict]`（结构化） |
| 函数 `search_knowledge()` | hybrid_retriever.py:185 | `str`（拼给 LLM 看的文本） |
| 函数 `add_knowledge()` | hybrid_retriever.py:203 | `str` |
| 函数 `load_file_to_knowledge()` | hybrid_retriever.py:218 | `str` |

两套出参格式：Web 要 dict（渲染引用），Agent 工具要 str（喂给模型）。这个差异**有合理性**，但代价是同一个文件里并存两种风格，且 `search_knowledge` 内部还要再调一次 `HybridRetriever`。

统一的做法是：类负责结构化结果，模块级函数作为"LLM 工具适配层"集中在一个 `tools/` 模块里，而不是和类挤在同一个文件。

### 4.2 引用去重逻辑逐字重复两处

`src/rag_agent.py`：

- 第 221-232 行（`rag_query` 内）
- 第 266-276 行（`stream_rag_query` 内）

两段代码**逐字相同**，而且第二处的注释是自己写的：

```python
# 去重引用来源（与 rag_query 一致）
```

这说明作者**知道**这里重复了，只是没抽。这类"我知道但没做"的痕迹，比无意重复更显眼。

### 4.3 根目录散落一次性脚本

| 文件 | 行数 | 说明 |
|---|---|---|
| `ingest_missing.py` | 72 | 一次性补数据脚本，使命已完成 |
| `run_eval_experiment.py` | 178 | 评测实验脚本，和 `scripts/` 下同类脚本分离 |

`scripts/` 目录已经存在，这两个没有理由留在根目录。178 行的脚本放根目录尤其扎眼。

---

## 五、可观测性：两套输出体系

| 区域 | 方式 | 数量 |
|---|---|---|
| `src/` 核心模块 | `print(..., file=sys.stderr)` | **37 处** |
| `scripts/` | `print()` | 44 处（命令行脚本，合理） |
| `src/agent/` `src/api/` `src/mcp_server.py` | `logging` | 仅 3 个文件 |

**问题**：核心 RAG 链路（`embeddings` / `vector_store` / `hybrid_retriever` / `reranker` / `parser` / `cleaner` / `chunker`）**完全没有 logging**，靠 print 写 stderr。

具体影响：

- 无法分级（DEBUG/INFO/WARNING/ERROR 混在一起）
- 无法开关（生产环境关不掉调试输出）
- 无法采集（结构化日志系统接不进来）
- 流式问答 + 并发请求时，stderr 的输出会交叉混杂，无法归因到某次请求

而 `src/agent/dev_agent_langgraph.py` 已经用得很规范（`logger.info` / `logger.warning` / `logger.error` 分级清楚）。说明**你知道该怎么做，只是核心模块没做**——这种"同一仓库内标准不统一"比"完全不懂"更值得修。

---

## 六、工程规范缺失

### 6.1 零配置

| 文件 | 状态 |
|---|---|
| `pyproject.toml` | 不存在 |
| `ruff.toml` / `.flake8` | 不存在 |
| `.pre-commit-config.yaml` | 不存在 |
| `Makefile` | 不存在 |

6000+ 行的 Python 项目，没有任何 linter/formatter 配置。后果：代码风格靠自觉，import 顺序、行宽、命名风格全靠手写一致性。

### 6.2 硬编码个人机器路径 4 处

```
src/agent/dev_agent_langgraph.py:332   def run_agent(question, work_dir: str = "C:/Users/24162/Desktop", ...)
src/agent/dev_agent_langgraph.py:377   def stream_agent(question, work_dir: str = "C:/Users/24162/Desktop", ...)
src/agent/dev_agent_langgraph.py:427   run_agent(q, work_dir="C:/Users/24162/Desktop", session_id="cli-interactive")
src/api/server.py:65                   default="C:/Users/24162/Desktop"
```

其中前两处是**函数默认参数**，第四处是 **Pydantic 字段默认值**。意味着：

- 别人 clone 下来直接跑 API，默认会指向一个不存在的目录
- 这是个人机器专属配置，不该出现在源码里
- 修法：默认值改为 `None`，运行时取 `os.getcwd()` 或从环境变量读

---

## 七、测试覆盖错位

### 7.1 做得扎实的部分（值得保留）

工具层测试 95 项，覆盖：清洗（15）、分块（28）、精排（11）、嵌入分批（含重试降级）、文档删除一致性、语义缓存、安全审查。这些测试覆盖的都是**有明确输入输出的确定性逻辑**，选得很对。

### 7.2 空白的部分

| 未被测试的核心 | 说明 |
|---|---|
| `src/rag_agent.py` | **线上 Streamlit 的主力问答路径，零测试** |
| `hybrid_retriever.py` 的 `search()` 全链路 | 只测了分词等零件，主流程没有直接测试 |
| `parser.py` 的 docx 表格 | 已知会静默丢内容，零测试（见 7.3） |

**结构性问题**：测试覆盖了"我新写的模块"，没覆盖"一直在用的主链路"。而主链路才是用户真正走的路。

### 7.3 一个已知的、测试应该拦住的 bug

`src/parser.py::parse_docx()` 只读段落：

```python
full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
```

**`doc.tables` 完全没被处理**。实测（临时构造带表格的 docx）：

- 段落一、段落二 → 保留
- 2×2 表格（基金代码 / 年化收益 / 025490 / 12.5%）→ **全部消失**

文档照样标记为 ready，用户看不到任何提示。表格是信息密度最高的部分，而且这类内容恰恰是 RAG 问答最常被问到的。

**这个 bug 至今零测试覆盖**——CI 跑了全会通过。这是"测试存在但不覆盖真实风险"的典型。

---

## 八、不是问题的地方（避免误伤）

审查要客观，以下项经核实**不是问题**，不要改：

1. **"两条入口，一个大脑"是合理设计。** README:45-52 明确写了分工：知识库问答走 RAG 快路径，文件工具操作走 Agent 慢路径，两条共用 `HybridRetriever`。这个取舍是对的——把纯问答塞进 ReAct 循环只会变慢。README 在这一节写得比第 11 行清楚得多。
2. **`except ... pass` 共 7 处，多数合理。** 抽查：`rag_agent.py:290` 有注释「缓存失败不影响回答」、`vector_store.py` 几处在清理/删除的防御路径上。属于正常的"非关键路径容错"，不是吞异常。
3. **`scripts/` 里 44 处 print 合理。** 命令行脚本的输出就是给终端看的，不需要 logging。
4. **CI 配置写得不错。** `.github/workflows/ci.yml` 只装 `requirements-dev.txt`、明确"不调 LLM、不碰密钥、不碰数据库"——这个边界划得清楚。
5. **配置集中在 `src/config.py`。** 114 行，无散落的魔数，环境变量覆盖齐全，`config.py` 里对精排/分块的取舍说明写得很清楚（虽然位置更适合放 notes/）。

---

## 九、修复优先级

### P0 — 信任层（约 40 分钟，纯文本改动，零风险）

| # | 动作 | 涉及文件 |
|---|---|---|
| 1 | README 第 11/19/31 行的"线上版本是 LangGraph"改为事实描述 | README.md |
| 2 | 删除或改写"早期版本已归档"（改为"早期版本已从仓库移除，仅存于 git 历史"） | README.md |
| 3 | 项目结构补全 7 个漏报模块，修正 `tools/` 描述 | README.md |
| 4 | `rag_agent.py` 改名（`rag_qa.py`），README 同步 | 全局 |

### P1 — 痕迹层（约 1 小时）

| # | 动作 | 涉及文件 |
|---|---|---|
| 5 | 把 `dev_agent_langgraph.py` 的教学式注释移入 `notes/`，生产文件只留"为什么" | notes/ + src/agent/ |
| 6 | 删掉第 416 行的教学 print | src/agent/ |
| 7 | 硬编码路径 4 处改为 `None` + 运行时求值 | src/agent/ + src/api/ |
| 8 | 根目录两个脚本移入 `scripts/` | ingest_missing.py / run_eval_experiment.py |

### P2 — 结构层（约 2 小时）

| # | 动作 |
|---|---|
| 9 | 抽 `_dedup_sources()`，消除 rag_agent 两处重复 |
| 10 | 加 `pyproject.toml` + ruff 配置（行宽/import 顺序/未使用变量） |
| 11 | 核心 RAG 链路从 print 迁到 logging（按模块逐步，不必一次做完） |
| 12 | 统一检索 API：类出结构化结果，工具适配层单独放 |

### P3 — 覆盖层（约 3 小时）

| # | 动作 |
|---|---|
| 13 | 修 docx 表格丢失 + 补测试（**同时这是内容正确性问题，不是测试问题**） |
| 14 | 给 `rag_agent.py` 主链路补测试（mock LLM 与检索） |
| 15 | 给 `hybrid_retriever.search()` 补全链路测试 |

---

## 十、一句话总结

**这个项目的技术内容配得上简历，但工程的呈现方式配不上技术内容。**

Rerank 精排、清洗管线、分批重试、RAGAS 评估——这些都是真东西，而且有 A/B 数据支撑。
但它们现在被埋在一个"README 说错话、目录漏报、注释是教学体、路径写死在我自己电脑上"的外壳里。

先把壳修对，成本一个小时；不修，那些真东西会先被怀疑。
