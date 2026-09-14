# DataLens 反馈环方案（开发者视角）

> 一句话：**把「用户觉得这次答得对不对」变成「系统下一次真的不一样」的那条通路。**
>
> 本文是设计方案，不含实现。分 P0 / P1 / P2 三档，每档独立可交付、可验收、可回退。

---

## 1. 为什么要做：环断在哪

### 1.1 代码级现状

`data/datalens.db` 里只有三张表：

| 表 | 用途 | 存不存"这次答得怎么样" |
|---|---|---|
| `knowledge_bases` | 知识库元数据 | ✗ |
| `documents` | 文档元数据 | ✗ |
| `query_cache` | 语义缓存（性能优化） | ✗ |

`pages/3_💬_智能问答.py` 全文只有一个交互按钮：`🗑️ 清空对话`。

### 1.2 环断在"写侧"

整条链路读得通、写了没了：

```
提问 ──→ 追问消解 ──→ 检索(向量+BM25) ──→ 精排 ──→ 门控 ──→ 生成 ──→ 回答
  ▲                    ▲                    ▲          ▲
  │                    │                    │          │
  └────────────────────┴────────────────────┴──────────┘
                  这三样都是"只读"的，
                  没有任何一条线把"结果好不好"写回去
```

知识库侧的三个动作（传文档 / 删文档 / 重建索引）**全部由人手动触发**，系统自己不产生任何"我该改点什么"的结论。

→ 现在的定位是**「一次性问答工具」**，不是**「可自我改进的系统」**。

### 1.3 为什么这是最后一个空环

| 环 | 状态 |
|---|---|
| 解析 → 清洗 → 分块 → 嵌入 → 入库 | ✅ 通 |
| 检索 → 精排 → 门控 → 生成 → 引用 | ✅ 通 |
| 索引运维（原文留存 + 重建） | ✅ 通（09-14） |
| **反馈 → 归因 → 回流** | ❌ **全空** |

前几环解决的是「**这次答得对不对**」，反馈环解决的是「**下次能不能更对**」。前者是工程质量，后者是产品能不能自己长大。

---

## 2. 为什么 RAG 特别需要它

不是"锦上添花"，是这个架构的固有缺口。

### 2.1 RAG 的失败不可自证

检索没命中、库里根本没有这段内容 —— **模型不报错**。它拿手里的低分上下文，编一个读起来很通顺的答案。

门控（`answer_gate`）能挡住"整库都没有"，**挡不住"库里有、只是没检索到"**。这两种失败在系统内部长得一模一样：都是一次"成功"的调用，都返回 200，都写了缓存。

### 2.2 静态评测集的边界

`tests/golden_set.json` 现有 26 条，**全部由我人工构造**。它的能力边界很硬：

- ✅ 能证明「改完没变差」（回归）
- ✅ 能证明「某个已知方向有提升」（A/B）
- ❌ 测不出「真实用户会问什么我没想到的问法」

**只有真实用户会撞上你没想到的边界。** 这不是评测方法的问题，是静态数据集的本质限制 —— 你无法用一份自己写的题，发现自己的盲区。

### 2.3 所以反馈环是唯一能发现"未知的未知"的机制

三个层次的收益，从低到高：

1. **修错**：这条答错了 → 找到原因 → 改
2. **发现缺口**：被门控拦下的问题自动汇成"知识库缺什么"的清单
3. **扩评测集**：真实问法沉淀成回归用例，让未来的改动被真实需求兜住

第 2 条尤其重要，见 §4.4 —— **它不需要用户点任何按钮就能产生**。

---

## 3. 设计原则

### 3.1 先有黑匣子，再谈反馈

**没有现场快照的反馈，是无法归因的反馈。**

用户点 👎 说"不对"，如果看不到当时检索了什么、分数多少、门控有没有拦 —— 这条信号等于什么都没说。你只能去复现，而复现要先把用户的问法、当时的知识库状态都还原出来，成本高到不会有人真的去做。

→ **P0 必须先做日志，且日志要独立可用**。它不依赖 P1 的按钮。

### 3.2 信号要自己走到出口，别攒待办列表

按钮是最好做的部分。真正会让环断掉的是「收到 👎 之后谁去看、看了怎么处理」。

单人项目没有客服团队盯反馈列表。所以设计上必须让**每个信号直接绑到一个已存在的动作**上，而不是攒一个"待处理"列表等人来看。

好在三个出口都已经焊好了：

| 信号 | 出口 | 状态 |
|---|---|---|
| 👎 这条答错了 | `query_cache` → 作废该条缓存 | 表已在跑 |
| "库里这段是错的 / 缺的" | `reindex` → 补原文后一键重建 | 09-14 已完成 |
| "这个问法很重要" | `golden_set.json` → 转成回归用例 | 链路已通 |

### 3.3 自动 vs 人工，按"改动的可逆性"分级

收到信号就一把梭自动改系统，会引入**反馈污染**（用户点错、恶意反馈、单个人的偏好被当成全局真理）。

分级的判据不是"这个改动重不重要"，而是**"改错了能不能撤"**：

| 动作 | 可逆性 | 策略 |
|---|---|---|
| 作废一条缓存 | 完全可逆（大不了重算一次） | **自动** |
| 补一段知识库原文 | 半可逆（能删掉，但索引要重跑） | **人工确认** |
| 把问法写进评测集 | 可逆但会长期影响所有 A/B 基线 | **一键 + 人工填 expected_sources** |

---

## 4. P0：问答现场快照（黑匣子）

**目标**：任意一次历史问答，事后都能还原"当时系统看到了什么、做了什么决定"。

### 4.1 关键约束：门控会销毁证据 ⚠️

这是设计本文时核实到的第一件事，它决定了抓取点在哪。

`src/rag_qa.py:267-272`：

```python
grounded = is_grounded(contexts)
if not grounded:
    print(f"[AnswerGate] 检索最高分 {top_score(contexts):.4f} 低于阈值，...")
    contexts = []          # ← 分数在这里被销毁
```

返回体里的 `contexts` 是**清空之后**的。前端拿到的 `contexts == []`，事后无法知道当时的最高分是 0.28 还是 0.02。

**这两个数含义完全不同**：
- `0.28` → 差一点点过阈值，属于「检索没调好」
- `0.02` → 库里压根没有，属于「该补文档」

而它们在现有返回体里长得一模一样。

→ **快照必须在门控前抓**，把 `gate_score = top_score(contexts)` 和 `hits` 一起存下来。

### 4.2 快照点

| 位置 | 抓什么 |
|---|---|
| `rag_query` / `stream_rag_query` 检索完、门控前 | `gate_score`、`hits`（完整候选快照） |
| 流式生成结束（`gen()` 尾部，与 `cache_answer` 同处） | `answer` 回填 + 写库 |
| 缓存命中分支（`stream_rag_query:304-310`） | `hit_cache=True`，`gate_score=None` |

⚠️ **缓存命中分支容易漏**：它 `return` 在最前面，**根本没检索**。但命中的回答同样可能被用户判为错 —— 而且这条 👎 价值特别高，因为它意味着「错的答案被缓存了，会持续错下去」。

### 4.3 数据模型

```sql
CREATE TABLE IF NOT EXISTS answer_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id           TEXT    NOT NULL,
    question        TEXT    NOT NULL,   -- 用户原话
    retrieval_query TEXT    NOT NULL,   -- 实际拿去检索的 query（追问消解后）
    answer          TEXT    NOT NULL,   -- 最终答案
    grounded        INTEGER NOT NULL,   -- 是否通过质量门控（1/0）
    gate_score      REAL,               -- 门控判定的最高精排分；缓存命中为 NULL
    hit_cache       INTEGER DEFAULT 0,  -- 是否命中语义缓存
    backend         TEXT,               -- cloud / ollama
    hits            TEXT,               -- 命中快照 JSON（见下）
    rating          TEXT,               -- P1：NULL / 'up' / 'down'
    comment         TEXT,               -- P1：用户补充说明
    rated_at        TEXT,               -- P1
    resolved        TEXT DEFAULT 'none',-- P2：none/ignored/cache_purged/kb_patched/case_added
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_log_kb ON answer_log(kb_id);
CREATE INDEX IF NOT EXISTS idx_answer_log_rating ON answer_log(rating);
```

`hits` 存什么（**这是复盘的全部依据**）：

```json
[
  {
    "source": "02-rag-pipeline.md",
    "page": null,
    "type": "text",
    "rerank_score": 0.6721,
    "rrf_score": 0.0325,
    "snippet": "前 200 字..."
  }
]
```

按现有惯例（`query_cache._init_table()`），**建表用 `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE` 补列**，兼容已存在的旧库，不做破坏性迁移。

### 4.4 独立价值：被动反馈（知识库缺口清单）

这是 P0 最值钱的地方，也是它**不该等到 P1 才做**的原因：

`grounded = 0` 的记录天然就是一份「**用户问了、但我库里的东西没撑住**」的清单。它不需要用户点任何按钮就自动产生 —— 属于 implicit feedback。

按 `gate_score` 排序就能分出两类完全不同的待办：

| `gate_score` | 含义 | 该做的事 |
|---|---|---|
| 0.20 ~ 0.30 | 差点过阈值 | 调检索（分块 / 权重 / 阈值），**不用补文档** |
| < 0.10 | 库里有语义邻居都没有 | 补文档，或确认这问题本就不该问库 |

**没有这个字段，这两类问题会被混在一起当成"检索效果不好"**，然后去调一个根本没调错的参数。

### 4.5 写入时机与失败处理

**时机**：流式生成结束后一次写入（与 `cache_answer` 同一位置）。

不由分两次写（先 INSERT pending 再 UPDATE）的原因写在 §4.6。

**失败处理**：沿用 `cache_answer` 的既有模式，**日志失败绝不能影响回答**：

```python
try:
    log_answer(...)
except Exception:
    pass   # 日志写入失败不影响回答
```

并且 `log_answer` 内部自身也要全包 try/except —— 日志是旁路，不是主链路。

### 4.6 已知局限（诚实标注）

**1. "用户等不到答案就关页面"这件事捕获不到。**

Streamlit 的 `st.write_stream` 在页面关闭时中断生成器，服务端抛 `GeneratorExit`，**不可靠捕获**。强行做需要引入心跳/超时机制，复杂度远大于收益，且仍然不准。

所以接受：**库里只记"答完了"的问答**。中断事件留空。

> 这条在单人使用场景下损失可接受。若将来有多人使用，这个信号会变得重要 —— 那时再考虑把日志改成两阶段写（先 INSERT `status='pending'`，生成完 UPDATE 成 `done`，超时未 UPDATE 的即视为中断）。

**2. 记录的是"回答"，不是"用户真实需求"。**

如果用户问 A、系统答 B，用户直接换个问法重问 —— 你看到的是两条独立记录，看不出它们是一回事。跨轮次的需求归并属于会话分析，不在本方案范围。

---

## 5. P1：反馈入口

**目标**：允许用户显式表达"这条对不对"。

### 5.1 交互设计

每条 assistant 回答下方、引用来源之后：

```
👍 有帮助    👎 没答对
```

点 👎 后展开一个可选输入框：`哪里不对？（选填，一句话就够）`

**为什么用可选而不是必填**：要求必填会让人干脆不点。默认零输入成本，愿意说的才说。

### 5.2 数据落点

复用 `answer_log` 的两个列 + `rated_at`，**不新建表**。

权衡：一个回答只能有一份当前评价，改主意就 UPDATE（保留不了修改历史）。理由：
- 单人项目下"用户改主意"的历史没有分析价值
- 独立 `feedback` 表会引入 JOIN，而查询场景只有两种：「哪些被 👎 了」「这条的反馈是什么」，列足够
- 若将来要做多人标注（同一问题多人评），再拆表 —— 那时才需要"多份评价"的语义

### 5.3 幂等与改主意

用户随时能改（👎 改 👍、撤销）。`rating` 是**当前状态**，不是事件流。

Streamlit 的 `st.button` 每次 rerun 状态会丢，需要一个显式写库 + `st.rerun()` 回显的模式（与 `pages/1` 里 `kb_msgs` 的做法同源），保证刷新后按钮显示的是已存的评价。

### 5.4 按钮定位

反馈按钮的 key 必须绑定 `answer_log.id`，否则 Streamlit 重跑后会串台（同一页多条历史消息，每个按钮都要有唯一身份）。

这要求 `answer_log.id` 能传到前端 —— 见 §8 的接口改造。

---

## 6. P2：三档回流

按 §3.3 的可逆性分级。三档都在**已有的出口**上接线，不新建机制。

### L1 自动：缓存作废（完全可逆）

👎 落在一条 `hit_cache = 1` 的记录上 → 自动作废那条缓存。

**卡点**：现有 `query_cache.clear_kb_cache(kb_id)` 是**整库清空**，粒度太粗。作废一条 👎 会顺带干掉几百条正常缓存。

→ 需要新增 `invalidate_cached_answer(kb_id, question)`：按 `question` 语义匹配（复用 `_embed` + `_cosine` + `QUERY_CACHE_THRESHOLD`），删掉命中的那一行。

**为什么这档可以自动**：删错了最坏结果只是"下次重算一遍"，完全可逆。

### L2 人工确认：知识库补充（半可逆）

👎 且 `comment` 提到内容错误/缺失 → 进「待补知识」列表。

**为什么必须人工**：一条用户反馈说"这是错的"不代表它真的错。自动改库 = 让单个用户（甚至误点）直接改写知识库。必须人看一眼再动手。

流程：
```
👎 + 备注 ──→ 待补知识列表 ──→ 人工补原文到 data/uploads/{kb_id}/
                              ──→ 点「♻️ 重建索引」（已存在）
                              ──→ 标记 resolved = 'kb_patched'
```

### L3 一键：用例沉淀（可逆但影响面大）

👎 记录 → 一键转成 `golden_set.json` 的 case。

生成骨架，**`expected_sources` 必须人工填**（机器不知道该命中哪个文档）：

```json
{
  "id": "q27",
  "question": "<用户原话>",
  "expected_sources": ["<人工填>"],
  "difficulty": "hard",
  "type": "single-hop",
  "note": "来自 2026-09-14 用户反馈（answer_log #123）"
}
```

**为什么要人工填**：评测集的全部价值在于 `expected_sources` 是**人类认定的正确答案**。如果让机器猜测并自动写入，这份评测集就变成了"系统自己给自己出题、自己判自己"——基线从此不可信。

⚠️ 新增 case 会让历史存档的指标**不再可比**（题数变了）。`golden_set.json` 的 `meta.note` 里已有这个惯例记录（v1.2 删 q23 时就记了），新增时照做。

---

## 7. 完整数据模型（新增部分）

```sql
-- P0
CREATE TABLE IF NOT EXISTS answer_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id           TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    retrieval_query TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    grounded        INTEGER NOT NULL,
    gate_score      REAL,
    hit_cache       INTEGER DEFAULT 0,
    backend         TEXT,
    hits            TEXT,
    rating          TEXT,
    comment         TEXT,
    rated_at        TEXT,
    resolved        TEXT DEFAULT 'none',
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_log_kb ON answer_log(kb_id);
CREATE INDEX IF NOT EXISTS idx_answer_log_rating ON answer_log(rating);

-- P2：待补知识列表（L2）
CREATE TABLE IF NOT EXISTS kb_gaps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id       TEXT NOT NULL,
    answer_log_id INTEGER NOT NULL,
    note        TEXT DEFAULT '',
    status      TEXT DEFAULT 'open',   -- open / done / ignored
    created_at  TEXT NOT NULL
);
```

**保留策略**（必须有，否则日志无限增长）：

```python
ANSWER_LOG_MAX_PER_KB = 500   # 对齐 QUERY_CACHE_MAX_PER_KB 的惯例
```

淘汰规则**不能是简单的"删最旧"**：

```sql
DELETE FROM answer_log
WHERE kb_id = ?
  AND rating IS NULL                    -- 被评价过的绝不淘汰
  AND resolved IN ('none', 'ignored')   -- 已处理的可以淘汰
ORDER BY id ASC LIMIT ?
```

理由：被 👎 但还没处理的记录，正是这个功能的全部产出。**按时间淘汰会把最该看的信号先删掉。**

---

## 8. 接口改造清单

| 文件 | 改动 | 备注 |
|---|---|---|
| `src/answer_log.py` | **新增**。`log_answer()` / `list_answers()` / `rate_answer()` / `gap_stats()` | 独立模块，对齐 `query_cache.py` 的写法（`_init_table` + 惰性 import） |
| `src/database.py` | `init_db()` 里追加建表 | 或由 `answer_log._init_table()` 自建，二选一 |
| `src/rag_qa.py` | 门控前抓 `gate_score` + `hits`；两处 return 挂 `log_answer` | **门控后的 `contexts = []` 不动** |
| `src/rag_qa.py` | `stream_rag_query` 返回值 **4 元组 → 5 元组**（第 5 个是 `log_ref` 可变容器） | 流式下 answer 还没生成，只有容器能穿透 |
| `src/query_cache.py` | 新增 `invalidate_cached_answer(kb_id, question)` | 现有 `clear_kb_cache` 是整库清空，粒度不够 |
| `pages/3_💬_智能问答.py` | 解包 5 元组；每条回答后渲 👍/👎 | key 绑 `answer_log.id` |
| `pages/4_📊_评估面板.py` | 新增「问答日志」区块 | 不新建页面 —— 它本质是观测数据，与评估面板同类。⚠️ 该页目前是**线性平铺**（无 `st.tabs`），先按现有风格追加 `st.divider()` + `st.subheader()`；内容变多再引入 tabs |
| `.env.example` | `ANSWER_LOG_ENABLED` / `ANSWER_LOG_MAX_PER_KB` | 对齐现有开关风格 |

**`log_ref` 用可变容器的原因**：`stream_rag_query` 返回时生成器还没跑，`answer` 尚未产生，日志 id 也不存在。前端 `st.write_stream(gen)` 跑完后才能从容器里取到 id。这是 Streamlit 流式的常态（`pages/3` 现有代码里 `cited = extract_cited_sources(answer, contexts)` 就是这个模式）。

---

## 9. 界面

### 问答页（`pages/3`）

```
┌──────────────────────────────────────────────┐
│ [用户] RAG 的五个步骤是什么？                  │
├──────────────────────────────────────────────┤
│ [助手] RAG 全链路包含五个步骤…                 │
│                                              │
│   📄 引自 02-rag-pipeline.md                  │
│      「文档解析、文本分块、向量化嵌入…」        │
│                                              │
│   👍 有帮助    👎 没答对       ← P1 新增      │
└──────────────────────────────────────────────┘
```

未命中（`grounded = 0`）时，在现有的「⚠️ 未命中知识库」提示旁加一行：

```
⚠️ 未命中知识库（最高相关度 0.08，低于阈值 0.3）
```

**把分数露出来**。用户看到 0.08 会知道"知识库里可能真没有"，看到 0.28 会知道"是差一点"。这一行同时也是在向用户解释系统的判断依据，比"没找到"三个字诚实得多。

### 评估面板（`pages/4`）新增「问答日志」区块

该页目前是线性平铺布局（无 `st.tabs`），按现有风格追加即可；内容变多再引入 tabs。

- 汇总：总问答数 / 命中缓存率 / 未命中率 / 👎 数
- 「知识库缺口」分组（按 `gate_score` 排序，见 §4.4 的两类分法）
- 「被 👎 的记录」列表，每条可展开看 `hits` 快照、可直接改 `resolved`

---

## 10. 验收标准

### P0

1. `tests/test_answer_log.py`：建表 / 写入 / 读取 / 淘汰策略（**断言被评价的记录不被淘汰**）/ 字段完整性
2. 一次真实问答后，`answer_log` 里能查到该条，`hits` 里含真实的 `rerank_score`
3. **端到端**：构造一个必然被门控拦下的问题，断言 `gate_score` 被记录下来且 **> 0**（证明抓取点在门控前，这是 §4.1 那个约束的回归测试）
4. 日志写入失败（如 monkeypatch 让 `log_answer` 抛异常）时，**回答仍正常返回** —— 旁路不能拖垮主链路
5. `log_answer` 关闭（`ANSWER_LOG_ENABLED=false`）时行为与改动前完全一致

### P1

6. 点 👎 后刷新页面，按钮仍显示 👎（状态真的落库了，不是 session 态）
7. 多条历史消息的反馈按钮**互不串台**（key 唯一性）
8. 👎 改 👍 后 `rating` 正确翻转

### P2

9. L1：👎 一条缓存命中的记录后，同问再查**不命中缓存**（且其他缓存条数不变 —— 验证不是整库清空）
10. L3：转出的 case 能被 `scripts/eval_retrieval.py` 直接读取，且 `expected_sources` 为空时**拒绝写入**（防止机器自己给自己出题）

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| **反馈污染**：误点 / 恶意反馈被自动应用 | 只有"作废缓存"是自动的（可逆）；改知识库、改评测集一律人工确认 |
| **隐私**：`question` / `comment` 会积累真实提问 | 本地单人部署风险低；若上公网 Demo，`ANSWER_LOG_ENABLED` 默认关 + 注明保留期 |
| **存储增长** | `ANSWER_LOG_MAX_PER_KB` + 淘汰时保护未处理的反馈（§7） |
| **拖慢主链路** | 日志写入在回答**之后**，且全包 try/except；不引入额外网络调用（`hits` 直接复用已算好的 contexts，**不重新检索**） |
| **日志与缓存数据不一致** | 两者是独立事实：`hit_cache=1` 时 `gate_score` 为 NULL 是**正确**的（那次根本没检索），不是缺失 |
| **评测基线不可比** | 新增 case 时在 `meta.note` 里记一笔（沿用 v1.2 的惯例） |

---

## 12. 非目标（明确不做）

- ❌ **不做自动调参**：没有足够反馈量支撑统计显著性，26 条测试集 + 零星反馈去做自动优化 = 过拟合到噪声
- ❌ **不做用户账号 / 权限**：单机单用户，引入认证是纯复杂度
- ❌ **不做反馈触发的模型微调**：标注量差几个数量级，且本项目的 RAG 链路不需要
- ❌ **不做跨会话的需求归并**（§4.6 局限 2）
- ❌ **不重构 `query_cache`**：只加一个按条作废的函数，不动现有结构

---

## 13. 交付顺序

| 阶段 | 内容 | 依赖 | 可独立交付 |
|---|---|---|---|
| **P0** | 黑匣子（建表 + 写入 + 评估面板 tab） | 无 | ✅ 有独立价值（缺口清单），不需要用户点按钮 |
| **P1** | 反馈入口（👍/👎 + 备注） | P0 的 id 贯通 | ✅ 没 P2 也能用（数据先攒着） |
| **P2-L1** | 缓存作废（自动） | P1 | ✅ |
| **P2-L2** | 待补知识列表（人工确认） | P1 | ✅ |
| **P2-L3** | 用例沉淀（一键） | P1 | ✅ |

**建议按 P0 → 停下来看数据 → 再决定 P1 的形态这个顺序。**

理由：先让黑匣子跑几天，看真实的 `gate_score` 分布和未命中率。如果发现自己根本不怎么用这个系统（比如一周才问 3 次），那么 P1/P2 的投入产出比就很低，**这个结论本身就是有价值的**——比先花两天做完按钮然后发现没人点要好。

---

## 附：与已有文档的关系

| 文档 | 关系 |
|---|---|
| `notes/rag-six-stages.md` | 六环节是"答得对"的链；本文是"下次更对"的链 |
| `notes/engineering-review.md` | §7 列的测试覆盖错位（测新模块不测主链路），本文 P0 的验收标准按此要求写 |
| `notes/enterprise-refactor-plan.md` | 企业级方向要的是"多用户 + 反馈运营"；本文是它的最小可落地版本 |
