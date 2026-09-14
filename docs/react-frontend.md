# React 前端技术文档（`frontend/`）

> 定位一句话：给 DataLens 做的**新版 Web 界面**，React 18 实现，产物是**单个自包含 `index.html`**——零外部依赖。由 FastAPI 同源托管在 `/app`，接真实后端；单独双击打开时自动回退演示数据。

---

## 0. 怎么跑起来

```bash
python src/api/server.py          # 或 uvicorn src.api.server:app
# 浏览器打开 http://localhost:8000/app      → React 前端（接真数据）
#            http://localhost:8000/docs     → API 文档（含 /api/* 全部端点）
```

后端不可达（比如直接双击 `frontend/index.html`）时，前端顶部与侧边栏会显示「演示数据」，所有页面回退到内置 mock 内容，仍可完整演示交互。

---

## 1. 为什么是"单文件打包"而不是常规脚手架

两个决策都有直接原因：

**① 为什么不用 CDN 引 React（Babel standalone）**
第一版用 `unpkg.com` 运行时加载 React + Babel，实测在直连/普通网络环境下拉不到资源，页面直接白屏。改成 esbuild **预编译打包**后，React 被打进产物，彻底消除运行时网络依赖——这对一个强调"可离线、数据不出域"的项目也是一致的选择。

**② 为什么不用 Vite/CRA 脚手架**
脚手架产物是一堆 chunk，必须起 dev server 或部署静态目录才能看。本项目把 React + 应用代码 + CSS 全部打进一个 HTML 文件（约 184 KB），**发给别人、挂 GitHub Pages、离线演示都是"发一个文件"的事**。代价是产物不可读——但源码（`app.jsx`）就放在旁边，构建一条命令可复现。

## 2. 目录结构与构建

```
frontend/
├── app.jsx          # 源码：全部页面 + 组件 + 演示数据（唯一需要改的文件）
├── template.html    # 页面骨架（<style> 全部 CSS + __BUNDLE__ 占位符）
├── build.mjs        # 构建脚本：esbuild 打包 → 注入占位符 → 产出 index.html
├── index.html       # 构建产物（自包含，可直接双击打开 / 发给任何人）
└── package.json     # 依赖：react / react-dom + esbuild
```

构建流程：

```bash
cd frontend
npm install        # 首次
npm run build      # 改完 app.jsx 后重跑
# 产物：index.html（约 184 KB，含 React）
```

`build.mjs` 做三件事：esbuild 把 `app.jsx` 连同 react/react-dom 打成单 bundle（minify、`process.env.NODE_ENV=production`）→ 安全校验 bundle 内不含 `</script>`（防止内联时截断 HTML）→ 替换 `template.html` 的 `__BUNDLE__` 占位符写出 `index.html`。

**改前端只改 `app.jsx`，然后 `npm run build`，不要手改 `index.html`**（它会被覆盖）。

## 3. 页面结构（与 Streamlit 五页一一对应）

| React 页面 | 对应 Streamlit | 界面内容 |
|---|---|---|
| 智能问答 | `pages/3_💬_智能问答.py` | 聊天气泡；引用 `[n]` 点击展开原文片段与精排分；每条回答带门控/缓存/耗时元信息徽章；演示「阈值拒答」话术；右侧检索流水线可视化（改写→混合召回→精排→门控→生成） |
| 知识库管理 | `pages/1_📚_知识库管理.py` | 统计卡片（文档数 / chunk 数 / 索引状态）+ 文档列表（就绪/解析中/失败状态标签、重建索引入口） |
| 文档上传 | `pages/2_📄_文档上传.py` | 拖拽上传 + 模拟进度条 + 解析选项开关（清洗管线 / 本地 OCR / Word 表格序列化） |
| 评估面板 | `pages/4_📊_评估面板.py` | RAGAS 四维雷达图（手写 SVG，无图表库依赖）+ 历史趋势柱状图 + 多配置对比表 |
| 问答日志 | Streamlit 内嵌于评估面板 | 未命中问题按最高精排分分档：「差点过阈→调检索」/「无语义邻居→补文档」两张待办清单 |

组件全部为函数组件 + Hooks（`useState / useRef / useEffect`），无路由库（顶层 `page` state 切页）、无 UI 框架（手写 CSS 变量主题）、无图表库（雷达图/柱状图为手写 SVG）——保持产物精简。

## 4. 演示数据与真实能力的对应

mock 数据刻意按真实系统的口径编写，讲解时可直接对照：

- **引用**：`citations` 里每条有 `file / page / score / snip`，对应 `src/citations.py` 的"序号→真实来源映射 + 命中原文"；
- **门控**：问"天气/大盘"类问题时前端返回拒答话术，对应 `src/answer_gate.py` 的阈值拒答（阈值 0.3）；
- **评估**：RAGAS 四指标（Context Precision / Recall、Faithfulness、Answer Relevancy，0-100）对应 `src/evaluation_ragas.py`；
- **问答日志**：两类待办分档对应 `src/answer_log.py` 的反馈环设计。

## 5. 后端接口（`src/api/web_api.py`）

前端通过同源相对路径调 `/api/*`。这一层**只做协议转换**（HTTP ↔ 现有模块调用），检索、门控、解析、入库全部复用 Streamlit 已用的模块，保证两条入口行为一致。

| 端点 | 方法 | 说明 | 复用的后端能力 |
|---|---|---|---|
| `/api/health` | GET | 连通性探测（前端据此判断是否回退演示数据） | — |
| `/api/kbs` | GET | 知识库列表 + 文档数 / chunk 数 | `database.list_kbs` / `get_kb_stats` / `vector_store.collection_count` |
| `/api/kbs/{kb_id}/docs` | GET | 文档列表 + 统计 | `database.list_documents` |
| `/api/ask` | POST | RAG 问答：question + kb_id + history（最近 6 轮） | `rag_qa.rag_query` 全链路 |
| `/api/kbs/{kb_id}/upload` | POST | 上传入库（multipart） | `parser.parse_file` → `chunker.chunk_parsed` → `vector_store.add_chunks` |
| `/api/kbs/{kb_id}/reindex` | POST | 按当前参数重建索引 | `reindex.rebuild_kb` |
| `/api/docs/{doc_id}` | DELETE | 删文档（SQLite + Chroma 双清） | `database.delete_document` + `vector_store.delete_chunks_by_source` |
| `/api/answer_log` | GET | 问答日志 + 缺口分类 | `answer_log.gap_stats` / `list_answers` |
| `/api/eval/history` | GET | RAGAS 评估存档列表 | `evaluation_ragas.list_history` |
| `/api/eval/history/{file}` | GET | 某份存档完整结果 | `evaluation_ragas.load_history` |

`/api/ask` 的返回刻意带上现场元信息，前端据此展示门控与检索质量：

```json
{
  "answer": "……",
  "grounded": true,              // 是否通过检索质量门控
  "gate_score": 0.903,           // 本次最高精排分（门控未过时从本次日志回填）
  "retrieval_query": "……",       // 实际用于检索的 query（追问时是消解后的自包含问句）
  "sources": [{"n": 1, "file": "rag_agent_guide.md", "page": 0, "snippets": ["…原文片段…"]}],
  "log_id": 42
}
```

> 门控未通过时 `rag_query` 会清空 contexts，此时最高分从本次落下的 answer_log 记录回填——前端要展示「0.28 差一点但没过阈」只能这么拿。

部署相关：FastAPI 加了 CORS（`allow_origins=["*"]`，本机服务 + 允许 `file://` 直开调试），并把 `frontend/` 目录挂在 `/app` 静态托管，因此前端用相对路径即可，不需要配置 API 地址。上传接口依赖 `python-multipart`（已加入 `requirements.txt`），缺了会在定义路由时直接抛 `RuntimeError`。

## 6. 已知限制

- 问答为非流式（`/api/ask` 一次返回），打字机效果尚未做（后端已有 `stream_rag_query`，可后续加 SSE）；
- 评估面板展示的是**历史存档**，面板内不触发新的 RAGAS 评测（评测耗时长，仍在 Streamlit 评估面板跑）；
- 移动端未适配（以桌面演示为主，最小可用宽度约 1100px）；
- 无路由、无持久化，刷新回到初始状态。
