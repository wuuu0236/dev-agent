# Dev-Agent 企业级优化重构方案

> 生成时间：2026-09-09 ｜ 依据：全量代码审读（src/ 全部模块、pages/、tests/、CI、Dockerfile、docker-compose.yml、README）
>
> 定位：把「功能完整的线上 Demo」推向「企业级 ready 的作品集项目」。每一项都标注**面试可讲点**——重构不只是改代码，更是把"我知道生产环境差距在哪"变成可展示的事实。

---

## 一、现状评估

**一句话结论：功能满分、算法层有真实亮点，但工程基础约 60 分——短板集中在安全、API 规范、配置管理，而不是检索算法。**

### 已达标项（不用动，这是底气）

| 维度 | 现状 |
|------|------|
| 架构 | LangGraph StateGraph + 双入口分工（RAG 快路径 / Agent 慢路径） |
| 检索 | BM25(jieba) + 向量 + RRF，A/B 测试后可配置切换；BM25 索引带失效缓存 |
| 评估 | RAGAS 四指标 + 多配置对比 + 历史存档（业界标准口径，不是自造轮子） |
| 可观测 | Langfuse @observe + callback，token/延迟进面板 |
| 引用防编造 | extract_cited_sources 只让模型给序号、代码映射真实来源 |
| 测试/CI | 6 个确定性测试（不调 LLM 不需要 key），GitHub Actions 已通 |
| 安全意识 | 三层审查（黑名单→敏感文件→白名单），路径段匹配防误杀 |
| 工程细节 | 惰性初始化（CI/测试友好）、语义缓存失效钩子、Chroma 单例防崩 |

### 短板总览

| 维度 | 现状 | 企业级标准 | 差距等级 |
|------|------|-----------|:---:|
| API 认证 | 无，任何人可调 /chat 读文件 | 至少 API Key + 限流 | 🔴 |
| work_dir 控制 | 客户端任意指定 + 硬编码个人桌面路径 | 服务端白名单决定 | 🔴 |
| Docker 打包 | `COPY . .` 无 .dockerignore | .env/data 绝不进镜像 | 🔴 |
| 请求韧性 | 同步阻塞、无超时无重试 | 超时+退避重试+并发控制 | 🔴 |
| 配置管理 | config.py 模块常量，缺 key 运行时才崩 | pydantic-settings 启动校验 | 🟡 |
| API 规范 | 裸返回 dict、无统一错误处理 | 统一响应体+异常处理器+request_id | 🟡 |
| 分层 | src/ 平铺 20 个文件，pages 直调底层 | api/service/infra 分层 | 🟡 |
| 测试覆盖 | 只有纯逻辑单测，API 层零覆盖 | TestClient 集成测试 + coverage 进 CI | 🟡 |
| LLM 韧性 | 裸调 OpenAI SDK，失败直接抛给用户 | 统一客户端封装+重试+token 记账 | 🟡 |
| 依赖锁定 | requirements 全是 `>=`，构建不可复现 | pyproject + lock 文件 | 🟢 |
| 监控指标 | 只有 Langfuse trace | +Prometheus /metrics（QPS/延迟/成本） | 🟢 |

---

## 二、P0 —— 安全与正确性硬伤（第 1 周初做完，约 2-3 天）

> 判断标准：**这四项任何一项被面试官或攻击者发现，都会让项目的"企业级"叙事当场穿帮**。先修它们，再谈架构。

### P0-1 API 认证 + 限流

- **问题**：`/chat`、`/chat/stream` 完全无认证。任何拿到地址的人都能让 Agent 读白名单内的文件、消耗你的 DeepSeek token（真金白银）。
- **改法**：
  1. FastAPI 加 `APIKeyHeader` 依赖，校验 `X-API-Key`（配置进 .env，`API_AUTH_KEY`），未带 key 返回 401。
  2. 加简单限流：`slowapi`（令牌桶）或自写中间件，如每 IP 每分钟 10 次。
  3. `/health` 保持无认证（探活惯例），其余端点全部挂依赖。
- **验收**：无 key curl 返回 401；带 key 正常；1 秒内打 20 次返回 429。
- **面试可讲**：「API 暴露的 Agent 能操作文件系统，我把它当成生产服务对待——认证、限流、审计一样不缺，因为 Agent 的权限就是用户的权限。」

### P0-2 work_dir 收权（最容易被追问穿帮的一项）

- **问题**：三个硬伤叠加——
  1. `ChatRequest.work_dir` 由**客户端任意指定**，服务端照单全收；
  2. `safety.py` 的白名单 `ALLOWED_DIRS = ["C:\\Users", "/home", os.getcwd()]`——**整个用户目录都在范围内**，等于沙箱只防系统目录不防用户数据；
  3. 默认值硬编码 `C:/Users/24162/Desktop`，**个人路径已提交到公开 GitHub**。
- **改法**：
  1. work_dir 改为**服务端配置**（`AGENT_WORK_DIR` 环境变量），从请求模型中删除；多用户场景下由服务端按会话分配。
  2. 白名单收敛：默认只允许 `AGENT_WORK_DIR` 一个根，`C:\Users` 全目录白名单删除（或改为仅本机 CLI 模式启用）。
  3. 代码里所有硬编码个人路径清除（`dev_agent_langgraph.py` 的 run_agent/stream_agent 默认参数、server.py 的 Field default）。
- **验收**：`grep -rn "24162" src/` 零命中；传任意 work_dir 的请求被拒绝；Agent 只能访问配置的目录。
- **面试可讲**：「我做过一次威胁建模：Agent 工具的路径校验层如果信任客户端输入，三层审查就是装饰。所以我把信任边界从『客户端说了算』移到『服务端配置决定』。」

### P0-3 .dockerignore + Docker 泄密修复

- **问题**：Dockerfile `COPY . .` 且项目**没有 .dockerignore**——`.env`（两把 API key）、`data/`（个人知识库、上传文件、评估记录）、`.git/` 全部会被打进镜像。一旦镜像推到 registry 就是密钥泄露。
- **改法**：
  1. 新建 `.dockerignore`：`.env`、`data/`、`.git/`、`__pycache__/`、`.pytest_cache/`、`notes/`、`docs/`、`knowledge/`（按需）。
  2. docker-compose 挂载 `.env` 改为 `env_file` 注入环境变量而非拷文件进容器（现在已是 env_file，保持）；宿主机 Desktop 挂载（`C:/Users/24162/Desktop:/app/host-desktop`）从 compose 默认配置移除，改为可选 profile。
- **验收**：`docker build` 后 `docker run --rm image ls -a /app` 看不到 .env / data。
- **面试可讲**：「密钥进镜像是容器安全的经典事故源，镜像分层历史即使删掉文件也能挖出来——所以必须在构建上下文就排除。」

### P0-4 请求韧性：超时 + 重试 + 并发保护

- **问题**：`/chat` 是同步 def 端点，LLM 调用无超时（云端 DeepSeek 默认能挂 600s），一个慢请求占住一个 worker 线程；无重试，API 抖动直接把异常抛给用户；Ollama 有 120s 超时但云端没有。
- **改法**：
  1. OpenAI SDK 调用统一加 `timeout=60, max_retries=2`（SDK 自带指数退避）。
  2. Agent 图整体加单请求超时（uvicorn 层或 asyncio.wait_for，同步端点可先用 `uvicorn --timeout-keep-alive` + SDK timeout 兜底）。
  3. `/chat` 改 async def + `run_in_executor`，或至少把 uvicorn workers 参数写进部署文档。
- **验收**：拔网线/填错 key 时，接口 60s 内返回结构化错误而不是无限挂起。
- **面试可讲**：「LLM API 是最不可靠的外部依赖，超时和退避重试是标配；没有它，上游一抖你的 P99 就爆炸。」

---

## 三、P1 —— 企业级骨架（第 1-2 周，约 5-8 天）

### P1-1 配置层：config.py → pydantic-settings

- **问题**：现在是模块级常量，缺 key 时不报错，直到用户提问才崩；无类型校验；环境（dev/prod）无法区分；阈值、开关散落。
- **改法**：`src/config.py` 改为 `BaseSettings`：必填项（`DEEPSEEK_API_KEY`）声明为必填，`Settings()` 在启动时实例化——缺配置**fail-fast**；用 `Field(ge=1, le=20)` 校验 TOP_K 之类；区分 `APP_ENV=dev|prod`。
- **迁移注意**：项目里 20+ 处 `from src.config import X`——保留旧常量名做兼容层（从 settings 实例取值），或一次性全改（机械但安全）。
- **验收**：删掉 .env 里的 key 启动 → 立刻报清晰错误并退出，而不是等到第一次提问。
- **面试可讲**：「12-factor 的 Config 原则：配置和代码分离、启动时校验。fail-fast 比运行时才崩省掉的排查时间是以小时计的。」

### P1-2 API 规范化

- **问题**：接口裸返回 dict；无全局异常处理（未捕获异常→500 带堆栈）；无 request_id（日志串不起来）；无 CORS；无版本化。
- **改法**（都在 `src/api/server.py` + 新增中间件）：
  1. 统一响应体 `{code, message, data}` + 全局 exception_handler（业务错 4xx、未知错 500 但不漏堆栈）。
  2. request_id 中间件（uuid4 → 响应头 X-Request-ID → 日志前缀），与 Langfuse session_id 打通。
  3. 路由挂 `/api/v1` 前缀；CORS 白名单配置化。
  4. 结构化日志：logging 改 JSON 格式（python-json-logger），字段含 request_id/latency/tokens。
- **验收**：随便构造一个错误请求，返回的是规范 JSON 而不是 HTML 堆栈页；同一 request_id 能串起访问日志和 Langfuse trace。
- **面试可讲**：「request_id 是排障的锚点——我把 HTTP 层的 request_id 透传进 Langfuse 的 session_id，一个请求从网关到 LLM 调用一条线看完。」

### P1-3 分层重构（企业级最直观的信号）

- **问题**：`src/` 平铺 20 个文件，Streamlit pages 直接调 `rag_agent`/`vector_store` 底层函数；`hybrid_retriever.py` 里还混着 Agent 工具函数（`search_knowledge`/`add_knowledge`）和 RRF 检索两件事。
- **改法**（移动文件 + 改 import，不改行为）：

  ```
  src/
  ├── api/            # FastAPI 路由、中间件、schemas（pydantic 模型）
  ├── service/        # rag_service.py / agent_service.py / kb_service.py（业务编排）
  ├── core/           # config.py / logging.py / security.py（安全审查）
  ├── infra/          # vector_store.py / database.py / embeddings.py / query_cache.py
  ├── retrieval/      # hybrid_retriever.py / chunker.py / parser.py
  ├── agent/          # LangGraph 图、工具定义
  └── llm/            # 统一 LLM 客户端封装（见 P1-5）
  ```

- **迁移注意**：tests、mcp_server.py、scripts/seed.py 的 import 同步改；一次一个模块，改完跑 `pytest` 全绿再动下一个。
- **验收**：pages/ 里不再出现 `from src.vector_store import ...` 这类直调；pytest 全绿；线上 Streamlit Demo 行为不变。
- **面试可讲**：「分层不是为了好看——Web 入口换掉时（比如以后做小程序），service 层原封不动。企业里前端迭代速度远快于业务层。」

### P1-4 测试补齐

- **问题**：现有 6 个测试只覆盖纯逻辑（safety/cache/citations/tokenize/state），**API 层零测试**；CI 无覆盖率、无静态检查。
- **改法**：
  1. FastAPI `TestClient` 集成测试：/chat 用 mock LLM（不花钱）、401/429 路径、work_dir 拒绝路径——这些恰好是 P0 修复的回归保护。
  2. 检索质量回归集：固定小语料 + 固定 10 问，断言 top-1 命中（防检索代码改坏）。
  3. CI 加 `pytest --cov=src --cov-fail-under=50` + `ruff check` + `mypy src/`（先宽松配置，只查关键模块）。
  4. pre-commit（ruff + ruff-format），本地提交即拦截。
- **验收**：CI 面板显示覆盖率数字；故意改坏检索排序 → 回归测试红。
- **面试可讲**：「我的 CI 有三层网：静态检查拦低级错、单测拦逻辑错、检索回归集拦『代码没错但质量退化』——最后这个是 RAG 项目特有的。」

### P1-5 LLM 调用韧性封装 + prompt 外置

- **问题**：`rag_agent.py` 里裸用 OpenAI SDK；system prompt 硬编码在代码里（Langfuse 说按 prompt 版本聚合，但 prompt 根本没有版本）；Agent 与 RAG 两条路径各自调 LLM，超时/重试/记账逻辑无法统一。
- **改法**：
  1. `src/llm/client.py`：统一 `chat(messages, stream=False, timeout, max_retries)`，内部做超时、重试、token 用量累计（按 session 记到 SQLite 或日志），cloud/ollama 后端切换收敛到这里。
  2. prompt 外置：`prompts/rag_system.md`、`prompts/agent_system.md`，代码读文件并带版本注释；可选接 Langfuse prompt management。
- **验收**：改 prompt 不用动 Python 代码；日志/面板能看到每次调用的 token 消耗与成本估算。
- **面试可讲**：「prompt 就是 LLM 应用的源代码——源代码不能埋在函数体里。外置 + 版本化之后，评估面板才能回答『这版 prompt 比上版好多少』。」

---

## 四、P2 —— 进阶加分（按需挑选，每项 1-3 天）

### P2-1 大文件解析异步化
现在上传文档在请求线程里同步跑 OCR/切分/嵌入，大 PDF 会卡住 Streamlit 页面几分钟。改法：解析任务丢 `BackgroundTasks`/线程池，`documents.status` 已有 processing→ready 状态机，前端轮询即可。**不必上 Celery。**
→ 面试点：异步任务 + 状态机 + 幂等（重复上传同名文件的处理策略）。

### P2-2 依赖锁定
requirements.txt 全是 `>=`，今天能装明天未必。改法：迁 `pyproject.toml`，用 `uv lock` 或 `pip-compile` 生成 lock 文件，Dockerfile 安装走 lock。
→ 面试点：可复现构建（"我这条 pipeline 任何机器上构建结果一致"）。

### P2-3 Prometheus /metrics
用 `prometheus-fastapi-instrumentator`：QPS、延迟分位（P50/P95/P99）、LLM token 成本计数器、语义缓存命中率。Grafana 本地起一个面板截图进 README。
→ 面试点：Langfuse 管 trace，Prometheus 管 metrics——两条观测支柱各管什么，这正是企业里的标准分工。

### P2-4 Docker 生产化
多阶段构建（builder 装依赖 → runtime 只拷 venv）、非 root 用户、`HEALTHCHECK` 指向 /health、镜像 tag 用 git sha 而非 latest。
→ 面试点：镜像从 1.2GB 减到 ~400MB 的数字很直观。

### P2-5 数据层演进评估（写文档即可，不必实施）
SQLite + Chroma 本地目录 → 多实例部署时必须迁 Postgres + pgvector（或 Qdrant）。现阶段**只写一页迁移评估文档**（何时迁、迁什么、代价），面试被问"并发上来了怎么办"时直接掏出来。
→ 面试点：知道何时**不**迁移，比无脑上 Postgres 更显判断力。

---

## 五、P3 —— 明确不做（防止过度设计）

| 不做的事 | 理由 |
|----------|------|
| 微服务拆分 / K8s | 单体应用流量撑不起，拆了只会增加运维面，面试官反而会追问"为什么拆" |
| Celery + Redis + RabbitMQ | BackgroundTasks/线程池足够，消息队列在这个量级是负资产 |
| Streamlit 换 React 前端 | 换汤不换药，投入产出比极低；Streamlit 对"演示型知识库"是正确选型 |
| 自建评估框架替换 RAGAS | RAGAS 恰恰是加分项（业界标准口径），换掉是倒退 |

> 原则：**企业级 ≠ 堆技术**。每一项基础设施都要能回答"解决了我项目的什么真实问题"，答不上来的就不做。

---

## 六、执行路线

| 阶段 | 内容 | 工作量 | 完成标志 |
|------|------|:---:|----------|
| 第一阶段：堵漏 | P0-1 ~ P0-4 | 2-3 天 | 无 key 返回 401；work_dir 收权；docker 镜像无密钥；超时重试生效 |
| 第二阶段：立骨 | P1-1 ~ P1-5 | 5-8 天 | 配置 fail-fast；统一响应体+request_id；src 分层完成且 pytest 全绿；覆盖率进 CI |
| 第三阶段：增肌 | P2 按需选 2-3 项 | 3-5 天 | README 新增「工程化」章节：指标面板截图、覆盖率徽章、迁移评估文档 |

**执行纪律**：
1. 开 `refactor/enterprise` 分支，一个 P 项一个 commit，改坏可回滚。
2. 动手顺序永远是**先补测试、再动代码**（尤其 P1-3 分层重构前，P1-4 的 API 测试要就位）。
3. 每完成一项，跑一次 `pytest` + 手工过一遍线上 Demo 核心路径（上传→问答→评估），保证 Streamlit Cloud 不炸。

---

## 七、面试价值映射（做完之后的叙事升级）

| 重构项 | 面试官问「你和玩具项目的区别」时的回答素材 |
|--------|--------------------------------------------|
| P0-1/P0-2 | "我做过威胁建模：Agent 有文件系统权限，认证和工作目录必须服务端说了算" |
| P0-4/P1-5 | "LLM API 是最不可靠依赖，我统一封装了超时/退避/token 记账" |
| P1-1 | "配置 fail-fast——12-factor 原则，缺 key 启动即报错" |
| P1-2 | "request_id 从 HTTP 层透传到 Langfuse，一个请求全链路一条线" |
| P1-3 | "分层让 Web 入口可替换——service 层不知道 Streamlit 的存在" |
| P1-4 | "三层测试网：静态检查 / 单测 / RAG 检索质量回归集" |
| P2-3/P2-5 | "知道什么时候上 Postgres、什么时候不上——容量判断比堆技术重要" |

**简历联动**：项目一「企业级知识库场景，以线上 Demo 验证」的诚实定位不变；重构后可补一句「并按企业级标准完成安全加固（认证/限流/沙箱收权）、API 规范化与分层重构，CI 覆盖率 X%」——把"未落地企业"的短板转成"我清楚差距并补齐了"的加分项。

---

## 八、风险提示

1. **重构不改行为**：P1-3 分层只挪文件改 import，任何"顺手优化"都单独开 commit，防止牵连检索质量。
2. **线上兼容**：Streamlit Cloud 部署读的是 GitHub main——合并前在本地 Docker 里完整验证一遍。
3. **`git grep 24162`**：个人路径已进 git 历史，代码里清除即可，历史记录是否清洗（filter-repo）看你意愿——公开仓库建议至少把未来提交清理干净。
