# AGENTS.md

本仓库的强制约定。在此项目中工作的任何 Agent 都必须遵守。

## 注意事项

1. **改动即提交**：每次改动完成后，都必须创建一个对应的 Git commit，以便后续追踪和回滚。
2. **测试必须通过**：每次改动后，都必须编写或更新相关测试，并在交付给用户前，确保所有测试和验证全部通过。
3. **改动即推送**：commit 之后必须 `git push origin master`。

   线上 Demo（Streamlit Cloud）连的是 GitHub `master`，**靠 push 触发自动部署**。
   因此 **不推 = 线上默认落后于本地**，这不是可选项。曾经的实际后果：
   本地已修的 Word 表格 bug 在线上演示时仍然存在、首页评估口径线上仍是已废弃的
   LLM-as-Judge —— 都只因为"忘了推"。

4. **对外文案必须与代码一致**：`app.py` 首页与 README 里宣称的能力，必须真的可用、
   且真的在线上存在。**只在本机运行的能力**（HTTP API / MCP 工具服务 / 文件操作 Agent /
   Ollama）**必须显式标注**，不得混进"技术栈"这类会让读者默认线上也有的清单。
   守卫测试：`pytest tests/test_app_homepage.py`（改首页或 README 文案后必须跑）。

5. **不许让文档与行为脱钩**：改代码时若让某条描述失效（如关掉某个开关、重命名模块），
   必须**全仓 grep 同一关键词**把同源的表述一起改掉，不能只改被报告的那一处。

6. **导入不得要求凭据**：模块级不许构造 API 客户端。

   `OpenAI(api_key="")` 会当场抛 `OpenAIError: Missing credentials`（openai 2.x / 3.x
   行为一致，已实测）。写在模块级就等于「**没有凭据就不能 import 本模块**」，于是所有
   不需要调 API 的用法都会在导入期崩：CI（无任何密钥）、MCP、只跑清洗/分块的离线流程。
   实际后果：`src/embeddings.py` 的模块级客户端让三个测试文件在**收集阶段**集体
   ImportError，test job 红了一个月（2026-09-14）。本地有 `.env` 所以完全看不出来。
   → 一律**延迟构造**（首次调用时才建），写法参考 `src/embeddings.py::_get_client`。
   **凭据缺失该在调用时报错，而不是在 import 时报错——那是两件事。**
   守卫测试：`pytest tests/test_import_no_credentials.py`。

   同一条原则也适用于**测试**：不许依赖本机有没有配 key。默认路径下要把外部调用替换成
   替身（或在夹具里把 key 设成一个假值），把「没 key 时的降级行为」留一条用例显式覆盖。

## CI 红了怎么查

CI 默认只给一句 `Process completed with exit code 2.`，按这个顺序拿原文：

1. **读 annotations**（匿名可读，不需要任何凭据）：
   ```bash
   curl -s "https://api.github.com/repos/wuuu0236/dev-agent/actions/runs/<RUN_ID>/jobs"
   curl -s "https://api.github.com/repos/wuuu0236/dev-agent/check-runs/<JOB_ID>/annotations"
   ```
   `.github/workflows/ci.yml` 的测试步骤会把 pytest 的关键失败行拼成一条注解，
   依赖自检也会发 `::error::` —— 所以这两步的失败原文在这里都看得到。
   ⚠️ **job 日志走 `/actions/jobs/<id>/logs` 需要鉴权（匿名 403）**，别在那浪费时间。
   ⚠️ 匿名调用 GitHub API 限 60 次/小时，别在循环里高频查。

2. **看清退出码**：**2 = 收集错误**（ImportError / 语法错误 / 导入期崩），
   **1 = 真的有测试失败**。（历史上两者叠过一次：新的收集错误把旧的真失败挡住，
   只修一层会以为没效果。）

3. 注解信息不够时，按这个顺序怀疑（每一条都真实发生过）：
   - `requirements-dev.txt` 漏包 → `python scripts/check_ci_deps.py` 直接列出缺哪个；
     注意**判断依据不是"装起来快不快"，而是"测试有没有 import 到它"**；
   - 导入期要求凭据 → 见注意事项 6；
   - **平台假设**：写死 `\` 或 `C:\` 的测试在本机 Windows 永远绿、在 CI 的 ubuntu 必红
     （`test_path_within_boundary` 那次），写路径一律用 `os.sep` 拼；
   - 语法用了 3.12+ 的写法而 CI 跑 3.11（本机 3.13 看不出来）。

## 每个改动的收尾清单

**顺序是有理由的：从便宜到贵、从确定到不确定。** 静态检查一秒内给出行号，全量测试要几秒，
push 不可逆。

1. 静态检查（`pip install -r requirements-dev.txt` 里已含 pyflakes）：
   ```bash
   python -m pyflakes src/ pages/ app.py scripts/ tests/
   ```
   **要求零输出**。CI 里已挂成独立 job（`.github/workflows/ci.yml` 的 `lint`，排在 `test` 前），
   历史告警在 2026-09-14 清过一次（原 17 条），之后再现就是新引入的。

   原则：**每条都要归到下面某一类，不许为了让输出变干净而把语义掩盖掉。**

   - `undefined name` / `redefinition` → **真问题，必须修**。这类错误能在一个后端分支里
     潜很久：`_get_ollama_client` 把 `OpenAI` 写成不存在的 `OpenAIClient`，云端路径全正常、
     只有选 ollama 时才 NameError，于是被宣传的「本地私有化 / 离线」一直是死的。
     而且**测试覆盖不到它**——那条路径在本机跑不起来。
   - `f-string is missing placeholders` → **先看它是不是漏了该填的值**，别直接删 `f`。
     `src/tools/safety.py` 那条拒绝信息原作「路径不在允许范围内」，是同函数四个拒绝分支里
     唯一不带上下文的；它的读者是 Agent，拿到后不知道该改哪个参数，只能原地重试或放弃。
     现修成带上被拒路径 + 白名单。**判据：接收者拿这条消息能不能自纠？**
   - `imported but unused` → 先确认**全仓有没有人从本模块反向导入**它（可能是再导出），
     没有就删。**不要用 `# noqa: F401`**：那是 flake8 的功能，**pyflakes 根本不认**，
     写了照报；`from a import x as x` 那种显式再导出它同样不认。两个例外写法：
     - 真的是「导入即断言」（如 `tests/test_smoke_imports.py`）→ 用
       `importlib.import_module("x.y")`，把意图直写进代码，静态检查也读得懂。
     - 别留「给未来用」的转发：仓库里曾有一个 `extract_cited_sources` 的兼容再导出，
       注释写着"保持老路径可用"，但**加它的同一次提交**就把唯一的调用方改到了新路径
       —— 它从未被任何人用过。
   - `assigned to but never used` → **重点看**。多半意味着「文案里写了、代码里没用」：
     曾有一个 `NEAR_MISS_RATIO` 只出现在评估面板的界面上，分类逻辑从没读过它 ——
     界面上因此写着一条根本不存在的分界线。

   补充：**删"未使用导入"前要看行号，不能只 grep** —— `grep '\bos\.'` 会命中 docstring
   里的说明文字，看着像在用，其实没在用。
2. 全量测试全绿 —— **而且要在「没有凭据」的前提下也绿**：
   ```bash
   # ① 本机正常跑
   pytest tests/ -q --ignore=tests/test_state_build.py --ignore=tests/test_smoke_imports.py

   # ② 等价 CI（CI 上没有任何密钥）。变量显式置空即可，不用动 .env ——
   #    load_dotenv() 默认不覆盖已存在的环境变量，所以置空不会被 .env 里的真值补回来。
   EMBEDDING_API_KEY= DEEPSEEK_API_KEY= RERANK_API_KEY= LLM_API_KEY= \
     pytest tests/ -q --ignore=tests/test_state_build.py --ignore=tests/test_smoke_imports.py
   ```
   （①②忽略的两个文件是**本机环境问题**：隔离 venv 缺 langchain / langfuse，非代码问题。）

   **②不能省。** 本机 `.env` 里有真 key，所以「测试偷偷依赖凭据」这类问题本地永远看不出来：
   2026-09-14 一次性暴露过 7 条 —— `rerank()` 在没 key 时会**提前返回粗排顺序**，
   于是所有 monkeypatch 了打分函数的用例静默失效（6 条）；另 1 条忘了替换客户端，
   真去构造 `OpenAI` 直接抛异常。**它们在本机全是绿的。**
3. 若动了首页 / README 文案，或改了 `src/config.py`、评估指标定义：
   ```bash
   pytest tests/test_app_homepage.py -q
   ```
4. `git commit`：提交信息用中文，写清「**问题 → 改动 → 测试/实测结果**」，与既有风格一致。
5. `git push origin master`，然后**核对远程真实 hash**：
   ```bash
   git ls-remote origin refs/heads/master
   ```
   - ⚠️ 别用 `git status` 的 ahead/behind 判断推没推上去：在本机沙箱环境中
     `.git/refs/remotes` 的写入被隔离，会永远显示 `origin/master: gone`。
   - ⚠️ 若 push 卡住或被杀（SIGTERM），先试非交互模式，**不要去折腾代理**：
     ```bash
     GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never git push origin master
     ```
     原因是本机凭据走 GCM，它会弹 GUI 输凭据 —— 无头环境里必然卡死。
