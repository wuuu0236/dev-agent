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

## 每个改动的收尾清单

1. 全量测试全绿：
   ```bash
   pytest tests/ -q --ignore=tests/test_state_build.py --ignore=tests/test_smoke_imports.py
   ```
   （这两个文件被忽略是**环境问题**：本机隔离 venv 缺 langchain / langfuse，非代码问题。）
2. 若动了首页 / README 文案，或改了 `src/config.py`、评估指标定义：
   ```bash
   pytest tests/test_app_homepage.py -q
   ```
3. 静态检查（`pip install -r requirements-dev.txt` 里已含 pyflakes）：
   ```bash
   python -m pyflakes src/ pages/ app.py scripts/ tests/
   ```
   **要求零输出**。CI 里已挂成独立 job（`.github/workflows/ci.yml` 的 `lint`），
   历史告警在 2026-09-14 清过一次（原 17 条），之后再现就是新引入的。

   原则：**每条都要归到下面某一类，不许为了让输出变干净而把语义掩盖掉。**

   - `undefined name` / `redefinition` → **真问题，必须修**。这类错误能在一个后端分支里
     潜很久：`_get_ollama_client` 把 `OpenAI` 写成不存在的 `OpenAIClient`，云端路径全正常、
     只有选 ollama 时才 NameError，于是被宣传的「本地私有化 / 离线」一直是死的。
     而且**测试覆盖不到它**——那条路径在本机跑不起来。
   - `f-string is missing placeholders` → **先看它是不是漏了该填的值**，别直接删 `f`。
     `src/tools/safety.py` 那条拒绝信息原作「路径不在允许范围内」，是同函数四个拒绝分支里
     唯一不带上下文的；它的读者是 Agent，拿到后不知道该改哪个参数，只能原地重试或放弃。
     现修成带上被拒路径 + 白名单。
   - `imported but unused` → 先确认**全仓有没有人从本模块反向导入**它（可能是再导出），
     没有就删。**不要用 `# noqa: F401`**：那是 flake8 的功能，**pyflakes 根本不认**，
     写了照报。两个例外写法：
     - 真的是「导入即断言」（如 `tests/test_smoke_imports.py`）→ 用
       `importlib.import_module("x.y")`，把意图直写进代码，静态检查也读得懂。
     - 别留「给未来用」的转发：仓库里曾有一个 `extract_cited_sources` 的兼容再导出，
       注释写着"保持老路径可用"，但**加它的同一次提交**就把唯一的调用方改到了新路径
       —— 它从未被任何人用过。
   - `assigned to but never used` → **重点看**。多半意味着「文案里写了、代码里没用」：
     曾有一个 `NEAR_MISS_RATIO` 只出现在评估面板的界面上，分类逻辑从没读过它 ——
     界面上因此写着一条根本不存在的分界线。
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
