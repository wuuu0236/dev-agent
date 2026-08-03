"""
页面 4：RAGAS 评估面板（百分制 + 配置对比）

为什么这个页面重要：
  面试官最想看的就是量化数据。「我做了一个 RAG」和
  「我的 RAG 在公开文档上 Precision 87%、Relevancy 83%」
  是完全不同的说服力。

两种评估引擎：
  · RAGAS（默认）—— 业界标准框架，0-100 百分制，支持「同一测试集多组配置对比」+ 历史存档
  · 手写 LLM Judge（对照/教学）—— 自实现四维打分，与 RAGAS 同方法论，1-5 分 ×20 转百分制
"""
import json
import streamlit as st
from src.database import list_kbs, get_kb_stats
from src.vector_store import collection_count, check_embedding_dim

# 演示测试集（基于 LangChain 公开文档，题型覆盖定义/对比/推理/应用/刁钻/细节）
DEMO_QUESTIONS = [
    # --- 定义型：问「是什么」---
    {
        "question": "什么是 LangGraph？它是用来做什么的？",
        "reference": "LangGraph 是 LangChain 团队开发的低级别编排框架，用于构建和管理长时间运行的、有状态的 AI Agent，通过有向图来建模 Agent 行为。",
        "type": "定义型"
    },
    {
        "question": "LangGraph 中的 StateGraph 是什么？创建 StateGraph 需要哪几步？",
        "reference": "StateGraph 是 LangGraph 的核心类，用于定义 Agent 的状态图。创建步骤：定义 State、创建实例、添加节点、添加边、编译、调用。",
        "type": "定义型"
    },
    # --- 对比型：问「A 和 B 的区别」---
    {
        "question": "LangChain 的 Chain 和 LangGraph 的 Graph 有什么本质区别？什么场景下应该用 LangGraph 而不是 Chain？",
        "reference": "Chain 是线性的 A→B→C，Graph 支持分支、循环和条件逻辑。当 Agent 需要「思考→行动→观察→再思考」的循环时应该用 LangGraph。",
        "type": "对比型"
    },
    {
        "question": "向量检索和 BM25 检索的优缺点分别是什么？",
        "reference": "向量检索擅长语义匹配但专有名词效果差；BM25 擅长精确关键词匹配但不理解语义。两者互补。",
        "type": "对比型"
    },
    # --- 推理型：问「为什么」---
    {
        "question": "为什么 RAG 系统需要对文档进行分块？如果 chunk 切得太大或太小会有什么后果？",
        "reference": "因为 LLM 上下文有限且大块包含无关信息干扰判断。太小会切碎上下文，太大会塞进无关内容导致检索不准。推荐 500 字左右。",
        "type": "推理型"
    },
    {
        "question": "为什么 RRF 算法不直接用两个系统的原始分数做加权平均，而是基于排名来计算？",
        "reference": "因为向量分数和 BM25 分数的尺度不同，不在同一量级上，无法直接比较。RRF 只关心排名，避免了分数归一化的问题。",
        "type": "推理型"
    },
    # --- 应用型 ---
    {
        "question": "假如我有一个智能客服系统，用户问「怎么退款」，Agent 需要先去知识库查退款政策，然后判断问题是否已解决。这个流程如果用 LangGraph 实现，大概的图结构长什么样？",
        "reference": "START → LLM节点（理解问题）→ 工具节点（检索退款政策）→ 条件判断（是否解决）→ 如果未解决则追问用户并回到 LLM 节点，如果已解决则结束。这是一个典型的循环 Agent 图。",
        "type": "应用型"
    },
    # --- 刁钻型：换个说法问，文档里没有原话 ---
    {
        "question": "overlap 设置为 50 是为了解决什么具体问题？如果不设 overlap 会怎样？",
        "reference": "Overlap 防止关键信息刚好落在两个 chunk 的分界线上被切断。50 字大约一句中文的长度。如果不设 overlap，关键信息可能被切断，导致检索不到完整的上下文。",
        "type": "刁钻型"
    },
    # --- 细节型：问细节参数 ---
    {
        "question": "RRF 公式中的平滑常数 k 一般取多少？它有什么作用？",
        "reference": "k 通常取 60，作用是避免分母太小导致排名靠后的文档分数异常放大，让融合结果更稳定。",
        "type": "细节型"
    },
    {
        "question": "在 LangGraph 的 State 定义中，add_messages 这个 reducer 做了什么？如果不加 Annotated 注解会怎样？",
        "reference": "add_messages 将新消息追加到列表中而不是替换。如果不加 Annotated 注解，State 字段会被直接覆盖而不是追加。",
        "type": "细节型"
    }
]

# 与 knowledge/ 公开文档库内容对齐的演示测试集（配合该库可跑出高分为例）
DEMO_QUESTIONS_MATCHED = [
    {"question": "AI Agent 的三个核心要素是什么？",
     "reference": "AI Agent 的三个核心要素是 LLM（大脑，理解意图、决定行动）、工具 Tools（执行实际操作）、循环 Loop（思考→行动→观察）。", "type": "定义型"},
    {"question": "RAG 解决了 LLM 的哪两个核心问题？",
     "reference": "RAG 解决了 LLM 的知识截止日期（不知道训练后的新知识）和幻觉（可能编造不存在的事实）两个核心问题。", "type": "定义型"},
    {"question": "MCP 协议是谁提出的？它要解决什么核心问题？",
     "reference": "MCP（模型上下文协议）由 Anthropic 提出，让 AI 模型自动发现和调用外部工具，解决不同应用工具调用方式不统一、不能跨应用复用的问题。", "type": "定义型"},
    {"question": "LangGraph 的核心概念是什么？",
     "reference": "LangGraph 的核心是 StateGraph：把 Agent 流程定义成节点和边的有向图，状态在节点之间流转。", "type": "定义型"},
    {"question": "Dev Agent 项目使用什么技术栈？",
     "reference": "Dev Agent 是基于 LangGraph + FastAPI 的 AI 开发助手，能操作本地文件、搜索知识库，支持 HTTP API 和 Docker 部署。", "type": "定义型"},
]

st.set_page_config(page_title="评估面板 - DataLens", page_icon="📊")
st.title("📊 检索质量评估")

st.markdown("""
- **RAGAS 四维指标**（0-100 百分制）：Context Recall / Precision（检索质量）· Faithfulness（有无幻觉）· Answer Relevancy（是否切题）
- **配置对比**：同一测试集跑多组 top_k，量化「调参到底有没有用」
- **历史存档**：每次对比自动保存，可回看「上次 vs 这次」
""")


# ---------- 辅助函数 ----------

_METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
}


def _show_metrics(metrics: dict):
    """四张百分制指标卡。"""
    cols = st.columns(4)
    for col, name in enumerate(["context_precision", "context_recall", "faithfulness", "answer_relevancy"]):
        v = metrics.get(name, 0)
        cols[col].metric(_METRIC_LABELS[name], f"{v:.1f}%")


def _show_details(details: list[dict]):
    """逐题明细。"""
    with st.expander("🔍 逐题明细"):
        for i, d in enumerate(details, 1):
            st.markdown(f"**Q{i}:** {d['question']}")
            if d.get("answer"):
                st.markdown(f"*A:* {d['answer'][:220]}")
            if d.get("scores"):
                st.caption("   ".join(f"{_METRIC_LABELS[k]}: {v:.1f}%" for k, v in d["scores"].items() if k in _METRIC_LABELS))
            st.divider()


def _show_comparison(cmp: dict):
    """配置对比结果：并排表格 + 逐题对比。"""
    import pandas as pd
    rows = []
    for res in cmp.get("results", []):
        row = {"配置": res["name"]}
        row.update({_METRIC_LABELS[k]: f"{v:.1f}%" for k, v in res.get("metrics", {}).items()})
        rows.append(row)
    if not rows:
        return
    st.markdown(f"**对比结果（{cmp.get('test_size', 0)} 题 · {cmp.get('timestamp', '')}）**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("🔍 逐题对比"):
        results = cmp.get("results", [])
        if not results:
            return
        n = len(results[0].get("details", []))
        for i in range(n):
            st.markdown(f"**Q{i + 1}:** {results[0]['details'][i]['question']}")
            for res in results:
                sc = res["details"][i].get("scores", {})
                line = "  ".join(f"{_METRIC_LABELS[k]}: {v:.1f}%" for k, v in sc.items() if k in _METRIC_LABELS)
                st.caption(f"· {res['name']}: {line}")
            st.divider()


# ---------- 选择知识库 ----------

kbs = list_kbs()
if not kbs:
    st.warning("请先在「知识库管理」中创建知识库并上传文档。")
    st.stop()

kb_names = {kb["name"]: kb["id"] for kb in kbs}
selected_name = st.selectbox("选择要评估的知识库", list(kb_names.keys()), key="eval_kb")
kb_id = kb_names[selected_name]

stats = get_kb_stats(kb_id)
if stats["total_chunks"] == 0:
    st.warning("该知识库还没有文档，请先上传。")
    st.stop()

compat, compat_msg = check_embedding_dim(kb_id)
if not compat:
    st.error(f"⚠️ {compat_msg}")
st.caption(f"📊 {stats['doc_count']} 个文档 | {stats['total_chunks']} 个 chunk")


# ---------- 测试集 ----------

st.subheader("📝 测试集")
demo_choice = st.selectbox(
    "测试集来源",
    ["内置·LangGraph 深度题（10 题）", "内置·匹配公开文档库（5 题）", "自定义 JSON"],
    help="「匹配公开文档库」的题直接对准 knowledge/ 目录内容，配合「公开文档」库可跑出高分为例；自定义 JSON 用于评估你自己的知识库。",
)

if demo_choice == "内置·LangGraph 深度题（10 题）":
    questions = DEMO_QUESTIONS
    st.caption(f"共 {len(questions)} 条（题型覆盖定义/对比/推理/应用/刁钻/细节）")
    with st.expander("查看测试问题"):
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q['question']}（{q['type']}）")
elif demo_choice == "内置·匹配公开文档库（5 题）":
    questions = DEMO_QUESTIONS_MATCHED
    st.caption(f"共 {len(questions)} 条，直接对齐 knowledge/ 公开文档内容")
    with st.expander("查看测试问题"):
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q['question']}")
else:
    st.info("自定义测试集：JSON 格式 [{\"question\": ..., \"reference\": ...}, ...]")
    custom_input = st.text_area("测试集 JSON", placeholder='[{"question": "...", "reference": "..."}]', height=180)
    if custom_input:
        try:
            questions = json.loads(custom_input)
            if not questions:
                st.warning("JSON 数组为空")
        except json.JSONDecodeError:
            st.error("JSON 格式错误")
            questions = []
    else:
        questions = []

if not questions:
    st.stop()


# ---------- 引擎选择 + 运行 ----------

engine = st.radio("评估引擎", ["RAGAS（0-100，推荐）", "手写 LLM Judge（对照）"], horizontal=True)


def _run_with_guard(fn, *a, **kw):
    """统一错误兜底：检索/评估失败给可读提示。"""
    try:
        return fn(*a, **kw)
    except Exception as e:
        msg = str(e)
        if "dimension" in msg.lower() or "embedding" in msg.lower():
            st.error(f"❌ 知识库 embedding 维度与当前模型不匹配（详见上方提示）。请用当前模型重新上传文档。")
        else:
            st.error(f"❌ 评估失败: {type(e).__name__}: {msg}")
        return None


if engine.startswith("RAGAS"):
    # 延迟导入，避免页面加载时引入 RAGAS/评估重依赖
    from src.evaluation_ragas import run_ragas_eval, compare_configs

    st.subheader("🏃 单配置评估")
    top_k = st.slider("检索 top_k", 1, 10, 5, key="ragas_topk")
    if st.button("🚀 跑 RAGAS 评估", type="primary", disabled=not questions):
        with st.spinner("RAGAS 每题会多次调用 LLM，10 题约 1-2 分钟..."):
            res = _run_with_guard(run_ragas_eval, kb_id, questions, top_k=top_k)
            if res and "error" not in res:
                _show_metrics(res["metrics"])
                _show_details(res["details"])
            elif res:
                st.error(f"❌ {res['error']}")

    st.divider()

    st.subheader("📈 配置对比（对比评分）")
    st.caption("同一测试集、多组 top_k 各跑一遍——量化「检索深度调到多少最好」。对比可保存进历史。")
    topk_options = st.multiselect("对比哪些 top_k", [3, 5, 8, 10], default=[3, 5], key="ragas_topk_cmp")
    save_hist = st.checkbox("保存结果到历史存档", value=True)
    if st.button("🚀 跑配置对比", type="primary", disabled=not questions or len(topk_options) < 2):
        configs = [{"name": f"top_k={k}", "top_k": k} for k in topk_options]
        with st.spinner(f"对比中（{len(configs)} 组配置，耗时约 = 单配置 × {len(configs)}）..."):
            cmp = _run_with_guard(compare_configs, kb_id, questions, configs, save=save_hist)
            if cmp and "error" not in cmp:
                _show_comparison(cmp)
            elif cmp:
                st.error(f"❌ {cmp['error']}")

else:
    # 手写 LLM Judge（对照/教学）
    from src.evaluation import run_evaluation

    st.caption("自实现四维 LLM Judge，1-5 分 ×20 转百分制，与 RAGAS 同方法论，可对照两种引擎的结果。")
    if st.button("🚀 跑手写评估", type="primary", disabled=not questions):
        with st.spinner("评估中（每题一次检索 + 一次打分）..."):
            res = _run_with_guard(run_evaluation, kb_id, questions)
            if res and "error" not in res:
                metrics = {k: round(v * 20, 1) for k, v in res["metrics"].items()}
                details = [{**d, "scores": {k: round(v * 20, 1) for k, v in d.get("scores", {}).items()}} for d in res["details"]]
                _show_metrics(metrics)
                _show_details(details)
            elif res:
                st.error(f"❌ {res['error']}")


# ---------- 历史存档 ----------

st.divider()
st.subheader("🗂️ 历史存档")
from src.evaluation_ragas import list_history, load_history

history = list_history()
if not history:
    st.caption("暂无存档。跑「配置对比」并勾选保存后，结果会出现在这里。")
else:
    labels = {f"[{h['timestamp']}] {', '.join(h['configs'])}（{h['test_size']}题）": h["file"] for h in history}
    sel_label = st.selectbox("查看历史对比结果", list(labels.keys()))
    data = load_history(labels[sel_label])
    if data:
        _show_comparison(data)
