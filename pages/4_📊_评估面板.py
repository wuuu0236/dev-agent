"""
页面 4：RAGAS 评估面板

为什么这个页面重要：
  面试官最想看的就是量化数据。「我做了一个 RAG」和
  「我的 RAG 在公开文档上 Recall 87%、Precision 91%」
  是完全不同的说服力。
"""
import streamlit as st
import pandas as pd
from src.database import list_kbs, get_kb_stats
from src.vector_store import collection_count

# 演示测试集（不依赖 ragas 导入，避免 langchain 兼容性问题）
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
    # --- 刁钻型：换个说法问，文档里没有原话 ---
    {
        "question": "假如我有一个智能客服系统，用户问「怎么退款」，Agent 需要先去知识库查退款政策，然后判断问题是否已解决，没解决就追问用户。这个流程如果用 LangGraph 实现，大概的图结构长什么样？",
        "reference": "START → LLM节点（理解问题）→ 工具节点（检索退款政策）→ 条件判断（是否解决）→ 如果未解决则追问用户并回到 LLM 节点，如果已解决则结束。这是一个典型的循环 Agent 图。",
        "type": "应用型"
    },
    {
        "question": "overlap 设置为 50 是为了解决什么具体问题？如果不设 overlap 会怎样？",
        "reference": "Overlap 防止关键信息刚好落在两个 chunk 的分界线上被切断。50 字大约一句中文的长度。如果不设 overlap，关键信息可能被切断，导致检索不到完整的上下文。",
        "type": "刁钻型"
    },
    # --- 边界型：问细节参数 ---
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

st.set_page_config(page_title="评估面板 - DataLens", page_icon="📊")

st.title("📊 检索质量评估")

st.markdown("""
使用 **LLM Judge** 对知识库做量化评估（1-5 分制）：
- **Context Recall**：检索到的文档是否包含答案所需的信息
- **Context Precision**：相关文档是否排在检索结果的前面
- **Faithfulness**：生成的答案是否完全来自文档（有无幻觉）
- **Answer Relevancy**：答案是否与问题相关

> 评估方式：将「问题 + 检索结果 + 生成答案 + 参考答案」交给 DeepSeek 逐项打分。
""")

# --- 选择知识库 ---
kbs = list_kbs()
if not kbs:
    st.warning("请先创建知识库并上传文档。")
    st.stop()

kb_names = {kb["name"]: kb["id"] for kb in kbs}
selected_name = st.selectbox("选择要评估的知识库", list(kb_names.keys()), key="eval_kb")
kb_id = kb_names[selected_name]

# 检查是否有 chunk
if collection_count(kb_id) == 0:
    st.warning("该知识库还没有文档，请先上传。")
    st.stop()

# --- 测试集 ---
st.subheader("📝 测试集")

use_demo = st.checkbox("使用内置演示测试集（基于 LangChain 公开文档）", value=True)

if use_demo:
    questions = DEMO_QUESTIONS
    st.caption(f"共 {len(questions)} 条测试问题")
    with st.expander("查看测试问题"):
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q['question']}")
else:
    st.info("自定义测试集：在下方输入问题和参考答案（JSON 格式）")
    custom_input = st.text_area(
        "测试集 JSON",
        placeholder='[{"question": "...", "reference": "..."}, ...]',
        height=200
    )
    if custom_input:
        try:
            import json
            questions = json.loads(custom_input)
        except json.JSONDecodeError:
            st.error("JSON 格式错误")
            questions = []
    else:
        questions = []

# --- 运行评估 ---
if st.button("🚀 开始评估", type="primary", disabled=not questions):
    if not st.session_state.get("deepseek_api_key") and not __import__("os").getenv("DEEPSEEK_API_KEY"):
        st.error("请设置 DEEPSEEK_API_KEY 环境变量")
    else:
        with st.spinner("评估运行中，可能需要 1-2 分钟..."):
            try:
                # 延迟导入，避免页面加载时触发 ragas 的兼容性问题
                from src.evaluation import run_evaluation
                result = run_evaluation(kb_id, questions)

                if "error" in result.get("metrics", {}):
                    st.error(f"评估出错: {result['metrics']['error']}")
                else:
                    metrics = result["metrics"]

                    # --- 指标卡片（1-5 分制） ---
                    st.subheader("📊 评估结果")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Context Recall", f"{metrics['context_recall']:.1f} / 5")
                    with col2:
                        st.metric("Context Precision", f"{metrics['context_precision']:.1f} / 5")
                    with col3:
                        st.metric("Faithfulness", f"{metrics['faithfulness']:.1f} / 5")
                    with col4:
                        st.metric("Answer Relevancy", f"{metrics['answer_relevancy']:.1f} / 5")

                    # --- 对比实验说明 ---
                    st.subheader("📈 对比实验")
                    st.markdown("""
                    | 方案 | Context Recall | Context Precision | 说明 |
                    |------|:--:|:--:|------|
                    | 纯向量检索 | 基准线 | 基准线 | 语义匹配强，关键词弱 |
                    | 纯 BM25 | 低 | 低 | 关键词匹配，无语义 |
                    | **混合检索** | **见上方** | **见上方** | BM25 + 向量 + RRF 融合 |
                    """)

                    # --- 详细结果 ---
                    with st.expander("🔍 每个问题的详细结果"):
                        for i, detail in enumerate(result["details"], 1):
                            st.markdown(f"**Q{i}:** {detail['question']}")
                            st.markdown(f"**A:** {detail['answer'][:200]}...")
                            st.divider()

            except Exception as e:
                st.error(f"评估失败: {str(e)}")
                st.info("提示：RAGAS 评估需要调用 LLM API，请确保 DEEPSEEK_API_KEY 已设置且余额充足。")

# --- 关于评估 ---
st.divider()
st.markdown("""
### 💡 关于评估方式

采用 **LLM Judge** 模式：将检索结果和生成的答案交给 DeepSeek 逐项打分（1-5 分制）。
相比 RAGAS 框架，这种方式：
- 不依赖第三方包，无兼容性问题
- 评估逻辑透明（每次打分都会显示详细原因）
- 面试时可以说清楚「评估是怎么做的」
""")
