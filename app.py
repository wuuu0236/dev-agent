"""
DataLens — 生产级 RAG 知识库平台

首页：项目介绍 + 导航入口
"""
import streamlit as st

st.set_page_config(
    page_title="DataLens - 智能知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 初始化数据库 ---
from src.database import init_db
init_db()

# --- 侧边栏 ---
st.sidebar.title("📚 DataLens")
st.sidebar.markdown("**智能知识库问答平台**")
st.sidebar.divider()
st.sidebar.markdown("""
### 功能导航
→ 左侧页面选择功能

### 技术栈
- 🔍 混合检索（BM25 + 向量）
- 🧠 LangGraph Agent
- 📊 RAGAS 评估
- 🗄️ SQLite + Chroma
- 🚀 Streamlit 部署
""")

# --- 主页 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("📚 DataLens")
    st.subheader("上传文档 → 构建知识库 → 智能问答 → 量化评估")

    st.markdown("""
    ### 为什么做这个项目？

    大部分 RAG 项目的问题：**本地运行、假数据、没有量化指标**。
    DataLens 解决这三个问题：

    | 问题 | DataLens 的做法 |
    |------|----------------|
    | ❌ 本地玩具 | ✅ **线上部署**，面试官点开就能用 |
    | ❌ 假数据 | ✅ 上传**真实文档**，支持 PDF/Word/TXT |
    | ❌ 没评估 | ✅ 内置 **RAGAS 评估面板**，量化展示检索质量 |

    ### 核心技术亮点

    - **混合检索**：BM25 关键词 + 向量语义，RRF 融合 + Cross-Encoder 精排
    - **多知识库**：每个知识库独立隔离，支持多用户场景
    - **文档解析**：PDF（PyMuPDF）+ Word（python-docx）+ 文本
    - **量化评估**：RAGAS 四指标（Recall / Precision / Faithfulness / Relevancy）
    - **引用溯源**：每个回答标注来源文件和页码
    """)

with col2:
    st.info("""
    ### 🚀 快速开始

    1. **创建知识库** — 给它起个名字
    2. **上传文档** — PDF、Word、TXT 都支持
    3. **开始问答** — 用中文提问
    4. **查看评估** — RAGAS 量化数据
    """)

    st.success("""
    ### 💡 面试演示路径

    打开链接 → 上传文档
    → 提问看答案 + 引用
    → 评估面板看数据
    → 5 分钟展示完整能力
    """)

st.divider()
st.caption("Built with Streamlit + Chroma + DeepSeek | 吴永健 | 2026")
