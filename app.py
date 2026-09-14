"""
DataLens — 个人 RAG 知识库平台

首页：项目介绍 + 导航入口
"""
import streamlit as st

st.set_page_config(
    page_title="DataLens - 智能知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 初始化数据库（只做一次）---
# Streamlit 每次交互都重跑整页脚本；init_db 是幂等的建表检查，没必要每次跑。
# cache_resource 让整个进程只初始化一次。
from src.database import init_db as _init_db


@st.cache_resource
def _init_db_once():
    _init_db()
    # 冷启动自动补演示知识库（scripts/seed.py，幂等）：
    # data/ 不进 git，新部署是空库；导入种子后评审打开即可直接问答。
    # 失败（如没配 embedding key）不阻塞启动，留待下次冷启动重试。
    try:
        from scripts.seed import ensure_seed_data
        ensure_seed_data(verbose=False)
    except Exception:
        pass


_init_db_once()

# --- 侧边栏 ---
st.sidebar.title("📚 DataLens")
st.sidebar.markdown("**智能知识库问答平台**")
st.sidebar.divider()
st.sidebar.markdown("""
### 功能导航
→ 左侧页面选择功能

### 技术栈
- 🔍 混合检索（BM25 + 向量）
- 🧠 LangGraph Agent（HTTP API 入口）
- 📊 RAGAS 评估（业界标准）
- 🖼️ 多模态 OCR（图片 / 扫描件）
- 🔒 私有化离线（Ollama 可选）
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
    | ❌ 假数据 | ✅ 上传**真实文档**，支持 PDF/Word/TXT + 图片(OCR)/扫描件 |
    | ❌ 没评估 | ✅ 内置 **RAGAS 评估面板**，四指标百分制量化检索与生成质量 |

    ### 核心技术亮点

    - **混合检索**：BM25 关键词 + 向量语义，RRF 融合
    - **多知识库**：每个知识库独立隔离，可按主题分别建库管理
    - **文档解析**：PDF / Word（含表格）/ Markdown / TXT / CSV，图片与扫描件走本地 OCR
    - **量化评估**：业界标准 RAGAS 四指标 —— Context Precision / Context Recall / Faithfulness / Answer Relevancy，0-100 百分制
    - **引用可核实**：回答标注引用序号 → 映射回真实来源 → 展开可见命中原文
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
