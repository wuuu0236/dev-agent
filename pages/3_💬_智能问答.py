"""
页面 3：智能问答
"""
import streamlit as st
from src.database import list_kbs, get_kb_stats
from src.rag_agent import rag_query
from src.config import TOP_K_RETRIEVE

st.set_page_config(page_title="智能问答 - DataLens", page_icon="💬")

st.title("💬 智能问答")

# --- 选择知识库 ---
kbs = list_kbs()
if not kbs:
    st.warning("请先在「知识库管理」中创建知识库并上传文档。")
    st.stop()

kb_names = {kb["name"]: kb["id"] for kb in kbs}
selected_name = st.selectbox("选择知识库", list(kb_names.keys()), key="qa_kb")
kb_id = kb_names[selected_name]

# 显示知识库统计
stats = get_kb_stats(kb_id)
if stats["total_chunks"] == 0:
    st.warning("该知识库还没有文档，请先上传。")
    st.stop()

st.caption(f"📊 {stats['doc_count']} 个文档 | {stats['total_chunks']} 个 chunk | 混合检索（BM25 + 向量）· 支持图片/扫描件 OCR")

# --- 侧边栏：模型设置（私有化 / 多模态）---
with st.sidebar.expander("⚙️ 模型设置", expanded=False):
    backend = st.selectbox(
        "推理后端", ["cloud", "ollama"],
        index=0 if st.session_state.get("qa_backend", "cloud") == "cloud" else 1,
        help="cloud = DeepSeek 等云端 API（默认）；ollama = 本地私有化/离线，需先安装并启动 Ollama",
    )
    st.session_state["qa_backend"] = backend
    vision_model = st.text_input(
        "视觉模型 (Ollama)", value=st.session_state.get("qa_vision", ""),
        help="如 minicpm-v:8b。填了才在命中图片块时调用本地视觉模型真·看图；"
             "留空则图片仅使用 OCR 文字（云端模式不支持图片直读，请勿填）",
    )
    st.session_state["qa_vision"] = vision_model
    if backend == "cloud" and vision_model:
        st.warning("⚠️ 云端文本模型不能读图，已忽略视觉模型设置，图片将只用 OCR 文字。")

# --- 聊天历史 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 引用来源"):
                for s in msg["sources"]:
                    page_info = f" — 第{s['page']}页" if s.get("page") else ""
                    badge = " 🖼️" if s.get("type") == "image" else ""
                    st.caption(f"• {s['source']}{page_info}{badge}")

# --- 输入区 ---
if query := st.chat_input("输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 显示 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            result = rag_query(
                kb_id, query, top_k=TOP_K_RETRIEVE,
                backend=st.session_state.get("qa_backend", "cloud"),
                vision_model=st.session_state.get("qa_vision", ""),
            )

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📖 引用来源"):
                for s in result["sources"]:
                    page_info = f" — 第{s['page']}页" if s.get("page") else ""
                    badge = " 🖼️" if s.get("type") == "image" else ""
                    st.caption(f"• {s['source']}{page_info}{badge}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

# --- 清空按钮 ---
if st.session_state.messages:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()
