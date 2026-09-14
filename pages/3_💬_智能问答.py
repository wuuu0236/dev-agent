"""
页面 3：智能问答
"""
import re

import streamlit as st
from src.database import list_kbs, get_kb_stats
from src.rag_qa import stream_rag_query
from src.citations import extract_cited_sources
from src.config import TOP_K_RETRIEVE

st.set_page_config(page_title="智能问答 - DataLens", page_icon="💬")

st.title("💬 智能问答")


# --- 引用来源渲染 ---
# 引用原文里可能带 markdown 特殊字符（#、*、> 等），转义后才能原样展示；
# 换行压成空格，避免把一段引用拆成多个 blockquote。
_MD_SPECIAL = re.compile(r"([\\*_`#\[\]|>])")


def _plain(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", (text or "").replace("\n", " "))


def _render_sources(sources: list) -> None:
    """渲染引用来源：文档名 + 页码 + 命中原文。

    此前只显示「文件名 + 页码」——用户看不到原文，也就无从判断模型是在照实回答、
    还是在拿别的内容硬编。**「验证」是产品闭环里最容易断的一环**，这里把它接上：
    展开引用就能逐句对照原文。
    """
    with st.expander(f"📖 引用来源（{len(sources)} 个文档）"):
        for s in sources:
            page_info = f" — 第{s['page']}页" if s.get("page") else ""
            badge = " 🖼️" if s.get("type") == "image" else ""
            snippets = s.get("snippets") or []
            if not snippets:
                # 旧缓存里的来源没有片段字段，退回旧展示（向后兼容）
                st.caption(f"• {s['source']}{page_info}{badge}")
                continue
            st.markdown(f"**{s['source']}**{page_info}{badge}")
            for snip in snippets:
                st.markdown(f"> {_plain(snip)}")


# --- 缓存：知识库列表 / 统计 ---
# Streamlit 每次交互都会重跑整页脚本，切下拉框、清空对话都会触发。
# 这两次 SQLite 查询单独看很轻，但次数一多页面就"不跟手"。
# 包一层 cache_data：5 秒内不重复查库（新传文档最多延迟 5 秒刷新，可接受）。
@st.cache_data(ttl=5)
def _cached_list_kbs():
    return list_kbs()


@st.cache_data(ttl=5)
def _cached_kb_stats(kb_id: str) -> dict:
    return get_kb_stats(kb_id)


# --- 选择知识库 ---
kbs = _cached_list_kbs()
if not kbs:
    st.warning("请先在「知识库管理」中创建知识库并上传文档。")
    st.stop()

kb_names = {kb["name"]: kb["id"] for kb in kbs}
selected_name = st.selectbox("选择知识库", list(kb_names.keys()), key="qa_kb")
kb_id = kb_names[selected_name]

# 显示知识库统计
stats = _cached_kb_stats(kb_id)
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
            _render_sources(msg["sources"])

# --- 输入区 ---
if query := st.chat_input("输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 显示 AI 回答（流式：检索 → 打字机输出）
    with st.chat_message("assistant"):
        status = st.status("检索知识库...", expanded=False)
        # 注入最近对话历史（除当前这条），让"那第二点呢"这类追问有上下文
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1] if m.get("content")
        ]
        gen, sources, contexts, retrieval_query = stream_rag_query(
            kb_id, query, top_k=TOP_K_RETRIEVE,
            backend=st.session_state.get("qa_backend", "cloud"),
            vision_model=st.session_state.get("qa_vision", ""),
            history=history,
        )
        status.update(label="生成回答...", state="running")
        answer = st.write_stream(gen)
        status.update(label="完成", state="complete")

        # 追问消解可见化：检索是拿改写后的 query 去的，不是用户原话。
        # 只在真的改写了才显示，单轮提问不打扰。
        if retrieval_query and retrieval_query != query:
            st.caption(f"🔍 追问消解后的检索用查询：{retrieval_query}")

        # 门控未通过（检索分数低于阈值）：这段回答不来自知识库，必须让用户知道，
        # 否则"没有依据的兜底回答"看起来和"有引用支撑的回答"一模一样。
        if not contexts:
            st.caption("⚠️ 未命中知识库（检索相关度低于阈值），以下回答不来自知识库文档")

        # 引用映射：回答里的 [n] → 真实来源（防 LLM 编造文件名/页码）；
        # 回答没标引用时回退到全部检索来源。两种都由 _render_sources 展开看原文。
        cited = extract_cited_sources(answer, contexts) if contexts else []
        display_sources = cited or sources
        if display_sources:
            _render_sources(display_sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": display_sources
    })

# --- 清空按钮 ---
if st.session_state.messages:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()
