"""
页面 1：知识库管理
"""
import streamlit as st
from src.database import create_kb, list_kbs, delete_kb, get_kb_stats
from src.vector_store import create_collection, delete_collection, collection_count

st.set_page_config(page_title="知识库管理 - DataLens", page_icon="📚")

st.title("📚 知识库管理")

# --- 创建知识库 ---
with st.expander("➕ 创建新知识库", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("知识库名称", placeholder="例如：LangChain 技术文档")
    with col2:
        desc = st.text_input("描述（可选）", placeholder="简要描述这个知识库的内容")

    if st.button("创建", type="primary"):
        if name.strip():
            kb = create_kb(name.strip(), desc.strip())
            create_collection(kb["id"])
            st.success(f"✅ 知识库「{name}」创建成功！")
            st.rerun()
        else:
            st.error("请输入知识库名称")

# --- 知识库列表 ---
st.subheader("📋 我的知识库")
kbs = list_kbs()

if not kbs:
    st.info("还没有知识库，点击上方「创建新知识库」开始。")
else:
    for kb in kbs:
        stats = get_kb_stats(kb["id"])
        chroma_count = collection_count(kb["id"])

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown(f"**{kb['name']}**")
            st.caption(f"{kb['description']} | 创建于 {kb['created_at']}")
        with col2:
            st.metric("文档", stats["doc_count"])
        with col3:
            st.metric("Chunk", chroma_count)
        with col4:
            if st.button("🗑️ 删除", key=f"del_{kb['id']}"):
                delete_collection(kb["id"])
                delete_kb(kb["id"])
                st.warning(f"已删除知识库「{kb['name']}」")
                st.rerun()

        st.divider()
