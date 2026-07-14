"""
页面 2：文档上传
"""
import os
import streamlit as st
from src.database import list_kbs, add_document, update_document_status, list_documents
from src.parser import parse_file
from src.chunker import chunk_parsed
from src.vector_store import add_chunks, create_collection, collection_count
from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR

st.set_page_config(page_title="文档上传 - DataLens", page_icon="📄")

st.title("📄 文档上传")

# --- 选择知识库 ---
kbs = list_kbs()
if not kbs:
    st.warning("请先在「知识库管理」中创建知识库。")
    st.stop()

kb_names = {kb["name"]: kb["id"] for kb in kbs}
selected_name = st.selectbox("选择知识库", list(kb_names.keys()))
kb_id = kb_names[selected_name]

# --- 已上传文档列表（放在上面，方便看到状态） ---
st.subheader("📋 已入库文档")
docs = list_documents(kb_id)

if not docs:
    st.info("还没有上传文档。")
else:
    for doc in docs:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        status_map = {"ready": "✅", "processing": "⏳", "error": "❌", "empty": "⚠️"}
        with col1:
            st.markdown(f"{status_map.get(doc['status'], '❓')} **{doc['filename']}**")
        with col2:
            st.caption(f"{doc['chunk_count']} chunks")
        with col3:
            st.caption(doc["status"])
        with col4:
            if doc["status"] in ("processing", "error"):
                if st.button("🗑️ 删除", key=f"del_{doc['id']}"):
                    from src.database import get_connection
                    conn = get_connection()
                    conn.execute("DELETE FROM documents WHERE id = ?", (doc['id'],))
                    conn.commit()
                    conn.close()
                    st.rerun()

st.divider()

# --- 上传文件 ---
st.subheader("📤 上传新文档")
st.caption(f"支持格式：{', '.join(ALLOWED_EXTENSIONS)} | 单文件最大 {MAX_FILE_SIZE_MB}MB")

uploaded_files = st.file_uploader(
    "拖拽文件到此处或点击上传",
    type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files:
    if st.button("🚀 开始解析入库", type="primary"):
        success_count = 0
        fail_count = 0

        # 用一个进度条显示整体进度
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uf in enumerate(uploaded_files):
            status_text.text(f"正在处理: {uf.name} ({i+1}/{len(uploaded_files)})")
            progress_bar.progress((i) / len(uploaded_files))

            # 检查大小
            if uf.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"❌ {uf.name} 超过 {MAX_FILE_SIZE_MB}MB 限制")
                fail_count += 1
                continue

            # 保存到本地
            file_path = UPLOAD_DIR / uf.name
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())

            # 记录到数据库
            doc_id = add_document(kb_id, uf.name, uf.size)

            try:
                # 1. 解析文档
                status_text.text(f"📖 解析: {uf.name}")
                parsed = parse_file(str(file_path))

                if not parsed:
                    st.warning(f"⚠️ {uf.name} 没有可提取的文本内容")
                    update_document_status(doc_id, "empty")
                    fail_count += 1
                    continue

                # 2. 切块
                status_text.text(f"✂️ 切块: {uf.name} ({len(parsed)} 段落)")
                chunks = chunk_parsed(parsed)

                if not chunks:
                    st.warning(f"⚠️ {uf.name} 内容太短，无法分块")
                    update_document_status(doc_id, "empty")
                    fail_count += 1
                    continue

                # 3. 写入向量库
                status_text.text(f"🧮 向量化: {uf.name} ({len(chunks)} chunks)")
                add_chunks(kb_id, chunks)

                # 4. 更新状态
                update_document_status(doc_id, "ready", len(chunks))
                success_count += 1
                st.success(f"✅ {uf.name} → {len(parsed)} 段落 → {len(chunks)} chunks")

            except Exception as e:
                st.error(f"❌ {uf.name} 处理失败: {str(e)}")
                update_document_status(doc_id, "error")
                fail_count += 1

            finally:
                # 清理临时文件
                if file_path.exists():
                    os.remove(file_path)

        # 完成
        progress_bar.progress(1.0)
        status_text.text(f"✅ 完成！成功 {success_count} 个，失败 {fail_count} 个")
        st.rerun()

st.divider()
st.metric("知识库总 Chunk 数", collection_count(kb_id))
