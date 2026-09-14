"""
页面 1：知识库管理
"""
import shutil

import streamlit as st
from src.database import create_kb, list_kbs, delete_kb, get_kb_stats
from src.vector_store import create_collection, delete_collection, collection_count
from src.config import kb_upload_dir
from src.reindex import rebuild_kb, rebuildable_docs

st.set_page_config(page_title="知识库管理 - DataLens", page_icon="📚")

st.title("📚 知识库管理")

# 删除 / 重建的反馈：st.rerun() 会冲掉瞬时的 success / error 消息，
# 用一次性的 session_state 把消息带到下一轮渲染（与上传页同一套做法）。
for _level, _text in st.session_state.pop("kb_msgs", []):
    {"success": st.success, "warning": st.warning, "error": st.error}[_level](_text)

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

# --- 缓存：统计数字（chroma 查询最重）---
# 管理页每次交互整页重跑，会对每个知识库循环查 SQLite + chroma。
# 库一多，点一下就要查 N 次 chroma，这是本页卡顿主因。
# 只缓存数字（文档数 / chunk 数），5 秒内复用；列表不缓存——删库/建库后要即时刷新。
@st.cache_data(ttl=5)
def _cached_kb_stats(kb_id: str) -> dict:
    return get_kb_stats(kb_id)


@st.cache_data(ttl=5)
def _cached_collection_count(kb_id: str) -> int:
    return collection_count(kb_id)


# --- 知识库列表 ---
st.subheader("📋 我的知识库")
kbs = list_kbs()

if not kbs:
    st.info("还没有知识库，点击上方「创建新知识库」开始。")
else:
    for kb in kbs:
        stats = _cached_kb_stats(kb["id"])
        chroma_count = _cached_collection_count(kb["id"])

        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        with col1:
            st.markdown(f"**{kb['name']}**")
            st.caption(f"{kb['description']} | 创建于 {kb['created_at']}")
        with col2:
            st.metric("文档", stats["doc_count"])
        with col3:
            st.metric("Chunk", chroma_count)
        with col4:
            # 重建索引：改了分块 / 清洗 / 嵌入配置之后，让老文档也应用上新规则。
            # 前提是原始文件还在（2026-09-14 起上传的都会保留）。
            if st.button("♻️ 重建索引", key=f"rebuild_{kb['id']}",
                         help="用保留的原始文件，按当前的分块 / 清洗 / 嵌入配置重新处理该库全部文档"):
                ready, missing = rebuildable_docs(kb["id"])
                if not ready:
                    st.session_state["kb_msgs"] = [(
                        "error",
                        f"「{kb['name']}」无法重建：{len(missing)} 个文档的原始文件都不在了。"
                        f"它们是在「保留原文」改造之前入库的，只能重新上传。"
                    )]
                    st.rerun()

                log = st.empty()
                with st.spinner("正在重建索引…"):
                    result = rebuild_kb(kb["id"], progress=lambda m: log.write(m))

                msgs = [(
                    "success",
                    f"✅ 「{kb['name']}」重建完成：{result['ok']}/{result['total']} 个文档，"
                    f"共写入 {result['chunks']} 个 chunk"
                )]
                if result["missing"]:
                    preview = "、".join(result["missing"][:3])
                    more = f" 等 {len(result['missing'])} 个" if len(result["missing"]) > 3 else ""
                    msgs.append((
                        "warning",
                        f"⚠️ {len(result['missing'])} 个文档缺原始文件、未重建（它们的旧索引保持原样）：{preview}{more}"
                    ))
                for _name, _err in result["failed"]:
                    msgs.append(("error", f"❌ {_name}：{_err}"))
                st.session_state["kb_msgs"] = msgs
                _cached_kb_stats.clear()
                _cached_collection_count.clear()
                st.rerun()

            if st.button("🗑️ 删除", key=f"del_{kb['id']}"):
                delete_collection(kb["id"])
                delete_kb(kb["id"])
                # 连带清掉该库保留的原始文件目录。路径严格限定为 data/uploads/{kb_id}，
                # kb_id 取自数据库（8 位 uuid），不会波及其他目录。
                shutil.rmtree(kb_upload_dir(kb["id"]), ignore_errors=True)
                st.session_state["kb_msgs"] = [("warning", f"已删除知识库「{kb['name']}」")]
                st.rerun()

        st.divider()
