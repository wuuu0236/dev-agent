"""入口层的权限粘合：FastAPI Depends + Streamlit session 检查。

为什么两套入口共用一个文件：同一条权限规则（谁能看哪个库）必须在 Web API、
Streamlit、React 前端三处表现一致。规则本身在 permissions.py，这里只负责
「从各自的请求上下文里取出当前用户」并转成对应框架的错误。

⚠️ Streamlit 相关函数内部 import streamlit：本模块同时被 FastAPI 加载，
   顶层 import streamlit 会把 streamlit 变成 API 服务的硬依赖。
"""
from fastapi import Header, HTTPException

from src.auth.permissions import check_kb_access
from src.auth.service import decode_token


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """FastAPI 依赖：从 `Authorization: Bearer <token>` 解出用户 payload。

    用 Header(default=None) + 手动 401，而不是 Header(...)：后者在缺头时由
    FastAPI 框架抛 422（参数校验错误），而"没登录"应该是 401。前端据此判断
    是"该跳登录页"还是"参数写错了"。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少或格式错误的认证头")
    payload = decode_token(authorization[7:])
    if not payload:
        raise HTTPException(status_code=401, detail="token 无效或已过期")
    return payload


def user_id_of(user: dict) -> str:
    """从 token payload 取用户 id（JWT 标准字段是 sub）。"""
    return user.get("sub") or ""


def require_kb_access(kb_id: str, user: dict, required: str = "viewer"):
    """通用权限检查，不通过抛 403。

    刻意不区分「库不存在」与「无权限」——两种情况都回 403，避免用这个接口
    探测出别人有哪些知识库 id。
    """
    if not check_kb_access(user_id_of(user), kb_id, required):
        raise HTTPException(status_code=403, detail=f"没有该知识库的 {required} 权限")


def require_login(user: dict, required: str = "viewer"):
    """只检查全局身份（不针对某个库）。用于 /api/ask 这类带 kb_id 但
    主要以「登录」为门槛的端点。"""
    if not user_id_of(user):
        raise HTTPException(status_code=401, detail="未登录")


# ================================================================
# Streamlit 侧
# ================================================================

def current_user() -> dict | None:
    """取当前登录用户（未登录返回 None）。"""
    import streamlit as st

    return st.session_state.get("user")


def render_login_and_stop():
    """未登录时渲染登录页并终止本次页面渲染。

    登录成功后写 session_state 并 st.rerun()，让页面从头走一遍——
    否则 Streamlit 会把「登录前已经执行的代码」和「登录后的页面」渲染在一起。
    """
    import streamlit as st

    st.title("登录 DataLens")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")
    if submitted:
        from src.auth.service import login

        result = login(username, password)
        if result:
            st.session_state["user"] = result["user"]
            st.session_state["token"] = result["token"]
            st.rerun()
        else:
            st.error("用户名或密码错误")
    st.stop()


def require_login_ui() -> dict:
    """页面顶部调用：返回当前用户；未登录则渲染登录页并停住。"""
    user = current_user()
    if not user:
        render_login_and_stop()
    return user


def require_kb_access_ui(kb_id: str, required: str = "viewer") -> None:
    """页面内针对某个知识库的权限检查，不通过就提示并停住。"""
    import streamlit as st

    user = require_login_ui()
    if not check_kb_access(user["id"], kb_id, required):
        st.error(f"没有该知识库的 {required} 权限")
        st.stop()
