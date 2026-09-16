"""认证与授权（多用户）。

三个子模块，职责严格分开：
  service.py      —— 凭据本身：哈希、签发/校验 token、注册、登录
  permissions.py  —— 谁能看/改哪个知识库（角色层级判断）
  deps.py         —— 入口层的粘合：FastAPI Depends + Streamlit session 检查

为什么单独开一个包而不是塞进 database.py：数据库层管的是「怎么存」，这里管的是
「谁可以做什么」。混在一起会让 database.py 同时承担存储与安全两件事，改任一边
都要动同一个文件。
"""
