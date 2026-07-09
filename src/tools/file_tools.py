"""
文件工具 —— 让 Agent 能读取本地文件
这是 Agent 的"眼睛"。所有路径操作都经过安全检查。
"""

import os
from .safety import check_path_safety


def list_files(directory: str = ".") -> str:
    """列出目录下的所有文件（含安全检查）"""
    safe, reason = check_path_safety(directory)
    if not safe:
        return f"安全拦截: {reason}"

    if not os.path.exists(directory):
        return f"目录 '{directory}' 不存在"

    files = os.listdir(directory)
    if not files:
        return f"目录 '{directory}' 是空的"

    result = [f"目录 '{directory}' 包含 {len(files)} 个文件/文件夹:"]
    for f in sorted(files):
        full_path = os.path.join(directory, f)
        tag = "[文件夹]" if os.path.isdir(full_path) else "[文件]"
        result.append(f"  {tag} {f}")
    return "\n".join(result)


def read_file(filepath: str) -> str:
    """读取文件内容（含安全检查）"""
    safe, reason = check_path_safety(filepath)
    if not safe:
        return f"安全拦截: {reason}"

    if not os.path.exists(filepath):
        return f"文件 '{filepath}' 不存在"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        if len(lines) > 100:
            # 文件太长只返回前后各 30 行 + 行数统计
            head = "\n".join(lines[:30])
            tail = "\n".join(lines[-30:])
            return (
                f"文件 '{filepath}' 共 {len(lines)} 行（以下为前 30 行和后 30 行）:\n"
                f"=== 开头 ===\n{head}\n\n=== 结尾 ===\n{tail}"
            )
        return f"文件 '{filepath}' 共 {len(lines)} 行:\n{content}"

    except UnicodeDecodeError:
        return f"文件 '{filepath}' 不是文本文件，无法读取"
    except Exception as e:
        return f"读取文件 '{filepath}' 出错: {str(e)}"


def search_in_files(directory: str, keyword: str) -> str:
    """在目录中搜索包含关键词的文件（含安全检查）"""
    safe, reason = check_path_safety(directory)
    if not safe:
        return f"安全拦截: {reason}"

    if not os.path.exists(directory):
        return f"目录 '{directory}' 不存在"

    results = []
    for root, _, files in os.walk(directory):
        # 跳过隐藏目录和 venv
        if any(skip in root for skip in [".git", "venv", "__pycache__", "node_modules"]):
            continue
        for f in files:
            if f.endswith((".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg")):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    if keyword.lower() in content.lower():
                        # 统计出现次数
                        count = content.lower().count(keyword.lower())
                        results.append(f"  {filepath} (出现 {count} 次)")
                except Exception:
                    pass

    if not results:
        return f"在 '{directory}' 中没有找到包含 '{keyword}' 的文件"

    return f"找到 {len(results)} 个包含 '{keyword}' 的文件:\n" + "\n".join(results)
