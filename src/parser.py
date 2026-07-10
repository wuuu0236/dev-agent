"""
文档解析器：支持 PDF、Word、TXT、Markdown、CSV

PDF  → PyMuPDF (fitz)
Word → python-docx
TXT/MD/CSV → 直接读取

返回统一的格式：[{text, page, source}, ...]
"""
import fitz  # PyMuPDF
from docx import Document
from pathlib import Path


def parse_pdf(file_path: str) -> list[dict]:
    """解析 PDF，按页提取文本"""
    results = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if text:
            results.append({
                "text": text,
                "page": page_num + 1,
                "source": Path(file_path).name
            })
    doc.close()
    return results


def parse_docx(file_path: str) -> list[dict]:
    """解析 Word 文档，按段落提取文本"""
    doc = Document(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not full_text:
        return []
    # Word 没有页码概念，把整个文档当一个段落
    return [{"text": full_text, "page": None, "source": Path(file_path).name}]


def parse_txt(file_path: str) -> list[dict]:
    """解析纯文本文件"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return []
    return [{"text": text, "page": None, "source": Path(file_path).name}]


def parse_file(file_path: str) -> list[dict]:
    """统一入口：根据文件后缀分发给对应的解析器"""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".txt", ".md", ".csv"]:
        return parse_txt(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
