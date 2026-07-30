"""
文档解析器：支持 PDF、Word、TXT、Markdown、CSV，以及**多模态图片 + 扫描版 PDF 的本地 OCR**。

PDF  → PyMuPDF (fitz)，按页提取文本；文字为空的页（扫描版）自动用 RapidOCR 识别
Word → python-docx，按段落提取文本
图片 → RapidOCR 本地 OCR 提取文字（纯 CPU，数据不出域）；保留原图路径供视觉模型读取
TXT/MD/CSV → 直接读取

返回的块统一带上 type / image 字段，便于检索与多模态问答溯源：
  - type="text"：普通文本块，image=None
  - type="image"：来自图片或扫描页，image 为原图（或渲染页）的本地路径

OCR 依赖 rapidocr_onnxruntime；若未安装或识别失败，图片块降级为占位文本，
保证上传/索引流程不崩溃（向后兼容，云端 Demo 不受影响）。
"""
import os
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

from src.config import ENABLE_OCR

# 图片扩展名（小写、带点）
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _ocr_image(path: str):
    """真实 OCR（RapidOCR，纯 CPU、轻量）。

    返回拼接后的文字；未安装 / 失败 / 无文字时返回 None（由调用方兜底）。
    """
    if not ENABLE_OCR:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        result, _ = engine(path)
        if result:
            text = "\n".join(line[1] for line in result).strip()
            return text or None
    except Exception:
        return None
    return None


def parse_pdf(file_path: str) -> list[dict]:
    """解析 PDF，按页提取文本；空文本页（扫描版）用 OCR 兜底。"""
    results = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if not text:
            # 扫描版 / 图片型页：渲染成图再 OCR
            try:
                pix = page.get_pixmap(dpi=150)
                img_path = str(Path(file_path).with_suffix(f".page{page_num + 1}.png"))
                pix.save(img_path)
                ocr_text = _ocr_image(img_path)
                # OCR 临时图用完即删，避免堆积
                try:
                    os.remove(img_path)
                except Exception:
                    pass
                if ocr_text:
                    results.append({
                        "text": ocr_text,
                        "page": page_num + 1,
                        "source": Path(file_path).name,
                        "type": "image",      # 标记来自扫描/OCR
                        "image": None,        # 扫描页无原图，仅用 OCR 文字
                    })
                    continue
            except Exception:
                pass
            # 既无文本也无 OCR：跳过该页
            continue
        results.append({
            "text": text,
            "page": page_num + 1,
            "source": Path(file_path).name,
            "type": "text",
            "image": None,
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
    return [{
        "text": full_text,
        "page": None,
        "source": Path(file_path).name,
        "type": "text",
        "image": None,
    }]


def parse_txt(file_path: str) -> list[dict]:
    """解析纯文本 / Markdown / CSV 文件"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return []
    return [{
        "text": text,
        "page": None,
        "source": Path(file_path).name,
        "type": "text",
        "image": None,
    }]


def parse_image(file_path: str) -> list[dict]:
    """解析图片：本地 RapidOCR 提取文字；失败降级为占位文本。

    原图路径保留在 image 字段，供本地视觉模型（Ollama minicpm-v 等）直接读取。
    """
    source = Path(file_path).name
    if ENABLE_OCR:
        ocr_text = _ocr_image(file_path)
        if ocr_text:
            return [{
                "text": ocr_text,
                "page": 1,
                "source": source,
                "type": "image",
                "image": str(file_path),   # 保留原图，供视觉模型读取
            }]
    # 降级：无 OCR 结果时给占位文本，避免检索空块
    return [{
        "text": "[图片内容待 OCR / 本地视觉模型识别]",
        "page": 1,
        "source": source,
        "type": "image",
        "image": str(file_path),
    }]


def parse_file(file_path: str) -> list[dict]:
    """统一入口：根据文件后缀分发给对应的解析器"""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".txt", ".md", ".csv"]:
        return parse_txt(file_path)
    elif ext in _IMAGE_EXTS:
        return parse_image(file_path)
    else:
        # 未知类型：返回占位块，避免上游因异常中断
        return [{
            "text": f"[暂不支持的文件类型 {ext}]",
            "page": 1,
            "source": Path(file_path).name,
            "type": "text",
            "image": None,
        }]
