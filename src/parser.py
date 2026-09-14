"""
文档解析器：支持 PDF、Word、TXT、Markdown、CSV，以及**多模态图片 + 扫描版 PDF 的本地 OCR**。

PDF  → PyMuPDF (fitz)，按页提取文本；文字为空的页（扫描版）自动用 RapidOCR 识别
Word → python-docx，按**真实阅读顺序**提取段落 + 表格（表格序列化成行式自包含文本）
图片 → RapidOCR 本地 OCR 提取文字（纯 CPU，数据不出域）；保留原图路径供视觉模型读取
TXT/MD/CSV → 直接读取

返回的块统一带上 type / image 字段，便于检索与多模态问答溯源：
  - type="text"：普通文本块，image=None
  - type="image"：来自图片或扫描页，image 为原图（或渲染页）的本地路径

OCR 依赖 rapidocr_onnxruntime；若未安装或识别失败，图片块降级为占位文本，
保证上传/索引流程不崩溃（向后兼容，云端 Demo 不受影响）。
"""
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.config import ENABLE_OCR
from src.cleaner import clean_parsed

# 图片扩展名（小写、带点）
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 表格序列化：同一行内「列名：值」之间的分隔符
_TABLE_CELL_SEP = "；"
# 表格每行的前缀。必须与 cleaner 的 _NO_MERGE_PREFIX 兼容——
# 清洗层的 merge_broken_lines 会把「不以句末标点结尾」的行与下一行合并，
# 而单元格值常以 "12.5%" / "3.11" 这类非标点字符结尾，行结构会被吞掉。
# "-" 在 _NO_MERGE_PREFIX 里，天然免疫合并。
_TABLE_ROW_PREFIX = "- "
# 块与块之间的分隔（段落 / 表格）。必须用空行：单换行会让清洗层把
# 「不以标点结尾的段落」与下一个段落粘成一行。
_BLOCK_SEP = "\n\n"


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


def _iter_block_items(doc):
    """按文档**真实阅读顺序**产出 Paragraph / Table 两种块。

    为什么不能分别读 `doc.paragraphs` 和 `doc.tables`：
    python-docx 把这两者分开收集，各自内部有序，但合并后顺序就丢了——
    表格会被整体挪到文末。文档里 "段落 → 表格 → 段落" 这种交错一旦被压平，
    分块时上下文接不上，人看引用也判断不出表格在正文哪个位置。
    body 的 XML 子节点本身就是按阅读顺序排的，直接迭代它才拿得到真实顺序。

    注：python-docx 没有内置这个迭代器（官方 FAQ 给的就是这个 recipe）。
    """
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _cell_text(cell) -> str:
    """单元格文本压成一行：表格里换行会打断「列名：值」的配对。"""
    return re.sub(r"\s+", " ", cell.text or "").strip()


def _table_to_text(table) -> str:
    """把 Word 表格序列化成**行式自包含文本**（每行一个记录）。

    为什么不直接输出 Markdown 表格（`| 基金代码 | 年化收益 |`）：
      检索单元是 chunk（500 字 + 50 重叠），**不是整张表**。表格一旦被切开，
      表头大概率落不到同一块里——命中 `| 025490 | 12.5% |` 这一行时，模型
      既不知道 025490 是基金代码，也不知道 12.5% 是年化收益。更糟的是
      `|` 分隔符本身不携带语义，会稀释嵌入向量。
      行式「列名：值」让**任意一行单独拿出来都是自包含的**，这正是检索需要的形态。

    为什么每行前置 `- `：见 `_TABLE_ROW_PREFIX` 的注释（防清洗层合并）。

    降级策略：
      · 空行 / 全空单元格 → 跳过，不产生 `列名：` 这种空配对
      · 表头单元格为空 → 该列退回「第 N 列」，保证值不丢
      · 只有一行的表 → 没有列名可配，按原样输出该行
      · 整表无内容 → 返回空串（调用方跳过，不产生空块）
    """
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        seen_tc: set[int] = set()   # 横向合并会让同一个 <w:tc> 在多个列位重复出现
        for cell in row.cells:
            if id(cell._tc) in seen_tc:
                continue
            seen_tc.add(id(cell._tc))
            cells.append(_cell_text(cell))
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    header, data_rows = rows[0], rows[1:]
    if not data_rows:                      # 单行表：无列名可配，原样输出
        return _TABLE_ROW_PREFIX + _TABLE_CELL_SEP.join(c for c in header if c)

    lines: list[str] = []
    for row in data_rows:
        pairs: list[str] = []
        for i, value in enumerate(row):
            if not value:
                continue
            name = header[i] if i < len(header) and header[i] else f"第{i + 1}列"
            pairs.append(f"{name}：{value}")
        if pairs:
            lines.append(_TABLE_ROW_PREFIX + _TABLE_CELL_SEP.join(pairs))
    return "\n".join(lines)


def parse_docx(file_path: str) -> list[dict]:
    """解析 Word 文档：按阅读顺序提取段落与表格。

    ⚠️ 曾经只读 `doc.paragraphs` → **doc.tables 里的内容 100% 静默丢失**。
    python-docx 的 `doc.paragraphs` 只返回 body 直属的 `<w:p>`；表格段落嵌套在
    `<w:tbl>` 内，不在其中。而且这是**静默**的：文档照样标 ready，用户毫无感知，
    偏偏 Word 里信息密度最高的往往就是表格（参数表、对比矩阵、规格表）。
    实测（2×2 + 3×3 两张表）：7 个关键值全部丢失，13 个单元格内容一个不剩。
    """
    doc = Document(file_path)

    parts: list[str] = []
    for block in _iter_block_items(doc):
        if isinstance(block, Table):
            text = _table_to_text(block)
        else:
            text = block.text.strip()
        if text:
            parts.append(text)

    full_text = _BLOCK_SEP.join(parts).strip()
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
    """统一入口：根据文件后缀分发给对应的解析器。

    流水线：解析 → **清洗**（归一化 / 去页眉页脚 / 合并断行 / 丢垃圾块）→ 返回。
    清洗收口在这里而不是散在各调用方，保证任何入库路径（Web 上传、Agent 工具、
    脚本灌库）都跑不掉。可用 CLEANER_ENABLED=false 关闭做 A/B 对比。
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        parsed = parse_pdf(file_path)
    elif ext == ".docx":
        parsed = parse_docx(file_path)
    elif ext in [".txt", ".md", ".csv"]:
        parsed = parse_txt(file_path)
    elif ext in _IMAGE_EXTS:
        parsed = parse_image(file_path)
    else:
        # 未知类型：返回占位块，避免上游因异常中断
        return [{
            "text": f"[暂不支持的文件类型 {ext}]",
            "page": 1,
            "source": Path(file_path).name,
            "type": "text",
            "image": None,
        }]

    return clean_parsed(parsed)
