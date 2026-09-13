"""cleaner 数据清洗单元测试（纯逻辑，不调 API / 不碰数据库）。"""
import src.cleaner as C


# --- 断行合并 ---
def test_merge_chinese_broken_line():
    """中文句子被硬换行切断 → 合并（这是 PDF 最典型的噪声）"""
    src = "AI Agent 的三个核心要素是：\nLLM、工具和循环。"
    assert C.merge_broken_lines(src) == "AI Agent 的三个核心要素是：LLM、工具和循环。"


def test_keep_sentence_end():
    """上一行以句号结尾 → 不合并，保留段落结构"""
    src = "这是第一句话。\n这是第二句话。"
    assert C.merge_broken_lines(src) == "这是第一句话。\n这是第二句话。"


def test_keep_heading_and_list():
    """标题行、列表项不与上一行合并"""
    assert C.merge_broken_lines("正文内容\n# 一级标题") == "正文内容\n# 一级标题"
    assert C.merge_broken_lines("步骤如下\n- 第一步") == "步骤如下\n- 第一步"
    assert C.merge_broken_lines("步骤如下\n1. 第一步") == "步骤如下\n1. 第一步"


def test_merge_english_word_with_space():
    """英文单词之间补空格，避免把两个词粘在一起"""
    assert C.merge_broken_lines("Retrieval Augmented\nGeneration 是什么") == "Retrieval Augmented Generation 是什么"


def test_hyphen_line_break_no_space():
    """行尾是连字符（英文断词）→ 直接相连，不补空格。
    注：真正的断词还原（Aug-\\nmented → Augmented）需要词典，此处不处理。"""
    assert C.merge_broken_lines("Retrieval Aug-\nmented") == "Retrieval Aug-mented"


# --- 归一化 ---
def test_normalize_compress_blank_lines():
    assert C.normalize("第一段\n\n\n\n\n第二段") == "第一段\n\n第二段"


def test_normalize_strip_control_chars():
    assert C.normalize("正常\x00文\x07字") == "正常文字"


def test_normalize_trailing_spaces():
    assert C.normalize("行一   \n行二\t") == "行一\n行二"


# --- 页眉页脚 ---
def test_find_header_footer():
    """4 页里有 4 页都出现同一行 → 判为页眉页脚"""
    pages = [
        "XX公司 内部资料\n正文A内容\n第 1 页",
        "XX公司 内部资料\n正文B内容\n第 2 页",
        "XX公司 内部资料\n正文C内容\n第 3 页",
        "XX公司 内部资料\n正文D内容\n第 4 页",
    ]
    noise = C.find_header_footer(pages)
    assert "XX公司 内部资料" in noise


def test_no_header_footer_when_few_pages():
    """页数 < 3 不做判定（样本不足，避免误杀正文）"""
    assert C.find_header_footer(["重复行\n正文", "重复行\n正文"]) == set()


def test_strip_header_footer_removes_page_number():
    pages = ["正文A\n12", "正文B\n13"]
    out = C.strip_header_footer(pages)
    assert out == ["正文A", "正文B"]


# --- 垃圾块 ---
def test_is_trivial():
    assert C.is_trivial("")
    assert C.is_trivial("12")        # 纯页码
    assert C.is_trivial("------")    # 纯符号
    assert C.is_trivial("。")         # 过短
    assert not C.is_trivial("MCP 是 Anthropic 提出的开放协议")
    assert not C.is_trivial("Agent 三要素")


# --- 组合清洗 ---
def test_clean_text_pipeline():
    dirty = "RAG 是什么？\n它解决的问题是幻觉。\n\n\n\n第二段开始"
    assert C.clean_text(dirty) == "RAG 是什么？\n它解决的问题是幻觉。\n\n第二段开始"


def test_clean_parsed_drops_junk_blocks():
    parsed = [
        {"text": "这是有实质内容的段落", "page": 1, "source": "a.md", "type": "text"},
        {"text": "12", "page": 2, "source": "a.md", "type": "text"},      # 页码 → 丢
        {"text": "-----", "page": 3, "source": "a.md", "type": "text"},   # 噪声 → 丢
    ]
    out = C.clean_parsed(parsed)
    assert len(out) == 1
    assert out[0]["text"] == "这是有实质内容的段落"


def test_clean_parsed_keeps_structure_fields():
    """清洗不能丢掉 source / page / type 等溯源字段"""
    parsed = [{"text": "有效内容", "page": 7, "source": "b.pdf", "type": "image", "image": None}]
    out = C.clean_parsed(parsed)
    assert out[0]["page"] == 7
    assert out[0]["source"] == "b.pdf"
    assert out[0]["type"] == "image"
