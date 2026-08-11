"""确定性测试：BM25 中文分词（jieba 词语级切分，不碰数据库 / 不调 LLM）。

为什么测这个：检索如果按字切，"检索"就搜不到"混合检索"里的"检索"。
中文按词切是召回率的地基。
"""
from src.hybrid_retriever import _tokenize


def test_chinese_word_segmentation():
    """"什么是混合检索" 应切成词，而不是按字切。"""
    toks = _tokenize("什么是混合检索")
    assert "混合" in toks
    assert "检索" in toks


def test_english_terms_kept_as_words():
    """英文专有名词保留为单个 token（"python" 不该被切散）。"""
    toks = _tokenize("混合检索 python api")
    assert "python" in toks
