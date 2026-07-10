"""
RAG 问答 Agent

流程：
  用户问题 → 混合检索（HybridRetriever）
           → 拼接 prompt（system + context + question）
           → 调用 DeepSeek 生成答案（带引用来源）
           → 返回 {answer, sources}

使用 DeepSeek API（兼容 OpenAI 格式），不需要 LangChain 封装。
"""
from openai import OpenAI
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RETRIEVE
from src.hybrid_retriever import HybridRetriever

# DeepSeek 客户端
_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

SYSTEM_PROMPT = """你是一个基于知识库的问答助手。用户会提供「参考文档」和「问题」。

规则：
1. 只根据参考文档回答，不要编造文档中没有的信息。
2. 回答要简洁、准确，用中文。
3. 如果参考文档中没有相关信息，直接说「知识库中未找到相关信息」。
4. 回答末尾标注引用来源，格式：[来源: 文件名, 第X页]
5. 如果问题不涉及文档中的具体内容，可以根据常识简短回答。"""


def generate_answer(query: str, contexts: list[dict]) -> str:
    """
    基于检索到的上下文生成答案。

    contexts: [{"content": "...", "source": "...", "page": 1}, ...]
    """
    # 拼接上下文
    context_text = "\n\n---\n\n".join(
        f"[文档 {i+1}] 来源: {c['source']}"
        + (f", 第{c['page']}页" if c.get("page") else "")
        + f"\n{c['content']}"
        for i, c in enumerate(contexts)
    )

    user_message = f"参考文档：\n\n{context_text}\n\n问题：{query}\n\n请基于参考文档回答，并标注引用来源。"

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=1024
    )

    return response.choices[0].message.content


def rag_query(kb_id: str, query: str, top_k: int = TOP_K_RETRIEVE) -> dict:
    """
    完整的 RAG 查询流程：检索 + 生成

    返回：{answer, sources, query}
    """
    # 1. 检索
    retriever = HybridRetriever(kb_id)
    contexts = retriever.search(query, top_k=top_k)

    if not contexts:
        return {
            "answer": "知识库中没有找到相关内容，请先上传文档。",
            "sources": [],
            "query": query
        }

    # 2. 生成
    answer = generate_answer(query, contexts)

    # 3. 去重引用来源
    seen = set()
    unique_sources = []
    for c in contexts:
        key = c["source"]
        if key not in seen:
            seen.add(key)
            unique_sources.append({"source": c["source"], "page": c.get("page", 0)})

    return {
        "answer": answer,
        "sources": unique_sources,
        "query": query
    }
