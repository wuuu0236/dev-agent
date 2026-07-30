"""
RAG 问答 Agent（Langfuse 版 · v4 API）

主要能力：
  1. LLM 客户端换成 langfuse.openai.OpenAI —— Token / 延迟自动上报（云端模式）。
  2. rag_query / generate_answer 用 @observe 装饰，自动捕获输入输出。
  3. **多模态 / 私有化扩展**：
     - 推理后端可切换：cloud（DeepSeek 等云端 API）| ollama（本地私有化、离线、数据不出域）。
     - 命中图片块时：若配置了本地视觉模型（Ollama minicpm-v 等），先由视觉模型"真看图"
       生成识别结果，再交给主 LLM 综合回答并标注引用。
     - 云端模式无视觉模型，自动依赖 OCR 文字（已在 chunk 内容中），**绝不向文本模型发送图片**。

对外 API：
  rag_query(kb_id, query, top_k, backend=None, vision_model=None) -> {answer, sources, query}
  backend / vision_model 不传则读 config，保证已部署的云端 Demo 行为不变。
"""
import sys

from langfuse import observe
from langfuse.openai import OpenAI as LangfuseOpenAI
from openai import OpenAI as OpenAIClient

from src.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RETRIEVE,
    LLM_BACKEND, OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_VISION_MODEL,
)
# 生产检索器：基于 Chroma + BM25 + RRF，按 kb_id 检索（与已部署版本一致）
from src.hybrid_retriever import HybridRetriever

# 云端客户端（Langfuse 自动上报）；构造不依赖 langfuse key，缺配置只是不上报
_client = LangfuseOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# 本地 Ollama 客户端缓存（按 base_url）
_ollama_clients = {}


def _get_ollama_client(base_url: str) -> OpenAIClient:
    if base_url not in _ollama_clients:
        _ollama_clients[base_url] = OpenAIClient(
            api_key="ollama", base_url=base_url, timeout=120.0
        )
    return _ollama_clients[base_url]


SYSTEM_PROMPT = """你是一个基于知识库的问答助手。用户会提供「参考文档」和「问题」。

规则：
1. 只根据参考文档回答，不要编造文档中没有的信息。
2. 回答要简洁、准确，用中文。
3. 如果参考文档中没有相关信息，直接说「知识库中未找到相关信息」。
4. 回答末尾标注引用来源，格式：[来源: 文件名, 第X页]
5. 如果问题不涉及文档中的具体内容，可以根据常识简短回答。"""


def _vision_ground(query: str, image_paths: list[str], vision_model: str, base_url: str) -> str:
    """调用本地视觉模型识别图片，返回文字描述/答案（用于多模态接地）。"""
    try:
        client = _get_ollama_client(base_url)
        import base64
        user_content: list[dict] = [{
            "type": "text",
            "text": (
                f"请根据这张图片回答用户的问题。用户问题：{query}\n"
                "若图片是文档/表格/截图，请提取关键字段与要点；若看不清请如实说明，不要编造。"
            ),
        }]
        for p in image_paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        resp = client.chat.completions.create(
            model=vision_model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[Vision] 视觉模型调用失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return ""


@observe(name="rag.generate_answer", capture_input=True, capture_output=True)
def generate_answer(query: str, contexts: list[dict],
                    backend: str | None = None,
                    vision_model: str | None = None,
                    base_url: str | None = None,
                    llm_model: str | None = None) -> str:
    """基于检索到的上下文生成答案。支持本地视觉模型接地。"""
    backend = backend or LLM_BACKEND
    vision_model = vision_model if vision_model is not None else OLLAMA_VISION_MODEL
    base_url = base_url or OLLAMA_BASE_URL

    # --- 视觉模型接地（仅本地 Ollama + 配置了视觉模型）---
    vision_text = ""
    if backend == "ollama" and vision_model:
        img_chunks = [c for c in contexts if c.get("type") == "image" and c.get("image")]
        seen, img_paths = set(), []
        for c in img_chunks:
            if c["image"] not in seen:
                seen.add(c["image"])
                img_paths.append(c["image"])
        if img_paths:
            vision_text = _vision_ground(query, img_paths, vision_model, base_url)

    # --- 组装文本上下文 ---
    text_parts = []
    for i, c in enumerate(contexts):
        src = f"[文档 {i + 1}] 来源: {c['source']}"
        if c.get("page"):
            src += f", 第{c['page']}页"
        text_parts.append(f"{src}\n{c['content']}")
    if vision_text:
        text_parts.append(f"[文档 {len(contexts) + 1}] 来源: 本地视觉模型({vision_model}) 识别结果\n{vision_text}")

    context_text = "\n\n---\n\n".join(text_parts)
    user_message = f"参考文档：\n\n{context_text}\n\n问题：{query}\n\n请基于参考文档回答，并标注引用来源。"

    # --- 选择 LLM 客户端 ---
    if backend == "ollama":
        client = _get_ollama_client(base_url)
        model = llm_model or OLLAMA_LLM_MODEL
    else:
        client = _client
        model = llm_model or LLM_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=1024,
    )

    return response.choices[0].message.content


@observe(name="rag.query", capture_input=True, capture_output=True)
def rag_query(kb_id: str, query: str, top_k: int = TOP_K_RETRIEVE,
              backend: str | None = None,
              vision_model: str | None = None) -> dict:
    """完整的 RAG 查询流程：检索 + 生成。backend/vision_model 不传则读 config。"""
    # 1. 检索
    retriever = HybridRetriever(kb_id)
    contexts = retriever.search(query, top_k=top_k)

    if not contexts:
        return {
            "answer": "知识库中没有找到相关内容，请先上传文档。",
            "sources": [],
            "query": query,
        }

    # 2. 生成
    answer = generate_answer(query, contexts, backend=backend, vision_model=vision_model)

    # 3. 去重引用来源（图片块也带上类型标记，便于前端展示）
    seen = set()
    unique_sources = []
    for c in contexts:
        key = c["source"]
        if key not in seen:
            seen.add(key)
            unique_sources.append({
                "source": c["source"],
                "page": c.get("page", 0),
                "type": c.get("type", "text"),
            })

    return {"answer": answer, "sources": unique_sources, "query": query}
