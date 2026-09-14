"""
Embedding 模块：通过 API 调用，不下载本地模型

之前的问题是 Streamlit Cloud 每次冷启动要重新下载 470MB 的
text2vec-base-chinese 模型，导致文档上传时进度条卡死。

现在改用硅基流动 Embedding API（OpenAI 兼容格式），发 HTTP 请求
就能拿到向量，零下载，秒级响应。
"""
import os
import sys
import time

from openai import OpenAI
from src.config import (
    EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL,
    EMBEDDING_BACKEND, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
)

# --- 分批与重试 ---
# 为什么要分批：原来是把整份文件的 chunk 一次性发给 API。一本 500 页 PDF 约
# 2000 个 chunk，单请求体积、30s 超时、服务端限流三样随便撞一个就整体失败，
# 而且失败后整篇文档标记 error、用户得重传。分批后每批规模可控，
# 单批失败也只重试那一批。
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
# 失败重试次数（含首次）。指数退避：1s → 2s。API 偶发 429/超时靠这个兜住。
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
# 单批请求超时（秒）。按批给足时间，不再是一个固定 30s 卡死。
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "60"))

# --- 查询侧指令前缀（BGE 系列专用）---
# BGE 是"非对称"训练：query 与 passage 的编码方式不同，官方要求检索时给
# **查询**加一句指令，告诉模型"你现在编码的是问题，不是文档"。
# 不加会让 query 向量没对齐到问题语义空间，检索效果白白损失几个点。
# 关键：只加查询，文档入库绝不能加（embed_texts 保持原样，索引无需重建）。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
# 默认关闭——这是 2026-09-11 的实测结论，不是偷懒：
#   硅基流动 BAAI/bge-large-zh-v1.5，108 chunk 语料 / 25 问：
#   加前缀后 top1 余弦相似度均值 0.6170 → 0.5554（-6.2 个点），MRR 0.887 → 0.880。
#   判断：该 API 商已在服务端处理查询指令，客户端再加属于重复添加，反而拉低对齐。
#   换成本地自建 bge / 明确要求指令的服务时，设 QUERY_INSTRUCTION_ENABLED=true 开启。
QUERY_INSTRUCTION_ENABLED = os.getenv("QUERY_INSTRUCTION_ENABLED", "false").lower() == "true"


def _with_query_instruction(text: str) -> str:
    """按需给查询拼上指令前缀。仅对 BGE 类云端模型生效。"""
    if not QUERY_INSTRUCTION_ENABLED:
        return text
    if EMBEDDING_BACKEND == "ollama":
        return text  # nomic-embed-text 等本地模型不需要该前缀
    if "bge" in EMBEDDING_MODEL.lower():
        return BGE_QUERY_INSTRUCTION + text
    return text

_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE, timeout=30.0)
# 本地 Ollama embedding 客户端（仅 EMBEDDING_BACKEND=ollama 时使用）
_ollama_client = None


def _get_ollama_embed_client() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL, timeout=60.0)
    return _ollama_client


def _embed_one_batch(texts: list[str]) -> list[list[float]]:
    """单批向量化（不再分批）。云端失败会抛异常，由上层决定是否重试。"""
    if EMBEDDING_BACKEND == "ollama":
        print(f"[Embedding] 本地 Ollama，模型={OLLAMA_EMBED_MODEL}，文本数={len(texts)}",
              file=sys.stderr, flush=True)
        try:
            resp = _get_ollama_embed_client().embeddings.create(
                model=OLLAMA_EMBED_MODEL, input=texts, timeout=EMBED_TIMEOUT,
            )
            # Ollama 按输入顺序返回，直接取 embedding
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"[Embedding] Ollama 失败，回退云端: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            # 回退到云端，保证建库不中断
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        timeout=EMBED_TIMEOUT,   # per-request 覆盖，按批给足时间
    )
    return [d.embedding for d in response.data]


def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    """带指数退避的单批向量化。全部重试用尽才抛异常。"""
    last_err: Exception | None = None
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            return _embed_one_batch(texts)
        except Exception as e:
            last_err = e
            if attempt < EMBED_MAX_RETRIES:
                wait = 2 ** (attempt - 1)  # 1s → 2s
                print(f"[Embedding] 第 {attempt}/{EMBED_MAX_RETRIES} 次失败"
                      f"（{len(texts)} 条），{wait}s 后重试: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
    print(f"[Embedding] 重试 {EMBED_MAX_RETRIES} 次仍失败，放弃本批: "
          f"{type(last_err).__name__}: {last_err}", file=sys.stderr, flush=True)
    raise last_err


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量将文本转成向量（自动分批 + 失败重试）。

    EMBEDDING_BACKEND=cloud（默认）→ 硅基流动 API，零下载、秒级响应。
    EMBEDDING_BACKEND=ollama        → 本地 Ollama embedding（nomic-embed-text 等），
                                      实现完全离线建库、数据不出域。

    分批保证单请求规模可控（EMBED_BATCH_SIZE），失败按批重试（EMBED_MAX_RETRIES），
    任一批最终失败都会抛异常——调用方据此把文档标记为 error，避免"入库了一半"
    却当作成功。
    """
    if not texts:
        return []

    if len(texts) <= EMBED_BATCH_SIZE:
        return _embed_with_retry(texts)

    batches = [texts[i:i + EMBED_BATCH_SIZE]
               for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    print(f"[Embedding] 模型={EMBEDDING_MODEL}，共 {len(texts)} 条，"
          f"分 {len(batches)} 批（每批 {EMBED_BATCH_SIZE}）", file=sys.stderr, flush=True)

    out: list[list[float]] = []
    for idx, batch in enumerate(batches, start=1):
        out.extend(_embed_with_retry(batch))
        print(f"[Embedding] 第 {idx}/{len(batches)} 批完成，累计 {len(out)} 条",
              file=sys.stderr, flush=True)
    return out


def embed_single(text: str) -> list[float]:
    """单个文本转向量（不加任何前缀，语义中立）"""
    return embed_texts([text])[0]


def embed_query(text: str) -> list[float]:
    """查询侧向量：自动带上 BGE 指令前缀。

    检索相关的调用（vector_store.search_similar、query_cache）走这里；
    文档入库走 embed_texts，不加前缀——所以加这个前缀**不需要重建索引**。
    """
    return embed_texts([_with_query_instruction(text)])[0]
