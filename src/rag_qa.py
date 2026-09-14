"""
RAG 问答主链路（确定性管道，**非 Agent**）

链路：查询改写（多轮追问消解指代）→ HybridRetriever 检索 → 交叉编码精排
     → 质量门控（精排分数达标才把上下文交给 LLM）→ 拼上下文 → LLM 生成 → 引用映射。
本模块不含自主决策；Agent（LangGraph 工具循环）实现在 src/agent/dev_agent_langgraph.py，
服务于 HTTP API 路径。二者分工见 README「两条入口，一个大脑」。

主要能力：
  1. rag_query / generate_answer 用 @observe 装饰，自动捕获输入输出、延迟、Token。
  2. 注意：不用 langfuse.openai 客户端——它在部分环境（如 Python 3.14 + 新 openai）
     导入即崩，且会全局 patch openai 客户端干扰 RAGAS 评测。观测统一走 @observe。
  3. **多模态 / 私有化扩展**：
     - 推理后端可切换：cloud（DeepSeek 等云端 API）| ollama（本地私有化、离线、数据不出域）。
     - 命中图片块时：若配置了本地视觉模型（Ollama minicpm-v 等），先由视觉模型"真看图"
       生成识别结果，再交给主 LLM 综合回答并标注引用。
     - 云端模式无视觉模型，自动依赖 OCR 文字（已在 chunk 内容中），**绝不向文本模型发送图片**。

对外 API：
  rag_query(kb_id, query, top_k, backend=None, vision_model=None, history=None)
      -> {answer, sources, contexts, grounded, query, retrieval_query, log_id}
  stream_rag_query(...) -> (答案生成器, sources, contexts, retrieval_query, log_ref)
      · grounded=False（contexts 也为空）表示检索未达阈值，answer 是不依赖知识库的兜底回答
      · log_ref 是可变容器 {"id": ...}：流式下返回时生成器还没跑，答案与日志 id
        都要等 st.write_stream 跑完才能取到，只有容器能穿透（见 src/answer_log.py）
  extract_cited_sources -> 见 `src/citations.py`
      · 迁出是为了让引用逻辑能脱离 langfuse 单独测试。需要它就**直接从 src.citations 导入**
        —— 这里不再留转发（原先那句 re-export 的注释写着「保持老路径可用」，但加它的
        同一次提交 5e05dae 就把唯一的调用方 test_citations 改到了新路径，它从未被人用过）
  backend / vision_model 不传则读 config，保证已部署的云端 Demo 行为不变。

问答现场快照（反馈环 P0，见 src/answer_log.py）：
  每次问答落一条 answer_log；**快照在门控清空 contexts 之前抓**，否则事后无法
  知道当时的最高分是 0.28（该调检索）还是 0.02（该补文档）——这两个数含义相反，
  而它们在返回体里长得一模一样。写入是旁路，失败一律吞掉，绝不影响回答。
"""
import sys

from langfuse.decorators import observe
from openai import OpenAI

from src.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RETRIEVE,
    LLM_BACKEND, OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_VISION_MODEL,
    RAG_HISTORY_TURNS,
)
# 生产检索器：基于 Chroma + BM25 + RRF，按 kb_id 检索（与已部署版本一致）
from src.hybrid_retriever import HybridRetriever
# 查询改写：多轮追问先消解指代再检索（无历史时零成本原样返回）
from src.query_rewrite import rewrite_query
# 检索质量门控：精排分数低于阈值时不注入上下文，如实说"不知道"（防幻觉）
from src.answer_gate import is_grounded, top_score
# 引用来源整理（按文档去重 + 附命中原文，让引用可核实）
from src import citations
# 问答现场快照（反馈环 P0）：旁路写入，纯 stdlib 无重依赖，可顶层 import
from src import answer_log

# 云端 DeepSeek 客户端（观测由 @observe 装饰器完成，不依赖 langfuse.openai）
# 惰性创建：模块 import 不造客户端——没配 key 的环境（CI / 测试）也能 import，
# 只有真正调用 LLM 时才创建并校验凭据。
_client = None


def _get_client() -> OpenAI:
    """取云端 DeepSeek 客户端，首次调用时创建（惰性单例）。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client

# 本地 Ollama 客户端缓存（按 base_url）
_ollama_clients = {}


def _get_ollama_client(base_url: str) -> OpenAI:
    """取本地 Ollama 的 OpenAI 兼容客户端。

    ⚠️ 这里原先写的是 `OpenAIClient(...)`——**这个类在本文件里根本不存在**
    （只 import 了 `OpenAI`）。后果是：只要选 `ollama` 后端，第一次调用就
    NameError，整条"本地私有化 / 离线"路径一用就崩，而云端路径完全正常，
    所以一直没被发现。同类错字在静态检查缺位时能活很久——这也是它现在
    必须留一条注释的原因：改回来之前先看一眼这里。
    """
    if base_url not in _ollama_clients:
        _ollama_clients[base_url] = OpenAI(
            api_key="ollama", base_url=base_url, timeout=120.0
        )
    return _ollama_clients[base_url]


def _log_answer(**kw) -> int | None:
    """写一条问答快照，返回日志 id。

    双重保险：`answer_log.log_answer` 内部已全包 try/except，这里再包一层——
    日志是**旁路**，它的存在不能成为用户拿不到回答的理由。调用点也不必到处写
    try/except（本文件有 5 处写入点）。
    """
    try:
        return answer_log.log_answer(**kw)
    except Exception as e:
        print(f"[AnswerLog] 写入失败（不影响回答）: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return None


SYSTEM_PROMPT = """你是一个基于知识库的问答助手。用户会提供「参考文档」和「问题」。

规则：
1. 只根据参考文档回答，不要编造文档中没有的信息。
2. 回答要简洁、准确，用中文。
3. 如果参考文档中没有相关信息，直接说「知识库中未找到相关信息」。
4. 引用参考文档时，在引用内容后标注来源序号，格式：[1]、[2]……只使用参考文档中标注的序号，不要编造文件名或页码。
5. 如果问题不涉及文档中的具体内容，可以根据常识简短回答。"""

# 检索质量门控未通过时使用的系统提示（见 src/answer_gate.py）。
# 关键区别：**这份提示里没有参考文档**——低分上下文一律不喂给模型，
# 从根上断掉「从一堆不相关内容里硬编一个答案」这条幻觉路径。
NO_CONTEXT_SYSTEM_PROMPT = """你是一个基于知识库的问答助手。

当前情况：用户的问题**没有在知识库中检索到相关内容**（检索最高相关度低于门槛），
你手上没有可用的参考文档。

规则：
1. 首先要如实告知用户：知识库中没有找到相关资料。
2. 严禁编造、猜测知识库中的内容；严禁引用任何文件名、页码或来源。
3. 只有当问题属于通用常识或寒暄（如问候、简单算术）时，才可以简短正常回答，
   并说明这不来自知识库。
4. 回答简洁，用中文。"""


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


def _prepare_generation(query: str, contexts: list[dict],
                        backend: str | None = None,
                        vision_model: str | None = None,
                        base_url: str | None = None,
                        llm_model: str | None = None,
                        history: list[dict] | None = None,
                        grounded: bool = True) -> tuple:
    """组装生成请求：视觉接地 + 上下文 + 最近对话历史 + 选客户端。

    generate_answer（非流式）与 stream_generate_answer（流式）共用，
    返回 (client, model, messages)。

    history: [{role, content}]，最近的对话轮次。注入 system 与当前问题之间，
    让「那第二点呢」「具体怎么配置」这类追问有上下文（此前只渲染不注入）。

    grounded: 检索结果是否足以支撑回答（见 src/answer_gate.py）。False 时
    **一份参考文档都不注入**，改用 NO_CONTEXT_SYSTEM_PROMPT 如实说明没找到——
    低分上下文喂给模型只会诱发幻觉，不如不喂。
    """
    backend = backend or LLM_BACKEND
    vision_model = vision_model if vision_model is not None else OLLAMA_VISION_MODEL
    base_url = base_url or OLLAMA_BASE_URL

    if not grounded:
        # 未命中知识库：不接地、不给参考文档，直接问原问题
        system_prompt = NO_CONTEXT_SYSTEM_PROMPT
        user_message = query
    else:
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
        system_prompt = SYSTEM_PROMPT
        user_message = f"参考文档：\n\n{context_text}\n\n问题：{query}\n\n请基于参考文档回答，并标注引用来源。"

    # --- 选择 LLM 客户端 ---
    if backend == "ollama":
        client = _get_ollama_client(base_url)
        model = llm_model or OLLAMA_LLM_MODEL
    else:
        client = _get_client()
        model = llm_model or LLM_MODEL

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        # 注入最近对话轮次（system + 历史 + 当前问题），保留追问上下文；
        # 只取 user/assistant 纯文本，丢弃 sources 等展示性元数据
        for h in history[-RAG_HISTORY_TURNS:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    return client, model, messages


@observe(name="rag.generate_answer", capture_input=True, capture_output=True)
def generate_answer(query: str, contexts: list[dict],
                    backend: str | None = None,
                    vision_model: str | None = None,
                    base_url: str | None = None,
                    llm_model: str | None = None,
                    history: list[dict] | None = None,
                    grounded: bool = True) -> str:
    """基于检索到的上下文生成答案。支持本地视觉模型接地与最近对话历史。"""
    client, model, messages = _prepare_generation(
        query, contexts, backend=backend, vision_model=vision_model,
        base_url=base_url, llm_model=llm_model, history=history, grounded=grounded,
    )
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=LLM_TEMPERATURE, max_tokens=1024,
    )
    return response.choices[0].message.content


def stream_generate_answer(query: str, contexts: list[dict],
                           backend: str | None = None,
                           vision_model: str | None = None,
                           base_url: str | None = None,
                           llm_model: str | None = None,
                           history: list[dict] | None = None,
                           grounded: bool = True):
    """流式生成答案，逐块 yield（供网页打字机效果）。"""
    client, model, messages = _prepare_generation(
        query, contexts, backend=backend, vision_model=vision_model,
        base_url=base_url, llm_model=llm_model, history=history, grounded=grounded,
    )
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=LLM_TEMPERATURE,
        max_tokens=1024, stream=True,
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


@observe(name="rag.query", capture_input=True, capture_output=True)
def rag_query(kb_id: str, query: str, top_k: int = TOP_K_RETRIEVE,
              backend: str | None = None,
              vision_model: str | None = None,
              history: list[dict] | None = None) -> dict:
    """完整的 RAG 查询流程：检索 + 生成。backend/vision_model 不传则读 config。"""
    # 1. 追问消解：历史 + 当前问题 → 自包含的检索 query（无历史 / 失败则原样返回）
    retrieval_query = rewrite_query(query, history)

    # 2. 检索
    retriever = HybridRetriever(kb_id)
    contexts = retriever.search(retrieval_query, top_k=top_k)

    if not contexts:
        answer = "知识库中没有找到相关内容，请先上传文档。"
        # gate_score 记 0.0 而非 None：None 表示"压根没检索"（缓存命中），
        # 这里确实检索了、只是一条都没搜到（典型是空库），两者不是一回事。
        log_id = _log_answer(
            kb_id=kb_id, question=query, retrieval_query=retrieval_query,
            answer=answer, grounded=False, gate_score=0.0,
            backend=backend or LLM_BACKEND, hits=answer_log.snapshot_hits([]),
        )
        return {
            "answer": answer,
            "sources": [],
            "contexts": [],
            "grounded": False,
            "query": query,
            "retrieval_query": retrieval_query,
            "log_id": log_id,
        }

    # 3. 门控：检索结果分数过低 → 清空上下文，如实说没找到（防幻觉，见 answer_gate）
    # ⚠️ 快照必须在清空之前抓（反馈环 P0 的硬约束）：contexts = [] 之后，门控分数
    # 就再也拿不到了——事后无法区分"0.28 差一点"和"0.02 库里根本没有"，而这两个数
    # 指向完全相反的下一步动作。回归测试见 tests/test_answer_log.py。
    gate_score = top_score(contexts)
    ground_hits = answer_log.snapshot_hits(contexts)
    grounded = is_grounded(contexts)
    if not grounded:
        print(f"[AnswerGate] 检索最高分 {top_score(contexts):.4f} 低于阈值，"
              f"转为无知识库支撑回答", file=sys.stderr, flush=True)
        contexts = []

    # 4. 生成（带最近对话历史，多轮追问有上下文）
    answer = generate_answer(query, contexts, backend=backend, vision_model=vision_model,
                             history=history, grounded=grounded)

    # 5. 整理引用来源：按文档去重 + 附命中原文（用户展开即可核实，见 src/citations.py）
    sources = citations.unique_sources(contexts)

    # 6. 落快照（旁路，失败不影响上面任何一步）
    log_id = _log_answer(
        kb_id=kb_id, question=query, retrieval_query=retrieval_query, answer=answer,
        grounded=grounded, gate_score=gate_score, backend=backend or LLM_BACKEND,
        hits=ground_hits,
    )

    # contexts 一并返回：评估面板复用它做 LLM Judge，避免二次检索。
    # retrieval_query 是实际用于检索的 query（单轮时等于原 query，追问时是消解后的）,
    # 便于前端与排查时看清"检索到底拿什么去搜了"。
    # grounded=False 表示检索未达阈值：answer 是不依赖知识库的兜底回答，contexts 已清空。
    # log_id 是本次问答在 answer_log 里的 id（P1 的 👍/👎 要靠它定位记录）。
    return {"answer": answer, "sources": sources, "contexts": contexts,
            "grounded": grounded, "query": query, "retrieval_query": retrieval_query,
            "log_id": log_id}


def stream_rag_query(kb_id: str, query: str, top_k: int = TOP_K_RETRIEVE,
                     backend: str | None = None,
                     vision_model: str | None = None,
                     history: list[dict] | None = None):
    """流式版 RAG 查询。返回 (答案生成器, sources, contexts, retrieval_query, log_ref)。

    网页用 st.write_stream(gen) 渲染打字机效果；sources 用于展示引用来源。
    retrieval_query 是实际拿去检索的 query——单轮时等于用户原话，追问时是消解指代后的
    自包含问题，前端可据此把"检索用了什么"展示出来。
    rag_query（非流式）保留给评估面板使用。history 为最近对话轮次（多轮追问）。

    log_ref：可变容器 `{"id": ...}`，装着本次问答在 answer_log 里的 id。
    为什么用容器而不是直接返回 id：返回时生成器还没跑，答案尚不存在、日志也没写，
    id 无从谈起。前端 `st.write_stream(gen)` 跑完后才能从容器里取到
    （P1 的 👍/👎 按钮需要它做唯一 key）。缓存命中分支属于例外——当时就有答案，
    所以返回时 id 已经填好。
    """
    from src.query_cache import get_cached_answer, cache_answer  # 模块顶层 import 无副作用（embedding 惰性）

    log_ref: dict = {"id": None, "gate_score": None}

    # --- 语义缓存：无历史的独立提问先查缓存，命中直接秒回（不检索、不调模型）---
    # 命中时 sources 用缓存的引用来源；前端 display_sources = cited or sources 无缝兼容。
    if not history:
        cached = get_cached_answer(kb_id, query)
        if cached is not None:
            def gen_cached():
                yield cached["answer"]

            # 缓存命中也要记日志，而且这条信号价值特别高：它意味着"错的答案被缓存了，
            # 会持续错下去"。注意两处语义——
            #   · gate_score 记 None 而不是 0：这次**根本没检索**，0 会被误读成
            #     "检索了但一分没得"（见 answer_log 的语义边界说明）。
            #   · grounded 只能反推：缓存里带 sources 的回答，说明当时过了门控
            #     （未过门控时 contexts 被清空、sources 必为空，二者等价）。
            log_ref["id"] = _log_answer(
                kb_id=kb_id, question=query, answer=cached["answer"],
                grounded=bool(cached["sources"]), gate_score=None, hit_cache=True,
                backend=backend or LLM_BACKEND, hits=answer_log.snapshot_hits([]),
            )
            return gen_cached(), cached["sources"], [], query, log_ref

    # 追问消解：历史 + 当前问题 → 自包含的检索 query（无历史 / 失败则原样返回）。
    # 注意只作用于检索——生成阶段仍用用户原话，历史另由 _prepare_generation 注入。
    retrieval_query = rewrite_query(query, history)

    retriever = HybridRetriever(kb_id)
    contexts = retriever.search(retrieval_query, top_k=top_k)

    if not contexts:
        fallback = "知识库中没有找到相关内容，请先上传文档。"
        log_ref["gate_score"] = 0.0
        log_ref["id"] = _log_answer(
            kb_id=kb_id, question=query, retrieval_query=retrieval_query,
            answer=fallback, grounded=False, gate_score=0.0,
            backend=backend or LLM_BACKEND, hits=answer_log.snapshot_hits([]),
        )
        return iter([fallback]), [], [], retrieval_query, log_ref

    # 门控：检索结果分数过低 → 清空上下文。既不喂给模型（防幻觉），也不作为
    # 引用来源展示（没有依据就没有引用）。前端据 contexts 为空显示"未命中"提示。
    # ⚠️ 快照必须在清空之前抓（反馈环 P0 的硬约束，同 rag_query）。
    gate_score = top_score(contexts)
    ground_hits = answer_log.snapshot_hits(contexts)
    grounded = is_grounded(contexts)
    log_ref["gate_score"] = gate_score
    if not grounded:
        print(f"[AnswerGate] 检索最高分 {top_score(contexts):.4f} 低于阈值，"
              f"转为无知识库支撑回答", file=sys.stderr, flush=True)
        contexts = []

    # 整理引用来源（与 rag_query 同一套逻辑，已收口到 src/citations.py）
    sources = citations.unique_sources(contexts)

    def gen():
        full_answer = ""
        for chunk in stream_generate_answer(query, contexts, backend=backend, vision_model=vision_model,
                                            history=history, grounded=grounded):
            yield chunk
            full_answer += chunk
        # 无历史才缓存：带历史的追问是个性化的，命中率低且易错配
        if not history and full_answer:
            try:
                cache_answer(kb_id, query, full_answer, sources)
            except Exception:
                pass  # 缓存失败不影响回答
        # 快照放在生成之后一次写入（与 cache_answer 同处）。**不做两阶段写**
        # （先 INSERT pending 再 UPDATE）的原因：Streamlit 流式生成在用户关页面时
        # 抛 GeneratorExit，服务端不可靠捕获——两阶段写会留下永远 pending 的僵尸行，
        # 反而制造"看起来像中断"的假数据。详见 notes/feedback-loop-plan.md §4.6。
        if full_answer:
            log_ref["id"] = _log_answer(
                kb_id=kb_id, question=query, retrieval_query=retrieval_query,
                answer=full_answer, grounded=grounded, gate_score=gate_score,
                backend=backend or LLM_BACKEND, hits=ground_hits,
            )

    return gen(), sources, contexts, retrieval_query, log_ref

