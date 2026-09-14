"""
混合检索：BM25（关键词） + 向量（语义） + RRF 融合

为什么需要混合检索：
  - 向量检索擅长语义匹配，但搜专有名词（Dockerfile、requirements.txt）效果差
  - BM25 擅长关键词精确匹配，但搜不到近义词
  - 两者互补，结合后 Recall 和 Precision 都比单独好

RRF（倒数排名融合）：
  对每个文档，计算: score = weight1 / (k + rank1) + weight2 / (k + rank2)
  两个排名独立计算，然后加权融合成最终排名。
  RRF 的好处是不需要知道原始分数的分布（BM25 和向量分数的范围完全不同）。
"""
import jieba
from rank_bm25 import BM25Okapi
# 注意：不在这里 import vector_store（chromadb 是重型原生依赖）。
# 只有真正查库时才 import（惰性）——让分词/检索逻辑本身不依赖 chromadb，
# 也保证 CI 冒烟测试无需安装 chromadb。
from src.config import TOP_K_RETRIEVE, BM25_WEIGHT, VECTOR_WEIGHT, RETRIEVE_CANDIDATES

# 中文高频虚词（停用词）。只收「几乎每句话都有、本身不指示任何内容」的词：
# 助词 / 语气词 / 代词 / 指代词 / 轻动词。
#
# 刻意**不收**疑问词（什么、怎么、如何、为什么、哪些）：它们标记的是"问法"
# 而不是内容，留着不碍事，收进来反而容易误伤。
# 原则：宁可少收、不可多收——这份表唯一的作用是把噪声挤出查询词表，
# 挤多了就是自己削自己的召回。
#
# 为什么不引第三方停用词表：完整表上千词，而 BM25 的 IDF 本身已经能压制
# 语料内的高频词（词出现在半数以上文档时 IDF 会被夹到极小值）。这里只需要
# 处理「短查询里虚词权重占比过高」这一个具体问题。
_STOPWORDS = frozenset({
    # 助词 / 连词 / 介词
    "的", "了", "是", "在", "和", "与", "或", "也", "都", "就", "而", "及",
    "着", "过", "得", "地", "并", "且", "但", "为", "从", "对", "由",
    # 语气词
    "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "啦", "嘛",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "它们", "自己",
    # 指代词 / 泛指量词
    "这", "那", "这个", "那个", "这些", "那些", "一个", "一些", "一下",
    # 轻动词（本身不含检索信息）
    "有", "没有", "会", "能", "可以", "要", "把", "被", "给", "让",
    # 英文功能词
    "a", "an", "the", "of", "to", "is", "are", "was", "were",
    "and", "or", "not", "in", "on", "at", "for", "with", "by", "as",
})


def _tokenize(text: str) -> list[str]:
    """中文分词：用 jieba 做词语级切分，并滤掉对检索无意义的 token。

    为什么不用按字切分：
      "什么是混合检索" → 按字切: ["什","么","是","混","合","检","索"]
      → 搜"检索"只能匹配到"检"和"索"两个字，毫无意义

      "什么是混合检索" → jieba: ["什么","是","混合","检索"]
      → 搜"检索"能精确匹配到"检索"这个词，BM25 才有真正的关键词能力

    为什么还要过滤（两层噪声）：
      1. jieba 会把标点和空白也切出来（"，" "？" "\n"）。这种 token 在库里
         几乎随处可见，是纯噪声。
      2. "的 / 了 / 是 / 都 / 可以" 这类高频虚词，BM25 的 IDF 本来就会把它们
         压到接近 0。但短查询里实词就那么两三个，虚词仍会分走一部分打分权重；
         滤掉后「专有名词/数字/术语」占比更高，词面匹配更锐。
    说明：统一转小写，让 "Dockerfile" / "dockerfile" 算同一个词——
    BM25 是词面匹配，大小写不该被当成不同词。
    """
    tokens = []
    for t in jieba.cut(text):
        t = t.strip().lower()
        if not t or t in _STOPWORDS:
            continue
        # 必须至少含一个字母或数字，否则丢掉。
        # 注意这里**不能**写成 `ch.isalnum() or ch == "_"`：jieba 会把
        # "hybrid_retriever" 切成 ["hybrid", "_", "retriever"]，那个孤立的
        # 下划线就会被放行成一个 token。而分词是**文档侧和查询侧同一套规则**，
        # 所以拆成两半不影响匹配（文档里的 hybrid_retriever 也拆成同样两半）。
        if not any(ch.isalnum() for ch in t):
            continue
        tokens.append(t)
    return tokens


# --- BM25 索引缓存（模块级，跨实例共享）---
# 之前每个 HybridRetriever 实例、每次 search 都全量重建 BM25 索引，
# 知识库一大就 O(N)/query。现在按 kb_id 缓存，缓存键 = 当前 chunk 总数：
#   · 向量库有增删时 count 变化 → 自动失效重建
#   · count 从持久层（Chroma）读取，Streamlit / FastAPI 多进程共享 data 目录也能正确失效
# 唯一边界：删 1 条 + 增 1 条导致 count 不变时可能用到旧索引，下次增删即自愈。
_bm25_cache: dict[str, tuple[int, BM25Okapi | None, list[dict]]] = {}


def _get_bm25(kb_id: str) -> tuple[BM25Okapi | None, list[dict]]:
    """按 kb_id 取（BM25 索引, 全量 chunk），有增删时自动重建。

    BM25_WEIGHT 为 0 时直接短路（当前默认值见 config.py）：建索引要从 Chroma
    拉全量 chunk 再逐条 jieba 分词，成本不低，而融合时贡献是 0/(60+rank)=0
    ——纯属白算。把 BM25_WEIGHT 调成 1 即可恢复混合检索，这里会自动重新启用。
    """
    if BM25_WEIGHT == 0:
        return None, []
    from src.vector_store import collection_count, get_all_chunks  # 惰性：只有查库才拖 chromadb
    current_count = collection_count(kb_id)
    cached = _bm25_cache.get(kb_id)
    if cached and cached[0] == current_count:
        return cached[1], cached[2]

    chunks = get_all_chunks(kb_id)
    if not chunks:
        _bm25_cache[kb_id] = (current_count, None, [])
        return None, []

    bm25 = BM25Okapi([_tokenize(c["content"]) for c in chunks])
    _bm25_cache[kb_id] = (current_count, bm25, chunks)
    return bm25, chunks


class HybridRetriever:
    """
    混合检索器，每个知识库一个实例。

    使用方式：
      retriever = HybridRetriever(kb_id)
      results = retriever.search("查询内容", top_k=5)
    """

    def __init__(self, kb_id: str):
        self.kb_id = kb_id

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE,
               rerank: bool | None = None) -> list[dict]:
        """
        检索主流程（粗排 → 精排）：
        1. 向量检索（双塔，粗排）——先捞候选池 candidate_k 条，不是 top_k 条
        2. BM25 检索（关键词匹配）
        3. RRF 融合两路排名
        4. Rerank 精排（交叉编码）——从候选池里挑出最终 top_k

        为什么要先捞 candidate_k：精排只能从粗排给它的东西里挑。
        粗排只给 5 条，精排再准也只能在这 5 条里排序；给 50 条，精排才有
        足够的发挥空间。这一步的代价是向量检索多算了 10 倍，但向量检索本身
        很便宜（毫秒级），换来精排可用的候选池，是划算的。

        rerank: 显式覆盖精排开关（评测做 A/B 用），None 则读 config.RERANK_ENABLED
        """
        # 候选池：0 表示自动 = top_k * 10
        candidate_k = RETRIEVE_CANDIDATES or top_k * 10

        # --- 向量检索（粗排）---
        from src.vector_store import search_similar  # 惰性：只有真查库才拖 chromadb
        vector_results = search_similar(self.kb_id, query, top_k=candidate_k)
        # 给每个结果标上向量检索的排名（1 = 最相关）
        for rank, r in enumerate(vector_results, start=1):
            r["vector_rank"] = rank

        # --- BM25 检索（索引带缓存，向量库有增删时自动重建）---
        bm25, chunks = _get_bm25(self.kb_id)
        bm25_mapped = []
        if bm25 is not None:
            scores = bm25.get_scores(_tokenize(query))
            # 只保留与查询有词面重叠的块（score > 0）。
            # 为什么不能直接取 top candidate_k：BM25 对"一个词都没对上"的块也返回
            # 0 分，按分数排序时这些 0 分块会顶上来凑满候选池。它们没有带来任何
            # 新信息，却会让精排多送一批文档——精排是按文档条数计费的，
            # 这是纯浪费。宁可池子小一点、每一条都言之有物。
            # 反过来也正合语义：查询与全库毫无词面重叠时，BM25 本来就不该有发言权，
            # 这一路交给向量去管。
            indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
            indexed.sort(key=lambda x: x[1], reverse=True)
            for rank, (idx, score) in enumerate(indexed[:candidate_k], start=1):
                chunk = chunks[idx]
                bm25_mapped.append({
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "type": chunk.get("type", "text"),
                    "image": chunk.get("image"),
                    "bm25_rank": rank,
                    "bm25_score": float(score)
                })

        # --- RRF 融合 ---
        # 用 dict 去重（同一内容可能两路都搜到），key = source + content[:50]
        merged = {}
        k = 60  # RRF 公式中的平滑常数

        for r in vector_results:
            key = f"{r['source']}_{r['content'][:50]}"
            merged[key] = {
                "content": r["content"],
                "source": r["source"],
                "page": r.get("page", 0),
                "type": r.get("type", "text"),
                "image": r.get("image"),
                "rrf_score": VECTOR_WEIGHT / (k + r["vector_rank"])
            }

        for r in bm25_mapped:
            key = f"{r['source']}_{r['content'][:50]}"
            bm25_contrib = BM25_WEIGHT / (k + r["bm25_rank"])
            if key in merged:
                merged[key]["rrf_score"] += bm25_contrib
            else:
                merged[key] = {
                    "content": r["content"],
                    "source": r["source"],
                    "page": r.get("page", 0),
                    "type": r.get("type", "text"),
                    "image": r.get("image"),
                    "rrf_score": bm25_contrib
                }

        # 按 RRF 分数从高到低排序，并**截断回候选池大小**（粗排完成）
        #
        # 为什么必须截断：RRF 只是把各路排名加起来，它本身不设上限——
        # 再加一路召回，融合结果就再长一截。而精排是**按文档条数计费**的
        # （每条候选都要单独过一遍交叉编码模型），不截断就等于"多开一路召回
        # 顺手把精排账单也加上去"。截断之后，精排的开销只由候选池大小决定，
        # 与开了几路召回无关——这才是「粗排 → 精排」这个架构里"池子有固定预算"
        # 的正确含义。
        #
        # 注意这不是简单的"砍掉尾巴"：截断发生在**融合之后**，所以两路都排得靠前
        # 的块会被顶上来，把只有一路支持的低分块挤出去。这正是 RRF 想要的效果——
        # 用有限的预算装下"多路共识"的候选。
        sorted_results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:candidate_k]

        # --- Rerank 精排（交叉编码）---
        # 失败会自动降级为「粗排顺序的前 top_k」，不会打断问答链路
        from src.reranker import rerank as _rerank
        return _rerank(query, sorted_results, top_k, enabled=rerank)
# ================================================================
# 模块级函数：供 Agent 和 MCP 调用（统一入口）
# ================================================================

_DEFAULT_KB = "agent"


def _ensure_kb() -> str:
    """获取最近的知识库 ID，没有就创建默认的"""
    from src.database import list_kbs, create_kb, init_db
    init_db()
    kbs = list_kbs()
    if kbs:
        return kbs[0]["id"]
    from src.vector_store import create_collection
    kb = create_kb(_DEFAULT_KB, "Agent 默认知识库")
    create_collection(kb["id"])
    return kb["id"]


def search_knowledge(query: str) -> str:
    """搜索知识库，返回纯文本（供 Agent/MCP 调用）"""
    kb_id = _ensure_kb()
    retriever = HybridRetriever(kb_id)
    results = retriever.search(query, top_k=5)
    if not results:
        return "知识库中未找到相关内容，请先上传文档。"
    lines = []
    for i, r in enumerate(results, 1):
        source = f"[{i}] 来源: {r['source']}"
        if r.get("page"):
            source += f", 第 {r['page']} 页"
        lines.append(source)
        lines.append(r["content"][:300])
        lines.append("")
    return "\n".join(lines)


def add_knowledge(texts: list[str]) -> str:
    """把文本添加到知识库"""
    kb_id = _ensure_kb()
    from src.chunker import chunk_parsed
    from src.cleaner import clean_parsed
    docs = [{"text": t, "page": None, "source": "agent_"} for t in texts]
    docs = clean_parsed(docs)
    chunks = chunk_parsed(docs)
    if not chunks:
        return "没有可添加的内容"
    from src.vector_store import add_chunks
    add_chunks(kb_id, chunks)
    return f"已添加 {len(chunks)} 条内容到知识库"


def load_file_to_knowledge(filepath: str) -> str:
    """把文件加载到知识库（含安全检查）"""
    from src.tools.safety import check_path_safety
    safe, reason = check_path_safety(filepath)
    if not safe:
        return f"安全拦截: {reason}"
    from src.parser import parse_file
    from src.chunker import chunk_parsed
    from src.vector_store import add_chunks
    kb_id = _ensure_kb()
    try:
        parsed = parse_file(filepath)   # 内部已含清洗
        if not parsed:
            return f"文件 '{filepath}' 没有可解析的内容"
        chunks = chunk_parsed(parsed)
        if not chunks:
            return f"文件 '{filepath}' 没有足够的内容"
        add_chunks(kb_id, chunks)
        return f"已加载文件 '{filepath}'，共 {len(chunks)} 条内容到知识库"
    except Exception as e:
        return f"加载失败: {type(e).__name__}: {e}"