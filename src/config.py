"""
全局配置：路径、模型、超参数
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- 项目路径 ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "datalens.db"
CHROMA_DIR = DATA_DIR / "chroma"
UPLOAD_DIR = DATA_DIR / "uploads"

# 确保目录存在
for d in [DATA_DIR, CHROMA_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- LLM 配置 ---
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.3  # RAG 场景用低温度，减少幻觉

# --- 推理后端 ---
# "cloud"  → 走 DeepSeek 等云端 API（默认，已部署 Demo 行为不变）
# "ollama" → 走本地 Ollama，实现私有化 / 离线 / 数据不出域
LLM_BACKEND = os.getenv("LLM_BACKEND", "cloud")

# --- Ollama 私有化后端（仅 LLM_BACKEND=ollama 时生效）---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
# 视觉模型：填了才在命中图片块时调用本地视觉模型"真看图"；留空则图片只用 OCR 文字
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "")

# --- Embedding 后端 ---
# "cloud"  → 硅基流动 API（默认，零下载、秒级响应）
# "ollama" → 本地 Ollama embedding（如 nomic-embed-text），实现完全离线索引
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "cloud")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- 多模态 / OCR ---
# 是否对图片、扫描版 PDF 做本地 OCR（RapidOCR，纯 CPU、数据不出域）
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
# 图片类型扩展名（解析层与上传层共用）
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

# --- Embedding API ---
# 使用硅基流动 Embedding API，不需要下载本地模型
# Streamlit Cloud 上不会卡进度条
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = 1024  # BGE-large-zh 输出 1024 维

# --- Chunk 配置 ---
CHUNK_SIZE = 500       # 每个 chunk 的字符数
CHUNK_OVERLAP = 50     # 相邻 chunk 重叠的字符数
# 标题树分块（对标 MaxKB）：按 Markdown 标题切段，段内再滑窗；小于该长度的段并入相邻段
# 默认关闭——2026-09-11 实测（同源 14 篇 md，新库 16684ef4 vs 旧库 83cd2d0c）：
#   Recall@5 0.960→0.940 / Hit@1 0.840→0.800 / MRR 0.887→0.873，四项全降或持平。
# 原因：md 语料的标题文字本就在正文里，路径前缀是冗余；且按 500 字打包后粒度与滑窗无异。
# 该策略的收益场景是 PDF/Word（原文无结构标题），拿到真实 PDF 样本后应重开并复测。
CHUNK_TREE_ENABLED = os.getenv("CHUNK_TREE_ENABLED", "false").lower() == "true"
CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "120"))  # 过短段落合并阈值
CHUNK_PATH_SEP = os.getenv("CHUNK_PATH_SEP", " > ")       # 标题路径分隔符
CHUNK_PATH_MAXLEN = int(os.getenv("CHUNK_PATH_MAXLEN", "120"))  # 路径前缀最长字符数

# --- 检索配置 ---
TOP_K_RETRIEVE = 5     # 检索返回的文档数
RAG_HISTORY_TURNS = 6  # 注入的最近对话条数（按消息条数切，非严格"轮"；Web 与 API 两条路径一致）

# 引用来源展示的原文片段长度（字符）。
# 为什么要有：引用只写"文件名 + 页码"是没法核实的——用户看不到原文，也就无从判断
# 模型是在照实回答还是在拿别的内容硬编。带上命中原文，引用才真的可验证。
SOURCE_SNIPPET_CHARS = int(os.getenv("SOURCE_SNIPPET_CHARS", "300"))

# 候选池大小：粗排阶段先捞多少条给精排用。
# 这是「粗排 → 精排」架构的前提——召回只取 top_k 的话，精排再准也只能
# 从这几条里挑；必须先多捞，精排才有发挥空间。
# 设为 0 表示自动 = top_k * 10（top_k=5 时捞 50 条）。
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "0"))

# --- 查询改写（追问消解）---
# 为什么需要：多轮追问「那第二点呢」直接拿去检索是捞不到东西的——指代没被消解。
# 历史本来就在手（pages/3 传了 history），但此前只接进生成阶段，检索那一行吃的
# 还是原始字符串。本开关让检索前多一层改写：历史 + 当前问题 → 一句自包含的查询。
# 只在有历史时触发（单轮独立提问直接返回原 query，零成本），失败/空/超长一律
# 退回原 query，与 reranker 同一套降级策略，不阻断问答。
# 代价：每次追问多一次 LLM 调用（约 0.3–1s）。要跑对照设 QUERY_REWRITE_ENABLED=false。
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"
QUERY_REWRITE_MAX_CHARS = int(os.getenv("QUERY_REWRITE_MAX_CHARS", "200"))  # 超长视为"开始答题"而非改写

# --- Rerank 精排（交叉编码）---
# 为什么需要：向量检索是「双塔」——query 和 doc 分开编码成向量再比余弦，
# 快但两者从未见过面，丢失词级交互；reranker 是「交叉编码」——把 query 和
# doc 拼在一起过一遍模型，直接输出相关性分数，准但慢，只能对小候选集用。
# 所以架构是：粗排（双塔捞候选池）→ 精排（交叉编码挑最终 top_k）。
# 另一个关键差别：reranker 的分数有绝对意义（实测相关 0.86 / 不相关 0.00002），
# 而 RRF 分数 1/(60+rank) 无绝对意义——所以「相似度阈值拒答」应该建在
# rerank 分数上，不是建在 RRF 分数上。
# 默认开启（2026-09-14 改为 true）——依据 2026-09-13 的 A/B 实测：
#   同库同脚本（kb 83cd2d0c，27 条测试集），只切这一个开关：
#   Recall@5 0.960→1.000 / Hit@5 0.960→1.000 / Hit@1 0.840→0.920 / MRR 0.887→0.960
#   「未完全命中」列表清空，老大难 q05（答案在库里但排不到第一）被修复。
# 代价：每次检索多一次 API 调用（延迟 + 费用）。要跑无精排的对照，
# 设 RERANK_ENABLED=false，或评测时用 scripts/eval_retrieval.py --no-rerank。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", EMBEDDING_API_KEY)
RERANK_API_BASE = os.getenv("RERANK_API_BASE", EMBEDDING_API_BASE)
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "60"))
RERANK_MAX_DOC_CHARS = int(os.getenv("RERANK_MAX_DOC_CHARS", "1500"))  # 单条送审文本上限，防超长

# --- 检索质量门控（阈值拒答）---
# 为什么需要：不管检索到什么，top_k 都会被塞进 prompt。库里没有答案时，模型面对
# 一堆不相关的「参考文档」仍会努力编一个——这是幻觉最直接的来源。
# 为什么建在精排分数上：精排（交叉编码）分数有绝对意义（相关 0.6+ / 不相关 <0.01），
# RRF 分数是 1/(60+rank) 只有相对顺序、没有阈值可言。上面 rerank 那节早就论证过，
# 这里把它落地。
# 阈值依据（kb 83cd2d0c，2026-09-14 实测精排 top1）：
#   可答：0.620（「混合检索是怎么做的」）；历史正样本最低 0.578
#   不可答：≤0.125（「你好」0.125 /「你能做什么」0.122 /「1+1 等于几」0.091 /
#          「帮我写首诗」0.010 /「今天天气」0.001）；已知负样本最高 0.0005
# 取 0.3：不可答的全部拦下，可答的留约 2 倍余量（宁漏拒、不误拒）。
# 注意阈值的语义是「检索结果能否支撑回答」，不是「问题该不该问」；
# 设为 0 或负数即关闭门控（行为与改动前一致）。
# 未命中时的处理：不把低分上下文喂给 LLM，改由 NO_CONTEXT_SYSTEM_PROMPT 如实说明。
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.3"))

# --- 语义缓存（RAG 问答路径）---
# 原理：问题转向量，与缓存问题算余弦相似度，超过阈值命中则秒回、不调模型。
# 失效：知识库文档变化时清空该库缓存（见 vector_store 的 clear_kb_cache 钩子）。
QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
QUERY_CACHE_THRESHOLD = float(os.getenv("QUERY_CACHE_THRESHOLD", "0.90"))  # 语义命中阈值
QUERY_CACHE_MAX_PER_KB = 200  # 每知识库缓存上限（条），超限删最旧
BM25_WEIGHT = 0     # RRF 融合中 BM25 的权重（当前纯向量，经 A/B 测试该场景下纯向量优于混合检索）
VECTOR_WEIGHT = 1   # RRF 融合中向量检索的权重（调回 BM25_WEIGHT=1 即可恢复混合检索）

# --- 安全配置 ---
MAX_FILE_SIZE_MB = 20  # 上传文件大小限制
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".csv",
                      ".png", ".jpg", ".jpeg", ".bmp", ".webp"]  # 末尾为图片类型
