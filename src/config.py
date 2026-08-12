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

# --- 检索配置 ---
TOP_K_RETRIEVE = 5     # 检索返回的文档数
RAG_HISTORY_TURNS = 6  # 注入的最近对话条数（按消息条数切，非严格"轮"；Web 与 API 两条路径一致）

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
