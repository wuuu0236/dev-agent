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
BM25_WEIGHT = 0     # RRF 融合中 BM25 的权重
VECTOR_WEIGHT = 1   # RRF 融合中向量检索的权重

# --- 安全配置 ---
MAX_FILE_SIZE_MB = 20  # 上传文件大小限制
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".csv"]
