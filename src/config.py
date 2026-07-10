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

# --- Embedding 模型 ---
# 本地运行，免费，不需要 API
# text2vec-base-chinese: 中文优化，768维，已缓存在本地
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
EMBEDDING_DIM = 768  # 该模型的输出维度

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
