from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PDF_PATH = RAW_DATA_DIR / "apple_10k.pdf"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "apple_10k"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

RETRIEVAL_K = 3