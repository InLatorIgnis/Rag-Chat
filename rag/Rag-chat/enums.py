from pathlib import Path
from enum import Enum
#-------------------------------------------------------------------------
class filePaths(Enum):
    PREPROCESSED_CSV = Path("processed_data.csv")  # expects columns: id,text
    EMBEDDINGS_CSV = Path("embeddings.csv")  # expects columns: id,embedding
    DB_PATH        = Path("embeddings.db")# SQLite database
    RAW_PDF_PATH    = Path("docs/Tentamen___Linjyr_Algebra_FMAA55_2025_06_05.pdf")      # <‑‑ raw input file
    RAW_CSV_PATH    = Path("raw_data.csv")      # <‑‑ raw input file
    PROCESSED_CSV   = Path("processed_data.csv")  # output


#-------------------------------------------------------------------------
class modelNames(str, Enum):
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
    LLM_MODEL      = "gpt-oss:20b"       # Ollama model
    HUGGINGFACE_MODEL = "openai/gpt-oss-20b"         # HuggingFace model
