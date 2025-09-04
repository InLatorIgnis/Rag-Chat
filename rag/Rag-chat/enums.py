from pathlib import Path
from enum import Enum
#-------------------------------------------------------------------------
class filePaths(Enum):
    PREPROCESSED_CSV = Path("processed_data.csv")  # expects columns: id,text
    EMBEDDINGS_CSV = Path("embeddings.csv")  # expects columns: id,embedding
    DB_PATH        = Path("embeddings.db")# SQLite database
    RAW_PDF_PATH    = Path("docs/sds.pdf")      # <‑‑ raw input file
    RAW_CSV_PATH    = Path("docs/Candidate thesis gant chart-Blad1(1).csv")      # <‑‑ raw input file
    PROCESSED_CSV   = Path("processed_data.csv")  # output


#-------------------------------------------------------------------------
class modelNames(str, Enum):
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
    LLM_MODEL      = "gpt-oss:20b"       # Ollama model
    HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"         # HuggingFace model should be same as EMBEDDING_MODEL to preserve tokeniser compatibility
