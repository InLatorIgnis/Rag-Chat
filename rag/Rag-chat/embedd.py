#!/usr/bin/env python3
"""
embeddings_to_store.py

Load pre‑computed embeddings from a CSV, and store them in a simple
SQLite table for quick retrieval and later comparison.

NOTE: This is a *toy* persistence layer. In production you’d probably
use Pinecone, Weaviate, or a specialized vector DB.
"""

import csv
import json
import logging
import sqlite3
import enums
from pathlib import Path
from sentence_transformers import SentenceTransformer


def init_db(conn: sqlite3.Connection):
    """Create table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id      TEXT PRIMARY KEY,
            vector  BLOB,
            text    TEXT
        )
    """)
    conn.commit()
# ------------------------------------------------------------------
def list_to_blob(vec):
    """Naïvely store a list of floats as a BLOB."""

    import pickle
    return pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def save_embeddings_csv(chunks, embeddings, out_path="embeddings.csv"):
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["id", "embedding", "text"])
        writer.writeheader()
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            writer.writerow({
                "id": f"chunk_{idx}",
                "embedding": json.dumps(emb.tolist() if hasattr(emb, "tolist") else emb),
                "text": chunk
            })
# ------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    conn = sqlite3.connect(enums.filePaths.DB_PATH.value)
    init_db(conn)

    # 1. Read chunks from processed CSV
    chunks = []
    with open("processed_data.csv", "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(row["text"])  # column with raw text

    # 2. Generate embeddings
    logging.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)  # uses SentenceTransformer

    # 3. Save embeddings to CSV
    save_embeddings_csv(chunks, embeddings, out_path="embeddings.csv")
    logging.info(f"Saved embeddings to embeddings.csv")

    # 4. Load embeddings CSV and store in SQLite
    EMBEDDINGS_CSV = Path("embeddings.csv")
    with EMBEDDINGS_CSV.open("r", encoding="utf-8", newline='') as fin:
        reader = csv.DictReader(fin)
        count = 0
        for row in reader:
            embedding = json.loads(row["embedding"])
            blob = list_to_blob(embedding)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO vectors (id, vector,text) VALUES (?, ?, ?)",
                    (row["id"], blob, row["text"])
                )
                count += 1
            except sqlite3.IntegrityError as e:
                logging.warning(f"Skipping duplicate id {row['id']}: {e}")

    conn.commit()
    logging.info(f"Stored {count} vectors in {enums.filePaths.DB_PATH}")
    conn.close()



if __name__ == "__main__":
    main()
