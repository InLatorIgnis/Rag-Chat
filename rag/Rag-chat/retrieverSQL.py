import sqlite3
import pickle
import numpy as np
import csv
import json
import enums

class retrieverSQL:
    def __init__(self, db_path, top_k=5):
        self.db_path = enums.filePaths.DB_PATH
        self.top_k = top_k

    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve(self, query_embedding, top_k=None):
        if top_k is None:
            top_k = self.top_k

        conn = sqlite3.connect(self.db_path.value)
        cursor = conn.execute("SELECT id, vector, text FROM vectors")

        results = []
        for doc_id, blob, text in cursor:
            embedding = pickle.loads(blob)
            sim = self.cosine_similarity(query_embedding, embedding)
            results.append((text, sim))

        conn.close()
        results.sort(key=lambda x: x[1], reverse=True)
        return [t for t, s in results[:top_k]]


if __name__ == "__main__":
    # Example usage
    retriever = retrieverSQL(enums.filePaths.DB_PATH.value)

    # Example: load an embedding from CSV
    import csv, json
    with open("embeddings.csv", "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        first_row = next(reader)
        query = json.loads(first_row["embedding"])

    top_docs = retriever.retrieve(query, top_k=2)
    for doc in top_docs:
        print(f"Doc: {doc}")