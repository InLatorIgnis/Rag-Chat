#!/usr/bin/env python3
"""
preprocess.py

Read raw text data, tokenize it with a Hugging Face tokenizer,
and write the token IDs and attention masks to a new CSV file. 
"""
import enums
import csv
import logging
from pathlib import Path
import pdfplumber
# ------------------------------------------------------------------
from transformers import AutoTokenizer
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# ------------------------------------------------------------------
def process_row(text: str):
    """Tokenise `text` and return IDs + mask."""
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,          # adjust to your model’s limit
        return_tensors="np",
    )
    return encoded["input_ids"][0], encoded["attention_mask"][0]

def extract_chunks(pdf_path, chunk_size=500):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # Simple chunking by character count
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunks.append(chunk)
    return chunks

# ------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    RAW_PDF_PATH = enums.filePaths.RAW_PDF_PATH.value
    PROCESSED_CSV = enums.filePaths.PROCESSED_CSV.value
    if not RAW_PDF_PATH.exists():
        logging.error(f"Input file not found: {RAW_PDF_PATH}")
        return

    logging.info(f"Extracting chunks from {RAW_PDF_PATH}")
    chunks = extract_chunks(RAW_PDF_PATH)

    with PROCESSED_CSV.open("w", newline="", encoding="utf-8") as fout:
        fieldnames = ["text", "input_ids", "attention_mask"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for text in chunks:
            if not text.strip():
                continue
            input_ids, attn_mask = process_row(text)
            writer.writerow({
                "text": text,
                "input_ids": input_ids.tolist(),
                "attention_mask": attn_mask.tolist(),
            })

    logging.info(f"Processed data written to {PROCESSED_CSV}")

if __name__ == "__main__":
    main()
