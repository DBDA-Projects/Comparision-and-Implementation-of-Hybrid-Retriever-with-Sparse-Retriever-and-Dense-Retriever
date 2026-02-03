import os
import re
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import nltk

nltk.download("punkt")

RAW_DOCS_DIR = "data/raw_docs"
OUTPUT_FILE = "data/processed_docs/python_java_chunks.json"

CHUNK_SIZE = 500
OVERLAP = 80

SUPPORTED_LANGS = ["python", "java"]

def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup([
        "nav", "footer", "header", "aside",
        "script", "style", "form"
    ]):
        tag.decompose()

    text = md(str(soup))

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def chunk_text(text):
    sections = re.split(r"\n(?=# )", text)
    chunks = []

    for section in sections:
        tokens = section.split()
        start = 0

        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            start = end - OVERLAP

    return chunks

def preprocess_docs():
    all_chunks = []
    chunk_id = 0

    for lang in SUPPORTED_LANGS:
        lang_path = os.path.join(RAW_DOCS_DIR, lang)

        if not os.path.exists(lang_path):
            print(f"Folder not found: {lang_path}")
            continue

        for root, _, files in os.walk(lang_path):
            for file in files:
                if not file.endswith(".html"):
                    continue

                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()

                cleaned_text = clean_html(html)
                chunks = chunk_text(cleaned_text)

                for chunk in chunks:
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "language": lang,
                        "technology": lang,
                        "source_file": file,
                        "source_path": root.replace(RAW_DOCS_DIR + "/", "")
                    })
                    chunk_id += 1

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Preprocessing complete: {len(all_chunks)} chunks created")

if __name__ == "__main__":
    preprocess_docs()
