# Comparision-and-Implementation-of-Hybrid-Retriever-with-Sparse-Retriever-and-Dense-Retriever

## Overview

A Retrieval-Augmented Generation (RAG) system designed to answer programming questions related to Python and Java syntax by combining sparse retrieval (BM25) and dense retrieval (FAISS embeddings) using Reciprocal Rank Fusion (RRF). The system uses Google Gemini as the LLM to generate context-aware responses from retrieved documentation. Additionally, Tableau dashboards are used to perform business intelligence (BI) analysis and visualize the comparative performance of the hybrid, sparse, and dense retrieval methods.
---

## 🌐 Live Deployment
# Doc-Buddy: Hybrid Retrieval RAG for Programming Syntax

The application is publicly deployed using Streamlit Cloud:

https://hybridretriever.streamlit.app

The deployed app supports:

- Hybrid (RRF) retrieval
- Comparative query detection
- Structured LLM-generated answers
- End-to-end RAG pipeline

---

## Overview

Doc-Buddy ingests official Python and Java HTML documentation, preprocesses it into text chunks, indexes them in both a BM25 sparse index and a FAISS dense vector store, and at query time fuses results from both retrievers before passing the top context to Gemini 2.5 Flash Lite to generate a structured answer.

The system supports:
- Direct syntax and concept questions (`What is list comprehension in Python?`)
- Comparative questions (`Difference between ArrayList and LinkedList`) — automatically detected and handled with expanded retrieval (k=10)

---

## Project Structure

```
.
├── app/
│   └── streamlit_app.py         # Streamlit UI ("Doc-Buddy")
├── data/
│   ├── raw_docs/
│   │   ├── python/              # Official Python docs (HTML)
│   │   │   ├── library/         # exceptions.html, functions.html, stdtypes.html
│   │   │   └── tutorial/        # classes, controlflow, datastructures, errors, etc.
│   │   └── java/
│   │       └── api/java/        # io/, lang/, util/ — standard Java API docs
│   ├── processed_docs/
│   │   └── python_java_chunks.json   # Preprocessed chunks (output of preprocess_python_java.py)
│   └── faiss/
│       ├── index.faiss          # FAISS vector index
│       └── metadata.json        # Chunk metadata aligned with FAISS index
├── retrieval/
│   ├── bm25_retriever.py        # BM25Okapi sparse retriever
│   ├── faiss_retriever.py       # Dense retriever using all-MiniLM-L6-v2 + FAISS
│   └── hybrid_retriever.py      # RRF fusion of BM25 + FAISS
├── llm/
│   └── gemini_client.py         # Google Gemini 2.5 Flash Lite client
├── evaluation/
│   ├── metrics.py               # Precision@K, Recall@K, MRR
│   ├── queries.json             # 25 evaluation queries (Python, Java, cross-language)
│   ├── results.csv              # Evaluation results
│   └── run_evaluation.py        # Evaluation runner
├── preprocess_python_java.py    # HTML → chunks pipeline
├── rag_pipeline.py              # Core RAG query pipeline
└── requirements.txt
```

---

## Architecture

```
User Query
    │
    ├──▶ BM25Retriever (rank_bm25 + NLTK tokenizer)
    │         └── BM25Okapi scores over tokenized chunks
    │
    ├──▶ FAISSRetriever (all-MiniLM-L6-v2 + FAISS IndexFlatIP)
    │         └── Cosine similarity via normalized embeddings
    │
    └──▶ Reciprocal Rank Fusion (k=60)
              └── score = Σ 1 / (60 + rank)
                        │
                        ▼
              Top-k context chunks (truncated to 600 chars each)
                        │
                        ▼
              Gemini 2.5 Flash Lite
                        │
                        ▼
              Structured Answer (Language / Syntax / Explanation / Example / Common mistakes)
```

### Comparative Query Detection

If the query contains the keywords `compare`, `difference`, `vs`, or `versus`, `rag_pipeline.py` automatically expands retrieval to `k=10` to gather context from both languages before passing to the LLM.

---

## Data & Preprocessing

Raw data is official HTML documentation:

**Python** (from `data/raw_docs/python/`): `controlflow`, `datastructures`, `errors`, `classes`, `modules`, `inputoutput`, `expressions`, `stdtypes`, `functions`, `exceptions`, and more.

**Java** (from `data/raw_docs/java/api/`): `ArrayList`, `LinkedList`, `HashMap`, `HashSet`, `List`, `Map`, `Set`, `Collection`, `Object`, `Exception`, `RuntimeException`, `Thread`, `File`, `InputStream`, `OutputStream`.

`preprocess_python_java.py` processes these docs as follows:
1. Parses HTML with BeautifulSoup, strips `nav`, `footer`, `header`, `script`, `style`, `form` tags
2. Converts to Markdown via `markdownify`
3. Splits on Markdown `#` headings, then tokenizes by whitespace
4. Produces overlapping windows: **chunk size = 500 tokens**, **overlap = 80 tokens**
5. Each chunk is stored with `chunk_id`, `text`, `language`, `technology`, `source_file`, `source_path`
6. Output: `data/processed_docs/python_java_chunks.json`

---

## Retrievers

### BM25 (Sparse)
- Library: `rank_bm25` (`BM25Okapi`)
- Tokenization: NLTK `word_tokenize`, lowercased
- Returns ranked results with BM25 scores

### FAISS (Dense)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Index type: `faiss.IndexFlatIP` (inner product, i.e. cosine similarity with normalized embeddings)
- Batch size: 32 during index build
- Pre-built index persisted at `data/faiss/index.faiss`

### Hybrid (RRF)
- Fuses BM25 and FAISS result lists using Reciprocal Rank Fusion
- RRF formula: `score(d) = Σ 1 / (k + rank(d))` where `k=60`
- Returns top-k documents sorted by fused RRF score

---

## LLM Integration

**Model**: `gemini-2.5-flash-lite`  
**SDK**: `google-genai`  
**Parameters**: temperature=0.5, max_output_tokens=1024

The prompt instructs Gemini to answer only from provided context, structured as:
- Language / Syntax / Explanation / Example / Common mistakes
- For comparative queries: comparison table + syntax + one example per language

---

## Installation

**Requirements**: Python 3.12+

```bash
git clone https://github.com/DBDA-Projects/Comparision-and-Implementation-of-Hybrid-Retriever-with-Sparse-Retriever-and-Dense-Retriever.git
cd Comparision-and-Implementation-of-Hybrid-Retriever-with-Sparse-Retriever-and-Dense-Retriever

pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
```
beautifulsoup4==4.13.4
markdownify==1.2.2
nltk==3.9.2
rank_bm25==0.2.2
sentence_transformers==5.1.2
faiss-cpu
streamlit==1.39.0
google-genai
protobuf>=3.20,<6
```

---

## Usage

### Step 1: Preprocess Documents
Only required if the raw docs are updated or `python_java_chunks.json` is missing:
```bash
python preprocess_python_java.py
```

### Step 2: Build the FAISS Index
Only required if `data/faiss/index.faiss` is missing:
```bash
python -m retrieval.faiss_retriever
```

### Step 3: Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

### Run Evaluation
```bash
python -m evaluation.run_evaluation
```
Results are saved to `evaluation/results.csv`.

---

# 📊 Tableau Dashboard

An interactive Tableau dashboard visualizes:

- Precision comparison
- Recall comparison
- MRR comparison
- Query category breakdown
- Hybrid uplift analysis

(Data source: evaluation/results.csv)

**Dashboard link**: [View on Tableau Public](https://public.tableau.com/app/profile/mrunal.hadke/viz/Hybrid_retriever_DV/Dashboard1?publish=yes)

---

## Evaluation

Evaluation runs 25 queries across three categories against all three retrievers. Metrics computed at k=5: **Precision@5**, **Recall@5**, **MRR** (Mean Reciprocal Rank).

### Query Categories
- **Python** (queries 1–5, 11–15): list comprehension, for loops, functions, exceptions, modules, dictionaries
- **Java** (queries 6–10, 16–20): ArrayList, LinkedList, HashMap, List interface, Object class, RuntimeException
- **Cross-language** (queries 21–25): Python vs Java — lists, exceptions, collections, memory, dict vs HashMap

### Results (averaged over 25 queries)

| Retriever | Precision@5 | Recall@5 | MRR   |
|-----------|-------------|----------|-------|
| BM25      | 0.408       | 0.740    | 0.649 |
| FAISS     | 0.408       | 0.780    | 0.571 |
| **HYBRID**| **0.424**   | 0.760    | **0.725** |

**Key observations:**
- **Hybrid achieves the highest MRR (0.725)** — it consistently ranks the most relevant document higher than either retriever alone
- FAISS outperforms BM25 on Recall@5 (0.780 vs 0.740), reflecting its strength on semantic/paraphrase queries (e.g. *"compact way to create lists"* vs *"list comprehension"*)
- BM25 outperforms FAISS on keyword-heavy queries (e.g. *"ArrayList"*, *"HashMap"*)
- Both individual retrievers score 0.0 on certain abstract queries (*"How does Python organize reusable code?"*), showing the limits of single-strategy retrieval

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Sparse Retrieval | `rank_bm25` (BM25Okapi) |
| Dense Retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`) + `faiss-cpu` |
| Fusion | Reciprocal Rank Fusion (k=60) |
| LLM | Google Gemini 2.5 Flash Lite (`google-genai`) |
| Frontend | Streamlit |
| HTML Parsing | BeautifulSoup4 + markdownify |
| Tokenization | NLTK `word_tokenize` |

---

## Notes

- The FAISS index and preprocessed chunks are pre-built and committed to the repo (`data/` directory), so you can run the app directly without rebuilding
- Python 3.12 is specified in `.python-version`
- The `retrieval/` and `evaluation/` directories are Python packages (`__init__.py` included)
