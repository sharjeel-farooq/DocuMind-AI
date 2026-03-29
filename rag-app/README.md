RAG App
=======

# DocuMind AI 🧠📄

An intelligent Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask natural language questions to extract insights instantly.

## 🚀 Features

* 📄 Upload and process PDF documents
* ✂️ Recursive, semantic-aware text chunking
* 🔍 Semantic search using embeddings
* 🤖 AI-powered question answering
* ⚡ Fast API built with FastAPI

## 🛠️ Tech Stack

* Python
* FastAPI
* ChromaDB
* Sentence Transformers
* LangChain

## 📌 How it works

1. Upload complex, unstructured docs (reports, contracts, manuals)
2. Text is recursively chunked with semantic filters
3. Embeddings stored in the vector database
4. User query → semantic search over chunks
5. LLM generates the final answer grounded in retrieved context

## ▶️ Run Locally

```
git clone https://github.com/yourusername/documind-ai.git
cd documind-ai

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
uvicorn rag.api:app --reload
```

## 📊 Load Testing

Tested with Locust to evaluate performance and concurrency limits.

## 📈 Future Improvements

* Add UI (Streamlit / React)
* Optimize latency
* Add authentication
* Deploy to cloud

---

## 👨‍💻 Author

Sharjeel

Run instructions
----------------

1) Install deps (inside your venv):

```
pip install -r requirements.txt
```

2) Set required environment variables:

- `HUGGINGFACEHUB_API_TOKEN` (or `HF_TOKEN`): token for embeddings
- `GROQ_API_KEY`: LLM key

Optional overrides: `EMBEDDING_MODEL`, `GENERATION_MODEL` (default: llama-3.1-8b-instant), `TOP_K` (default: 5), `CHUNK_SIZE` (default: 300).

3) Run via Streamlit UI:

```
streamlit run frontend/app.py
```

- Upload one or more docs (pdf, txt, csv, docx, json, xlsx, xls)
- Click “Clear index” to wipe Chroma if needed
- Ask questions; sources are shown with snippets

4) Run via FastAPI:

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Upload docs (multipart):

```
curl -X POST "http://127.0.0.1:8000/upload" \
    -F "files=@\"data_api_testing/Annual_Report_Q4.pdf\"" \
    -F "files=@\"data_api_testing/Customer_Success_Playbook.docx\""
```

- Ask a question:

```
curl -X POST "http://127.0.0.1:8000/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"compare these 2 docs\"}"
```

- Batch eval:

```
curl -X POST "http://127.0.0.1:8000/eval" \
    -H "Content-Type: application/json" \
    -d "{\"items\":[{\"query\":\"compare these 2 docs\",\"expected_substrings\":[\"web\",\"machine\"]}]}"
```

Operational features
--------------------

- Latency: `/query` returns `latency_ms` for every response.
- Rate limiting: simple per-client throttle on `/query` (429 if called too fast).
- Answer cache: in-memory LRU (size 100) for repeated queries; cleared on ingest and reset.
- Eval: `/eval` measures accuracy against expected substrings.
- Logging: retrieval and LLM timings logged via `utils/logger.py`.

Quick stress test (example)
---------------------------

```
import requests

for i in range(50):
    r = requests.post("http://localhost:8000/query", json={"query": "task 3"})
    print(i, r.status_code, r.elapsed.total_seconds())
```

Locust load test
----------------

1) Start the API (`uvicorn app.main:app --host 0.0.0.0 --port 8000`).
2) Make sure the index has documents (upload via UI or `POST /upload`) so responses are meaningful.
3) Run Locust against the API:

```
locust -f locustfile.py -H http://localhost:8000
```

- UI: http://localhost:8089 → set users and spawn rate. The tasks hit `/query` with prompts from `QUERIES`.
- Headless example (1 minute, 10 users, spawn 2/s):

```
locust -f locustfile.py --headless -u 10 -r 2 -t 1m -H http://localhost:8000 --csv locust_run
```

Notes:
- API has a simple per-host rate limit (~1 req/s); `locustfile.py` uses 1–2s wait to avoid 429s. Adjust users/spawn if you see throttle errors.
- CSV output saved when `--csv` is provided (e.g., `locust_run_stats.csv`).

Suggested note: “Tested API with 50 concurrent requests; maintained stable response times (~X ms avg).”


Project structure
-----------------

frontend → FastAPI → pipeline.py →
        loader → chunker → embedder → vectorstore →
        retriever → generator → response


rag-app/
│
├── app/                     # FastAPI app and routes
│   ├── main.py              # App factory
│   ├── routes.py            # Upload/query/eval endpoints
│   └── schemas.py           # Request/response models
│
├── rag/                     # Core RAG pipeline
│   ├── loader.py            # File loaders (pdf/txt/csv/xls/xlsx/docx/json)
│   ├── chunker.py           # Text splitting and math symbol stripping
│   ├── embedder.py          # HF embeddings client
│   ├── vectorstore.py       # Chroma persistence helpers
│   ├── retriever.py         # Embedding-based retrieval
│   ├── generator.py         # Intent-aware LLM generation
│   └── pipeline.py          # Orchestration (ingest, clean, embed, store)
│
├── data/                    # Local data storage
│   ├── raw/                 # Uploaded raw files
│   └── processed/           # Cleaned/chunked data (if used)
│
├── db/                      # Vector DB persistence
│   └── chroma/              # Chroma files
│
├── frontend/                # Streamlit UI
│   └── app.py
│
├── utils/                   # Shared helpers
│   ├── config.py            # Env-driven settings
│   ├── logger.py            # Logging config
│   └── helper.py            # (if present)
│
├── tests/                   # Unit tests
│   ├── test_retriever.py
│   └── test_pipeline.py
│
├── requirements.txt
├── README.md
└── run.py                   # Scratch/entry script
frontend → FastAPI → pipeline.py →
    loader → chunker → embedder → vectorstore →
    retriever → generator → response
