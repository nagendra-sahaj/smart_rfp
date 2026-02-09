
# RFP Analysis: PDF → Chroma + Retrieval + Q&A


This project ingests RFP/contract PDFs into a Chroma vector store (HuggingFace embeddings), provides CLI and Streamlit UIs for retrieval, and supports Q&A answers via Groq.


Features:
- PDF ingestion with selectable loaders (PyMuPDF, PDFPlumber, PyPDF)
- Configurable chunking (size/overlap) with character-based splitter
- HuggingFace embeddings with configurable model
- Persistent Chroma vector store with per-collection counts
- Dynamic collection discovery in UI and CLI (from Chroma)
- Streamlit UI: List Collections, Create Collection (PDF upload), Retrieve, Ask a Question (Q&A)
- Unified CLI: List, Display, Retrieve, RAG, Quit (1–5) with runner script
- Retrieval via similarity search (optional scores) with source-aware snippets
- Q&A answers via Groq (configurable model and API key)
- Source metadata stored per chunk for traceability
- Centralized configuration via `.env` and [src/config.py](src/config.py)
- Easy runs with `uv`; helper scripts [run_streamlit.sh](run_streamlit.sh) and [run_cli.sh](run_cli.sh)

## Layout

- `src/config.py`: Centralized `.env` config and constants
- `src/core/`: Backend services
	- `vectorstore.py`: Embeddings/Chroma factories
	- `rag_service.py`: RAG chain setup (Groq)
	- `utils.py`: Display-agnostic helpers (retrieve, render results, list collections)
- `src/ingest/`: Ingestion tools
	- `build_chroma.py`: PDF -> chunks -> embeddings -> Chroma
- `src/cli/`: Command-line tools
	- `contracts_cli.py`: Unified CLI (List, Display, Retrieve, RAG, Quit)
- `src/ui/`: Streamlit UI
	- `app.py`: “RFP Analysis” app (List Collections, Create Collection, Retrieve, Ask a Question)

## Quick start

1) Install dependencies (uv recommended)

```bash
uv pip install -r requirements.txt
```

2) Configure `.env`

Set at minimum:
- `PDF_PATH`, `PERSIST_DIR`, `MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `COLLECTION_NAME` (optional; defaults to PDF filename stem)
- `GROQ_API_KEY` and `GROQ_MODEL_NAME` for RAG
- `PDF_LOADER` (optional: `pymupdf` | `pdfplumber` | `pypdf`)

3) Ingest a PDF (via script)

```bash
uv run python src/ingest/build_chroma.py
```


4) Run the Streamlit UI

```bash
./run_streamlit.sh
```

If running manually:

```bash
PYTHONPATH="$(pwd)" uv run streamlit run src/ui/app.py
```

5) CLI tools

```bash
uv run python -m src.cli.contracts_cli
./run_cli.sh
```

CLI actions (1–5):
- 1: List collections — shows name, item count, and collection sources (sampled)
- 2: Display info — shows chunks count, embedding model, and persist directory
- 3: Retrieve — similarity search and snippet display
- 4: RAG — Groq-backed answer generation using retrieved context
- 5: Quit


Streamlit UI actions:
- List Collections — shows live collections, items, and collection sources
- Create Collection — provide a new collection name and upload a PDF to ingest
- Retrieve — similarity search on the selected collection
- Ask a Question — answer generation (Q&A) for the selected collection

## Notes

- Each chunk stores `source` metadata with the PDF filename for traceability.
- Use `PDF_LOADER=pymupdf` for better whitespace preservation with some PDFs.
- TOP_K and model names are controlled via `src/config.py` + `.env`.
- UI and CLI now load collections dynamically from Chroma; no need to edit a static list to see new collections.

## Example: query via code

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=emb, collection_name="Construction_Contract")
docs = db.similarity_search("your query", k=5)
```
