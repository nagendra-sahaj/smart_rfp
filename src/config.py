"""Centralized configuration for the Contracts RAG project."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

# Load .env once here
load_dotenv()

# Disable tokenizers parallelism to avoid warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Core paths and models
PERSIST_DIR: str = str(Path(os.getenv("PERSIST_DIR", "../chroma_db")).expanduser())
MODEL_NAME: str = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K: int = int(os.getenv("TOP_K", "5") or 5)

# Groq config
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

# PDF ingestion config
PDF_LOADER: str = os.getenv("PDF_LOADER", "pymupdf").strip().lower()

# Collections displayed in UI/CLI
COLLECTIONS: List[Tuple[str, str]] = [
    ("Construction_Agreement", "Construction_Agreement.pdf"),
    ("Construction_Contract", "Construction_Contract-for-Major-Works.pdf"),
    ("Construction_Contract2", "Construction_Contract-for-Major-Works.pdf"),
]

__all__ = [
    "PERSIST_DIR",
    "MODEL_NAME",
    "TOP_K",
    "GROQ_API_KEY",
    "GROQ_MODEL_NAME",
    "PDF_LOADER",
    "COLLECTIONS",
]
