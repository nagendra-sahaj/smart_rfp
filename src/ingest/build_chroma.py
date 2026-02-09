#!/usr/bin/env python3
"""
Build a Chroma vector store from a PDF using LangChain + HuggingFace embeddings.

Reads configuration from .env via src/config and supports PDF loader selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
)
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

load_dotenv()


def build_chroma_from_pdf(
    pdf_path: str,
    persist_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    encoding_name: str = "cl100k_base",
    collection_name: Optional[str] = None,
) -> None:
    loader_choice = os.getenv("PDF_LOADER", "pymupdf").strip().lower()
    if loader_choice == "pdfplumber":
        loader = PDFPlumberLoader(pdf_path)
    elif loader_choice in ("pymupdf", "fitz"):
        loader = PyMuPDFLoader(pdf_path)
    else:
        loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs_split = splitter.split_documents(docs)

    for idx, doc in enumerate(docs_split):
        md = dict(doc.metadata or {})
        md["source"] = Path(pdf_path).name
        md["chunk_id"] = idx
        doc.metadata = md

    print(f"Created {len(docs_split)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap}).")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    persist_dir = str(Path(persist_dir).expanduser())
    vectordb = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    print(f"Chroma DB persisted to: {persist_dir} (collection: {collection_name})")


def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def main() -> None:
    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path:
        raise SystemExit("PDF_PATH not set in .env")

    persist_dir = os.getenv("PERSIST_DIR", "./chroma_db")
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = _get_env_int("CHUNK_SIZE", 1024)
    chunk_overlap = _get_env_int("CHUNK_OVERLAP", 100)
    encoding_name = os.getenv("ENCODING_NAME", "cl100k_base")
    collection_name = os.getenv("COLLECTION_NAME")

    if not collection_name:
        collection_name = Path(pdf_path).stem

    pdf_path = str(Path(pdf_path).expanduser())
    if not Path(pdf_path).exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    build_chroma_from_pdf(
        pdf_path=pdf_path,
        persist_dir=persist_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    main()
