"""Vector store and embeddings factories."""
from __future__ import annotations

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

from src.config import MODEL_NAME, PERSIST_DIR


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def get_db(collection_name: str) -> Chroma:
    emb = get_embeddings()
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=emb, collection_name=collection_name)
