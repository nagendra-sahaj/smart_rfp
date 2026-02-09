"""Common utilities for UI/CLI display and retrieval helpers."""
import os
from typing import Any, Dict, List

from chromadb import PersistentClient

from src.config import COLLECTIONS


def get_directory_size(path: str) -> str:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                pass
    if total_size < 1024 * 1024:
        return f"{total_size / 1024:.2f} KB"
    else:
        return f"{total_size / (1024 * 1024):.2f} MB"


def perform_retrieve(db, query: str, top_k: int) -> list:
    try:
        results = db.similarity_search_with_score(query, k=top_k)
    except Exception:
        docs = db.similarity_search(query, k=top_k)
        results = [(d, None) for d in docs]
    return results


def _display_collection_info(db, collection_name: str, persist_dir: str, model_name: str, document_name: str = None, subheader_func=None, write_func=None):
    try:
        count = db._collection.count()
        subheader_func(f"Collection: {collection_name}")
        if document_name:
            write_func(f"Document: {document_name}")
        write_func(f"Number of chunks: {count}")
        write_func(f"Embedding model: {model_name}")
        write_func(f"Persist directory: {persist_dir}")
    except Exception as e:
        write_func(f"Error retrieving collection info: {e}")


def display_results(results: list, subheader_func, write_func):
    for i, (doc, score) in enumerate(results, start=1):
        subheader_func(f"Result #{i}")
        if score is not None:
            write_func(f"Score: {score}")
        src = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
        if src:
            write_func(f"Source: {src}")
        text = doc.page_content.strip()
        snippet = text if len(text) < 800 else text[:800] + "..."
        write_func(snippet)


def list_collections_with_stats(persist_dir: str, sample_limit: int = 5) -> List[Dict[str, Any]]:
    """Return collection stats from Chroma persistent client.

    Each entry contains: name, count, sample_sources.
    """
    client = PersistentClient(path=persist_dir)
    results: List[Dict[str, Any]] = []
    for coll in client.list_collections():
        info: Dict[str, Any] = {
            "name": getattr(coll, "name", None),
            "count": 0,
            "sample_sources": [],
        }
        try:
            info["count"] = coll.count()
        except Exception:
            info["count"] = None
        try:
            sample = coll.get(limit=sample_limit, include=["metadatas"])
            metas = sample.get("metadatas") or []
            sources = {
                m.get("source")
                for m in metas
                if isinstance(m, dict) and m.get("source")
            }
            info["sample_sources"] = sorted(sources)
        except Exception:
            info["sample_sources"] = []
        results.append(info)
    return results
