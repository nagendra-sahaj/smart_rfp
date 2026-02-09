#!/usr/bin/env python3
"""Unified CLI for Chroma collections: Display, Retrieve, and RAG."""
from __future__ import annotations

import sys
from pathlib import Path

from src.config import PERSIST_DIR, MODEL_NAME, TOP_K, GROQ_API_KEY
from src.core.vectorstore import get_db
from src.core.utils import (
    perform_retrieve,
    _display_collection_info,
    display_results,
    list_collections_with_stats,
)
from src.core.rag_service import setup_rag_chain, build_custom_rag_chain


def _print_block(text: str) -> None:
    print(f"\n{text}")


def select_collection(persist_dir: str) -> tuple[str | None, str | None]:
    stats = list_collections_with_stats(persist_dir)
    if not stats:
        print(f"No collections found in: {persist_dir}")
        return None, None

    print("Available collections:")
    for i, s in enumerate(stats, start=1):
        name = s.get("name")
        count = s.get("count")
        print(f"{i}. {name} (items: {count})")

    while True:
        try:
            choice = int(input(f"Select collection (1-{len(stats)}): "))
            if 1 <= choice <= len(stats):
                sel = stats[choice - 1]
                name = sel.get("name")
                sources = sel.get("sample_sources") or []
                pdf_name = sources[0] if sources else None
                return name, pdf_name
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(stats)}.")
        except ValueError:
            print("Please enter a number.")


def display_collection_info(db, collection_name: str, persist_dir: str, model_name: str, document_name: str | None = None) -> None:
    _display_collection_info(db, collection_name, persist_dir, model_name, document_name, _print_block, print)


def main() -> None:
    persist_dir = str(Path(PERSIST_DIR).expanduser())
    if not Path(persist_dir).exists():
        raise SystemExit(f"Chroma persist directory not found: {persist_dir}")

    while True:
        print("Choose mode: \n 1. List collections \n 2. Display info \n 3. Retrieve \n 4. RAG \n 5. Custom RAG \n 6. Quit ")
        mode = input("Choose mode: ").strip()
        if mode not in ['1', '2', '3', '4', '5', '6']:
            print("Invalid choice. Please select 1, 2, 3, 4, 5, or 6.")
            continue
        if mode == '6':
            print("Exiting.")
            break

        if mode == '1':
            stats = list_collections_with_stats(persist_dir)
            if not stats:
                print(f"No collections found in: {persist_dir}")
            else:
                print("\nCollections:")
                for s in stats:
                    print(f"- {s.get('name')}")
                    print(f"  Items: {s.get('count')}")
                    sources = s.get('sample_sources') or []
                    print(f"  Collection sources: {sources}")
            continue

        collection_name, pdf_name = select_collection(persist_dir)
        if not collection_name:
            continue
        db = get_db(collection_name)

        if mode == '2':
            display_collection_info(db, collection_name, persist_dir, MODEL_NAME, pdf_name)
        elif mode == '3':
            query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your query: ")
            if not query:
                print("No query provided. Continuing.")
                continue
            results = perform_retrieve(db, query, TOP_K)
            display_results(results, _print_block, print)
        elif mode == '4':  # RAG
            if not GROQ_API_KEY:
                print("GROQ_API_KEY not set. Please set it in your environment to use RAG.")
                continue
            chain = setup_rag_chain(db, top_k=TOP_K)
            question = input("Enter your RAG question: ").strip()
            if not question:
                print("No question provided. Continuing.")
                continue
            try:
                answer = chain.invoke({"query": question})
                if isinstance(answer, dict) and "result" in answer:
                    print(f"\nAnswer:\n{answer['result']}")
                else:
                    print(f"\nAnswer:\n{answer}")
            except Exception as e:
                print(f"Error running RAG chain: {e}")
        elif mode == '5':  # Custom RAG
            if not GROQ_API_KEY:
                print("GROQ_API_KEY not set. Please set it in your environment to use Custom RAG.")
                continue
            chain = build_custom_rag_chain(db, TOP_K)
            question = input("Enter your Custom RAG question: ").strip()
            if not question:
                print("No question provided. Continuing.")
                continue
            try:
                answer = chain.invoke(question)
                print(f"\nCustom RAG Answer:\n{answer}")
            except Exception as e:
                print(f"Error running Custom RAG chain: {e}")

        cont = input("\nDo you want to perform another action? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
