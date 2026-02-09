#!/usr/bin/env python3
"""
Streamlit UI for retrieving from Chroma collections.

Run with: streamlit run src/ui/app.py
"""
import streamlit as st
from pathlib import Path
import tempfile

from src.config import PERSIST_DIR, MODEL_NAME, TOP_K, GROQ_API_KEY, GROQ_MODEL_NAME
from src.core.vectorstore import get_db
from src.core.rag_service import setup_rag_chain, build_custom_rag_chain
from src.core.utils import (
    perform_retrieve,
    _display_collection_info,
    display_results,
    list_collections_with_stats,
)
from langchain_chroma.vectorstores import Chroma
from src.ingest.build_chroma import build_chroma_from_pdf


def display_collection_info(db: Chroma, collection_name: str, persist_dir: str, model_name: str, document_name: str = None):
    """Display information about the selected collection."""
    _display_collection_info(db, collection_name, persist_dir, model_name, document_name, st.subheader, st.write)


def main():
    st.set_page_config(page_title="RFP Analysis")
    st.title("RFP Analysis")

    persist_dir = str(Path(PERSIST_DIR).expanduser())
    if not Path(persist_dir).exists():
        st.error(f"Chroma persist directory not found: {persist_dir}")
        return

    # Sidebar for selections
    st.sidebar.header("Options")
    action = st.sidebar.selectbox(
        "Choose action",
        ["List Collections", "Create Collection", "Retrieve", "Q &A"],
        index=3  # Default to Q &A
    )

    # Dynamically fetch collections from Chroma
    stats = list_collections_with_stats(persist_dir)
    collection_names = [s.get("name") for s in stats if s.get("name")]
    if not collection_names:
        st.sidebar.info("No collections found in the database.")
        selected_collection_name = None
        selected_sources = []
    else:
        default_collection = "RFP_Ryapte"
        default_index = collection_names.index(default_collection) if default_collection in collection_names else 0
        selected_collection_name = st.sidebar.selectbox(
            "Choose collection", collection_names, index=default_index
        )
        selected = next((s for s in stats if s.get("name") == selected_collection_name), None)
        selected_sources = (selected.get("sample_sources") if isinstance(selected, dict) else []) or []
    pdf_name = selected_sources[0] if selected_sources else None

    # Load the collection
    db = get_db(selected_collection_name) if selected_collection_name else None

    if action == "List Collections":
        st.header("All Collections")
        stats = list_collections_with_stats(persist_dir)
        if not stats:
            st.info("No collections found.")
        else:
            for s in stats:
                st.subheader(f"{s.get('name')}")
                st.write(f"Items: {s.get('count')}")
                sources = s.get('sample_sources') or []
                st.write(f"Collection sources: {sources}")

    elif action == "Create Collection":
        st.header("Create and Ingest Collection")
        new_collection = st.text_input("Enter new collection name")
        uploaded_pdf = st.file_uploader("Select a PDF", type=["pdf"])
        if st.button("Ingest"):
            if not new_collection:
                st.warning("Please enter a collection name.")
            elif not uploaded_pdf:
                st.warning("Please select a PDF file to upload.")
            else:
                with st.spinner("Ingesting PDF into Chroma..."):
                    try:
                        # Save uploaded file to a temporary path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_pdf.getbuffer())
                            tmp_path = tmp.name

                        # Build Chroma with defaults from config
                        build_chroma_from_pdf(
                            pdf_path=tmp_path,
                            persist_dir=persist_dir,
                            model_name=MODEL_NAME,
                            chunk_size=1024,
                            chunk_overlap=100,
                            encoding_name="cl100k_base",
                            collection_name=new_collection,
                        )
                        st.success(f"Ingestion complete: collection '{new_collection}' created.")
                        st.caption("This collection should now appear in the selectors above.")
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")

    elif action == "Retrieve":
        st.header("Retrieve from the Document")
        if not selected_collection_name:
            st.warning("No collection selected.")
            return
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            if not query:
                st.warning("Please enter a query.")
            else:
                results = perform_retrieve(db, query, TOP_K)
                display_results(results, st.subheader, st.write)

    elif action == "Display":
        st.header("Document Information")
        if not selected_collection_name:
            st.warning("No collection selected.")
        else:
            display_collection_info(db, selected_collection_name, persist_dir, MODEL_NAME, pdf_name)

    elif action == "Q &A":
        st.header("Ask a Question")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not set in .env")
        else:
            if not selected_collection_name:
                st.warning("No collection selected.")
                return
            # Set up custom RAG chain with default prompt (strict context)
            qa_chain = build_custom_rag_chain(db, TOP_K)

            # Show which collection is in use for Q &A, include source if known
            if pdf_name:
                st.caption(f"Collection: {selected_collection_name} — Source: {pdf_name}")
            else:
                st.caption(f"Collection: {selected_collection_name}")
            st.caption(f"Embedding Model: {MODEL_NAME}")
            st.caption(f"Generation Model: {GROQ_MODEL_NAME}")
            query = st.text_input("Enter your question:")
            if 'qa_answer' not in st.session_state:
                st.session_state.qa_answer = ''
                st.session_state.citation_map = {}
                st.session_state.chunk_to_show = None
            if st.button("Generate Answer"):
                if not query:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Generating answer..."):
                        try:
                            result = qa_chain.invoke(query)
                            answer = result.get('result', '')
                            sources = result.get('source_documents', [])
                            citation_map = {}
                            for ref_num, doc in enumerate(sources, start=1):
                                chunk_id = doc.metadata.get('chunk_id', None)
                                citation_map[ref_num] = {'chunk_id': chunk_id, 'doc': doc}
                            if sources:
                                citations = ' '.join(f'[^{i}]' for i in citation_map.keys())
                                answer = f"{answer}\n\n{citations}"
                            st.session_state.qa_answer = answer
                            st.session_state.citation_map = citation_map
                            st.session_state.chunk_to_show = None
                        except Exception as e:
                            st.error(f"Error: {e}")
            # Display answer and references if available
            if st.session_state.qa_answer:
                # Remove only trailing citation markers (if present), keep full answer
                answer_text = st.session_state.qa_answer
                # If answer ends with citation markers, remove them
                parts = answer_text.rsplit('\n\n', 1)
                if len(parts) == 2 and all(s.strip().startswith('[^') for s in parts[1].split()):
                    answer_text = parts[0]
                st.markdown("**Answer:**")
                st.markdown(answer_text)
                st.markdown("**References:**")
                # Display all references as buttons labeled by reference number
                ref_buttons = []
                for ref_num, ref_info in st.session_state.citation_map.items():
                    ref_buttons.append((ref_num, ref_info['chunk_id'], ref_info['doc']))
                if ref_buttons:
                    cols = st.columns(len(ref_buttons))
                    for idx, (ref_num, chunk_id, doc) in enumerate(ref_buttons):
                        btn_label = f"[{ref_num}]"
                        if cols[idx].button(btn_label, key=f"show_chunk_{ref_num}"):
                            st.session_state.chunk_to_show = (ref_num, chunk_id, doc)
                # Show the selected chunk if any
                if st.session_state.chunk_to_show:
                    ref_num, chunk_id, doc = st.session_state.chunk_to_show
                    chunk_label = f"**Reference [{ref_num}]"
                    if chunk_id is not None:
                        chunk_label += f" (chunk_id: {chunk_id})"
                    chunk_label += "**"
                    st.markdown(chunk_label)
                    st.write(doc.page_content)


if __name__ == "__main__":
    main()
