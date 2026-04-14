"""Streamlit UI for Agentic RAG System"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []


def load_sources():
    sources = []

    pdf_path = Path("data/BNPL_Customer_Support_Handbook.pdf")
    if pdf_path.exists():
        sources.append(str(pdf_path))

    urls_file = Path("data/url.txt")
    if urls_file.exists():
        with open(urls_file, "r", encoding="utf-8") as f:
            sources.extend([line.strip() for line in f if line.strip()])

    return sources


@st.cache_resource
def initialize_rag():
    try:
        llm = Config.get_llm()

        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()

        sources = load_sources()
        documents = doc_processor.load_documents(sources)
        documents = doc_processor.split_documents(documents)

        vector_store.create_vectorstore(documents)

        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0


def main():
    init_session_state()

    st.title("💳 BNPL AI Customer Support Assistant")
    st.markdown("Ask questions about BNPL policies, payments, and customer support")
    st.info("📄 Knowledge Base: BNPL Policy PDF + Fintech Help Pages (Sezzle, Klarna, Afterpay)")

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")

    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()

                result = st.session_state.rag_system.run(question)

                elapsed_time = time.time() - start_time

                st.session_state.history.append({
                    "question": question,
                    "answer": result["answer"],
                    "time": elapsed_time
                })

                st.markdown("### 💡 Answer")
                st.success(result["answer"])

                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:500],
                            height=150,
                            disabled=True
                        )

                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}...")
            st.caption(f"Time: {item['time']:.2f}s")


if __name__ == "__main__":
    main()