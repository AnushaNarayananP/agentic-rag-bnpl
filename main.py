"""Main application entry point for Agentic RAG system"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


class AgenticRAG:
    """Main BNPL Agentic RAG application"""

    def __init__(self, sources=None):
        """
        Initialize Agentic RAG system
        """
        print("🚀 Initializing Agentic RAG System...")

        self.sources = sources or []

        self.llm = Config.get_llm()
        self.doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore()

        self._setup_vectorstore()

        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )
        self.graph_builder.build()

        print("✅ System initialized successfully!\n")

    def _setup_vectorstore(self):
        """Setup vector store with processed documents"""
        print(f"📄 Processing {len(self.sources)} sources...")
        documents = self.doc_processor.load_documents(self.sources)
        documents = self.doc_processor.split_documents(documents)
        print(f"📊 Created {len(documents)} document chunks")

        print("🔍 Creating vector store...")
        self.vector_store.create_vectorstore(documents)

    def ask(self, question: str) -> str:
        """Ask a question to the RAG system"""
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")

        result = self.graph_builder.run(question)
        answer = result["answer"]

        print(f"✅ Answer: {answer}\n")
        return answer

    def interactive_mode(self):
        """Run in interactive mode"""
        print("💬 Interactive Mode - Type 'quit' to exit\n")

        while True:
            question = input("Enter your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if question:
                self.ask(question)
                print("-" * 80 + "\n")


def load_sources():
    """Load local documents and URLs"""
    sources = []

    pdf_path = Path("data/BNPL_Customer_Support_Handbook.pdf")
    if pdf_path.exists():
        sources.append(str(pdf_path))

    urls_file = Path("data/url.txt")
    if urls_file.exists():
        with open(urls_file, "r", encoding="utf-8") as f:
            sources.extend([line.strip() for line in f if line.strip()])

    return sources


def main():
    """Main function"""
    sources = load_sources()

    if not sources:
        print("❌ No data sources found in data/BNPL_Customer_Support_Handbook.pdf or data/url.txt")
        return

    rag = AgenticRAG(sources=sources)

    example_questions = [
    "How does BNPL work?",
    "What happens if a customer misses a payment?",
    "Are late fees charged in BNPL?",
    "How are refunds handled for installment purchases?",
]

    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")

    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    user_input = input("Would you like to enter interactive mode? (y/n): ")
    if user_input.lower() == "y":
        rag.interactive_mode()


if __name__ == "__main__":
    main()