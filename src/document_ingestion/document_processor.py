"""Document processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF files, PDF directories, or TXT files
        """
        docs: List[Document] = []

        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)

            if not path.exists():
                raise ValueError(f"Path does not exist: {src}")

            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. Use URL, .pdf, .txt, or PDF directory."
                )

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        """
        return self.splitter.split_documents(documents)

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)