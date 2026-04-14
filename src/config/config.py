"""Configuration for Agentic RAG"""

from langchain_ollama import ChatOllama


class Config:
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # keep empty if you want to load from local data files
    DEFAULT_URLS = []

    @staticmethod
    def get_llm():
        return ChatOllama(
            model="llama3.2",   # or "phi3"
            temperature=0
        )