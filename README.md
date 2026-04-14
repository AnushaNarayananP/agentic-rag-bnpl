# 🤖 AI-Powered BNPL Customer Support Assistant

This project is an **Agentic RAG (Retrieval-Augmented Generation) system** designed for a **fintech Buy Now Pay Later (BNPL)** use case.

## 🚀 Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 🤖 Agent-based reasoning using LangGraph
- 🧠 Local LLM (Ollama – no API cost)
- 📄 Supports PDF + Web data sources
- 💬 Streamlit UI for non-technical users

## 💡 Use Case

Simulates a **BNPL customer support assistant** capable of answering:

- How installment payments work  
- Late payment consequences  
- Refund handling  
- Credit checks and limits  

## 🏗️ Tech Stack

- LangChain
- LangGraph
- Ollama (Local LLM)
- HuggingFace Embeddings
- FAISS Vector Store
- Streamlit

## 📂 Data Sources

- BNPL policy PDF
- Customer support pages from:
  - Sezzle
  - Klarna
  - Afterpay

## ▶️ Run Locally

```bash
pip install -r requirements.txt
ollama pull llama3.2
streamlit run streamlit_app.py
