# RAG System

A Retrieval-Augmented Generation (RAG) application built using:

* Python
* FAISS Vector Database
* Sentence Transformers
* LangChain
* Groq LLM API
* PDF/Text Document Processing

This project allows users to:

* Load PDF and text documents
* Split documents into chunks
* Generate embeddings
* Store vectors in FAISS
* Retrieve relevant chunks
* Generate AI-powered answers using Groq LLMs

---

# Features

* PDF document ingestion
* Text file support
* Recursive chunking
* Embedding generation with Sentence Transformers
* FAISS vector search
* Semantic retrieval
* RAG pipeline
* Source tracking
* Confidence scoring
* Groq LLM integration

---

# Project Structure

```bash
Rag/
│
├── data/
│   ├── pdf/
│   ├── text_file/
│
├── faiss_store/
│   ├── faiss.index
│   ├── metadata.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│
├── notebook/
│   ├── rag.ipynb
│
├── requirements.txt
├── README.md
└── .env
```

---

# Installation

## Clone Repository

```bash
git clone <your-github-repo-link>
cd Rag
```

---

## Create Virtual Environment

```bash
python -m venv ragenv
```

Activate environment:

### Linux/macOS

```bash
source ragenv/bin/activate
```

### Windows

```bash
ragenv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install \
langchain \
langchain-community \
langchain-groq \
sentence-transformers \
faiss-cpu \
pymupdf \
python-dotenv \
numpy
```

---

# Setup Groq API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Get API key from:

https://console.groq.com/keys

---

# Run Embedding Pipeline

```bash
python src/embedding.py
```

---

# Build Vector Store

```bash
python src/vector_store.py
```

---

# Run RAG Pipeline

```bash
python src/rag_pipeline.py
```

---

# Example Query

```python
result = rag_advance(
    "What is attention mechanism?",
    rag_retriever,
    llm,
    top_k=3,
    min_score=0.1,
    return_context=True
)

print(result['answer'])
```

---

# Technologies Used

* LangChain
* SentenceTransformers
* FAISS
* Groq
* Python
* NumPy
* PyMuPDF

---

# Future Improvements

* Streamlit UI
* FastAPI backend
* Hybrid search
* Reranking
* Chat memory
* Multi-document QA
* OCR support

---

# Author

Mudassir

AI / ML Enthusiast 🚀
