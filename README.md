# 🏢 HR Policy RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** system built with **LangChain**, **Jina AI Embeddings**, **FAISS**, **Groq LLM**, and **Streamlit** to provide fast, contextual Q&A over HR policy documents.

---

## 📌 Features

* **Data Ingestion:** Supports text documents, JSON, Excel, and structured policy files.
* **Document Chunking:** Configurable text splitting (Character and Recursive Splitters) for accurate chunk retrieval.
* **Vector Embeddings:** Uses **Jina AI Embeddings** to generate dense semantic vector representations.
* **Vector Database:** Fast similarity search with **FAISS**.
* **LLM Inference:** Ultra-low latency responses powered by **Groq** (`openai/gpt-oss-120b`).
* **Web UI / Notebook:** Interactive Jupyter Notebook and Streamlit interface.

---

## 🔒 Security Notice

> ⚠️ **IMPORTANT:** Never expose or commit real API keys or `.env` files to GitHub.
> 
> The `.gitignore` file is configured to ignore all `.env` files. Copy `.env.example` to `.env` locally and populate your secret keys.

---

## 📂 Project Structure

```text
hr-policy-rag-assistant/
├── Basic RAG experimental/         # Experimental RAG pipeline & research
│   ├── data/
│   │   └── hr_policies.txt         # Sample HR policy document dataset
│   ├── .env.example                # Environment variables template
│   ├── rag.ipynb                   # RAG pipeline demonstration notebook
│   ├── rag.py                      # Python script implementation
│   └── requirements.txt            # Dependencies for experimental setup
├── Production RAG/                 # (Upcoming) Production-ready modular RAG system
├── .env.example                    # Root environment variables template
├── .gitignore                       # Git exclusion rules (ignores .env, venvs, vector indexes)
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/hr-policy-rag-assistant.git
cd hr-policy-rag-assistant
```

### 2. Create & Activate Virtual Environment
```bash
python3 -m venv basicragenv
source basicragenv/bin/activate
```

### 3. Install Dependencies
```bash
cd "Basic RAG experimental"
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to create your local `.env` file:
```bash
cp .env.example .env
```

Open `.env` and set your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
JINA_API_KEY=your_jina_api_key_here
```

---

## 🛠️ Tech Stack & Pipeline Overview

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Framework for prompt engineering and retrieval chains |
| **LLM Provider** | [Groq](https://groq.com/) | High-speed inference using `openai/gpt-oss-120b` |
| **Embeddings** | [Jina AI](https://jina.ai/) | High-quality text embedding model |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) | Efficient vector similarity search index |
| **Frontend** | [Streamlit](https://streamlit.io/) | Lightweight web UI |

---

## 📜 License

Distributed under the MIT License.
