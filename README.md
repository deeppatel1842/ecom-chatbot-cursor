# E-commerce RAG Chatbot

A modern, production-ready e-commerce chatbot using Retrieval-Augmented Generation (RAG), FAISS vector search, and a FastAPI + HTML/JS frontend. Built for robust, scalable, and accurate customer support and product Q&A.

---

## 🚀 Features
- **RAG Pipeline:** Combines semantic retrieval (FAISS + sentence-transformers) with LLM answer generation.
- **Multi-Agent System:** FAQ, Product, Policy, and Fallback agents for modular, extensible logic.
- **Modern Frontend:** Clean, responsive chat UI (FastAPI + HTML/JS).
- **Local Data:** All retrieval is from your local datasets (no external scraping).
- **OpenAI LLM Integration:** Secure API key management via `.env`.
- **Edge Case Handling:** Robust validation, error handling, and logging.

---

## 🗂️ Project Structure
```
ecom-chatbot/
├── data/                  # Raw and processed data
├── processed_data/        # Preprocessed CSV/JSON for fast loading
├── src/
│   ├── agents/            # FAQ, Product, Policy, Fallback agents
│   ├── rag_pipeline.py    # RAG pipeline logic
│   ├── langgraph_workflow.py # Agent routing/workflow
│   ├── data_preprocessing.py # Data cleaning and validation
│   └── app.py             # FastAPI backend
├── static/
│   └── index.html         # Chatbot frontend UI
├── requirements.txt       # Python dependencies
├── .env                   # API keys and secrets (not committed)
├── README.md              # Project documentation
└── ...
```

---

## ⚡ Quickstart
1. **Clone the repo & install dependencies:**
   ```bash
   git clone <repo-url>
   cd ecom-chatbot
   pip install -r requirements.txt
   ```
2. **Set your OpenAI API key:**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=sk-...
     ```
3. **Preprocess the data:**
   ```bash
   python src/data_preprocessing.py
   ```
4. **Run the chatbot web app:**
   ```bash
   python src/app.py
   ```
5. **Open your browser:**
   - Go to [http://localhost:8000](http://localhost:8000)

---

## 🧠 Architecture
- **FAISS Vector Search:** Fast, in-memory semantic search for FAQ, product, and policy retrieval.
- **Sentence Transformers:** Embedding model for all queries and documents.
- **RAG Pipeline:** Retrieved context + user query → LLM (OpenAI) → Answer.
- **LangGraph Workflow:** Modular agent routing and state management.
- **FastAPI Backend:** Handles chat requests and serves the frontend.
- **HTML/JS Frontend:** Clean, responsive chat interface.

---

## 💡 Improvements & Extensions
- Hybrid search (keyword + semantic)
- LLM fine-tuning on your own data
- Multi-turn conversation memory
- Show sources/citations in UI
- Analytics dashboard
- Dockerization & cloud deployment
- Integration with e-commerce platforms (Shopify, WooCommerce, etc.)

---

## 📝 License
MIT License

---

## 🙋‍♂️ Support & Contributions
- Open issues or pull requests for bugs, improvements, or questions.
- For help, contact the project maintainer. # ecom-chatbot-cursor
