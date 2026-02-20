# InfoBuddy

InfoBuddy is a **Streamlit-based conversational chatbot** (“InfoBot”) focused on answering **college-related queries concisely**. It uses a **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain + ChromaDB + HuggingFace embeddings**, and generates responses using **Groq-hosted Llama** models.

---

## Key Features

- **Chat UI in Streamlit** 
- **RAG-backed answers** by retrieving relevant knowledge chunks from a persisted **Chroma** vector store
- **Modern sentence-transformer embeddings**
- **LLM responses via Groq** using `llama-3.3-70b-versatile`
- Environment-based configuration via `.env` (Groq, LangChain, OpenAI, Gemini, HF )

---
<img width="928" height="731" alt="image" src="https://github.com/user-attachments/assets/38250108-b0be-4723-b54b-8ddef6fd4c7f" />

<img width="602" height="564" alt="image" src="https://github.com/user-attachments/assets/5c897a79-f4cf-4ffb-b584-36978f492714" />

<img width="829" height="807" alt="image" src="https://github.com/user-attachments/assets/14558113-8f63-4c57-9676-318b6695e675" />


## How It Works (Architecture)

At a high level, InfoBuddy follows this flow:

1. **User asks a question** in the Streamlit chat UI.
2. The question is sent to a LangChain **RetrievalQA** chain.
3. A **ChromaDB retriever** fetches the top `k=8` most relevant chunks from the local persisted vector store.
4. A **Groq Chat LLM** (Llama) generates an answer using the retrieved context (RAG).
5. The answer is displayed in the chat UI.

## Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **RAG framework:** LangChain
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace SentenceTransformers (`all-mpnet-base-v2`)
- **LLM Provider:** Groq (`langchain_groq`)
- **Env management:** python-dotenv

---

## Setup & Run Locally

### 1) Clone the repository

```bash
git clone https://github.com/peeyushdutt01/infobuddy.git
cd infobuddy
```

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in the project root (you can start from the existing template). At minimum, you should set:

- `GROQ_API_KEY` (required for `ChatGroq`)
- `LANGCHAIN_API_KEY` (optional, for LangChain tracing)
- `LANGCHAIN_PROJECT` (optional)

Example:

```bash
GROQ_API_KEY="your_groq_key_here"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="infobuddy"
OPENAI_API_KEY=""
GEMINI_API_KEY=""
HF_TOKEN=""
```

### 5) Ensure the Chroma vector store exists

The app loads Chroma from:

- `persist_directory = "datastore_db_new"`

That means a persisted vector store is expected to exist at runtime. If `datastore_db_new` is missing, you’ll need to generate/populate it (for example by running an ingestion script). If your repo already has an ingestion step but it isn’t documented yet, add it here.

### 6) Run the Streamlit app

```bash
streamlit run app.py
```

---

## Notes / Limitations

- The app’s QA chain is created with `return_source_documents=True`, but the UI currently displays only the answer text. If you want citations/sources shown, you can extend the UI to print `source_documents` metadata.
- There is a `truncate_history()` helper to manage token limits; ensure it’s applied if you plan to feed conversation history into prompts.
- `requirements.txt` includes `fastapi`, `uvicorn`, and `langserve`, suggesting future API-based serving is possible, even though the current entry point is Streamlit.
