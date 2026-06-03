# 🔍 Mini Search Engine

A semantic PDF search application built with **Streamlit**, **Pinecone**, and **OpenAI Embeddings**.

---

## Features

- Upload multiple PDF documents
- Automatic text extraction and chunking
- Embeddings generated via OpenAI `text-embedding-3-small`
- Vector storage and retrieval using Pinecone
- Ranked results with similarity score, PDF name, and extracted text

---

## Project Structure

```
mini_search_engine/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
└── utils/
    ├── __init__.py
    ├── pdf_processor.py    # PDF text extraction + chunking
    ├── embeddings.py       # OpenAI embedding generation
    └── pinecone_client.py  # Pinecone init, upsert, and query
```

---

## Setup

### 1. Clone / Download

```bash
git clone <your-repo-url>
cd mini_search_engine
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=mini-search
OPENAI_API_KEY=your_openai_api_key_here
```

> You can also enter API keys directly in the sidebar of the running app.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload PDFs** — Drag and drop at least 5 PDF files.
2. **Process & Index** — Click the button to extract, chunk, embed, and store.
3. **Search** — Type a natural language query and click **Search**.
4. **Results** — View the top-K matching chunks with PDF name and similarity score.

---

## API Keys

| Service  | Where to get                                      |
|----------|---------------------------------------------------|
| OpenAI   | https://platform.openai.com/api-keys             |
| Pinecone | https://app.pinecone.io → API Keys               |

---

## Deployment (Streamlit Cloud)

1. Push this project to a **GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Add your secrets in **Settings → Secrets**:
   ```toml
   PINECONE_API_KEY = "..."
   PINECONE_INDEX   = "mini-search"
   OPENAI_API_KEY   = "..."
   ```
4. Click **Deploy**.

---

## Learning Outcomes

- Understand text embeddings and vector similarity
- Use Pinecone as a vector database
- Implement semantic (meaning-based) search
- Build and deploy a retrieval system
