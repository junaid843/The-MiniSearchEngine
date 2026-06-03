import streamlit as st
import os
import io
import uuid
from typing import List, Dict, Any

import PyPDF2
import openai
from pinecone import Pinecone, ServerlessSpec

# ─────────────────────────────────────────────
#  PDF UTILITIES
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t.strip())
    return "\n\n".join(parts)


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []
    chunks, start = [], 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks


# ─────────────────────────────────────────────
#  EMBEDDING UTILITIES
# ─────────────────────────────────────────────

def get_embedding(text: str, api_key: str) -> List[float]:
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str], api_key: str) -> List[List[float]]:
    if not texts:
        return []
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [item.embedding for item in response.data]


# ─────────────────────────────────────────────
#  PINECONE UTILITIES
# ─────────────────────────────────────────────

EMBEDDING_DIM = 1536
BATCH_SIZE = 100


def init_pinecone(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def upsert_chunks(index, chunks: List[str], pdf_name: str, openai_api_key: str) -> None:
    if not chunks:
        return
    embeddings = get_embeddings_batch(chunks, openai_api_key)
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "pdf_name": pdf_name,
                "chunk_id": i,
                "text": chunk[:1000],
            },
        })
    for start in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[start: start + BATCH_SIZE])


def query_index(index, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return response.get("matches", [])


# ─────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────

st.set_page_config(page_title="Mini Search Engine", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.result-card {
    background: #f8f9fa;
    border-left: 4px solid #0066ff;
    padding: 1rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}
.score-badge {
    background: #0066ff; color: white;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.8rem; font-family: 'Space Mono', monospace;
}
.pdf-name { font-weight: 600; font-size: 1rem; margin-bottom: 0.4rem; }
.chunk-text { font-size: 0.9rem; color: #444; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Mini Search Engine")
st.caption("Semantic PDF search powered by Pinecone + OpenAI Embeddings")

with st.sidebar:
    st.header("⚙️ Configuration")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password",
                                     value=os.getenv("PINECONE_API_KEY", ""))
    pinecone_index   = st.text_input("Pinecone Index Name",
                                     value=os.getenv("PINECONE_INDEX", "mini-search"))
    openai_api_key   = st.text_input("OpenAI API Key", type="password",
                                     value=os.getenv("OPENAI_API_KEY", ""))
    top_k        = st.slider("Top-K Results", 1, 10, 5)
    chunk_size   = st.slider("Chunk Size (chars)", 200, 1000, 500)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 200, 50)

st.header("📄 Upload PDF Documents")
uploaded_files = st.file_uploader("Upload at least 5 PDF files", type=["pdf"],
                                   accept_multiple_files=True)
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")

if st.button("📥 Process & Index PDFs", disabled=not uploaded_files):
    if not pinecone_api_key or not openai_api_key:
        st.error("Please provide both API keys in the sidebar.")
    else:
        index    = init_pinecone(pinecone_api_key, pinecone_index)
        progress = st.progress(0, text="Initialising…")
        total    = len(uploaded_files)
        for i, pdf_file in enumerate(uploaded_files):
            progress.progress(i / total, text=f"Processing {pdf_file.name}…")
            text   = extract_text_from_pdf(pdf_file)
            chunks = split_into_chunks(text, chunk_size, chunk_overlap)
            upsert_chunks(index, chunks, pdf_file.name, openai_api_key)
        progress.progress(1.0, text="Done!")
        st.success("✅ All PDFs indexed successfully.")

st.header("🔎 Search")
query = st.text_input("Enter your search query", placeholder="e.g. What is machine learning?")

if st.button("Search", disabled=not query):
    if not pinecone_api_key or not openai_api_key:
        st.error("Please provide both API keys in the sidebar.")
    else:
        with st.spinner("Searching…"):
            index     = init_pinecone(pinecone_api_key, pinecone_index)
            query_vec = get_embedding(query, openai_api_key)
            results   = query_index(index, query_vec, top_k)
        st.subheader(f"Top {top_k} Results")
        if not results:
            st.info("No results found.")
        else:
            for match in results:
                meta  = match.get("metadata", {})
                score = round(match.get("score", 0), 4)
                name  = meta.get("pdf_name", "Unknown")
                text  = meta.get("text", "")
                st.markdown(f"""
                <div class="result-card">
                    <div class="pdf-name">📄 {name} &nbsp; <span class="score-badge">score: {score}</span></div>
                    <div class="chunk-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)
    
