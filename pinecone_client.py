"""
utils/pinecone_client.py
Handles all Pinecone vector database interactions.
"""

import uuid
from typing import List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from utils.embeddings import get_embeddings_batch


# Dimension for text-embedding-3-small
EMBEDDING_DIM = 1536
BATCH_SIZE    = 100   # Pinecone upsert batch size


def init_pinecone(api_key: str, index_name: str):
    """
    Initialise the Pinecone client and return an Index object.
    Creates the index if it does not already exist.

    Args:
        api_key:    Your Pinecone API key.
        index_name: The name of the Pinecone index to use.

    Returns:
        A Pinecone Index object.
    """
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name      = index_name,
            dimension = EMBEDDING_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)


def upsert_chunks(index, chunks: List[str], pdf_name: str, openai_api_key: str) -> None:
    """
    Embed all chunks and upsert them into Pinecone with metadata.

    Args:
        index:          A Pinecone Index object.
        chunks:         List of text chunks from the PDF.
        pdf_name:       Original PDF filename (stored as metadata).
        openai_api_key: OpenAI key used to generate embeddings.
    """
    if not chunks:
        return

    # Generate embeddings in one batch call
    embeddings = get_embeddings_batch(chunks, openai_api_key)

    # Build Pinecone upsert vectors
    vectors: List[Dict[str, Any]] = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id":       str(uuid.uuid4()),
            "values":   embedding,
            "metadata": {
                "pdf_name": pdf_name,
                "chunk_id": i,
                "text":     chunk[:1000],   # Pinecone metadata value limit
            },
        })

    # Upsert in batches to avoid payload limits
    for start in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[start : start + BATCH_SIZE]
        index.upsert(vectors=batch)


def query_index(index, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the Pinecone index and return the top-k matches.

    Args:
        index:        A Pinecone Index object.
        query_vector: Embedding of the user query.
        top_k:        Number of results to return.

    Returns:
        List of match dictionaries containing id, score, and metadata.
    """
    response = index.query(
        vector          = query_vector,
        top_k           = top_k,
        include_metadata= True,
    )
    return response.get("matches", [])
