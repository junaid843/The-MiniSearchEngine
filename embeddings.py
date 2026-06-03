"""
utils/embeddings.py
Generates text embeddings using the OpenAI API.
"""

from typing import List
import openai


def get_embedding(text: str, api_key: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate a single embedding vector for the given text.

    Args:
        text:    The input string to embed.
        api_key: Your OpenAI API key.
        model:   The embedding model to use.

    Returns:
        A list of floats representing the embedding vector.
    """
    client   = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Generate embeddings for a list of texts in a single API call (more efficient).

    Args:
        texts:   List of strings to embed.
        api_key: Your OpenAI API key.
        model:   The embedding model to use.

    Returns:
        A list of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    client   = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(input=texts, model=model)
    # Results are returned in the same order as the input
    return [item.embedding for item in response.data]
