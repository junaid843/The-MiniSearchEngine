"""
utils/pdf_processor.py
Handles PDF text extraction and chunking.
"""

import io
from typing import List
import PyPDF2


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from an uploaded PDF file object."""
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split a long string into overlapping chunks of roughly `chunk_size` characters.

    Args:
        text:       The full document text.
        chunk_size: Target character length per chunk.
        overlap:    Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        # Try to break at a sentence boundary
        if end < length:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end - overlap > start else end

    return chunks
