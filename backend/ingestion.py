import fitz
from typing import List, Dict

def extract_text_from_pdf(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def chunk_text(pages: List[Dict], chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page": page_number
            })

            start += chunk_size - overlap

    return chunks