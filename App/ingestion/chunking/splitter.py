import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

#splitting and chunking
def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # customize order/list as needed
            length_function=len,
        )
        chunks = splitter.split_text(text)
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"Generated {len(valid_chunks)} chunks")
        return valid_chunks