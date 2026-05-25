import pytest
from langchain_core.documents import Document

from src.chunker import chunk_documents


def test_chunk_documents_returns_chunks():
    documents = [
        Document(
            page_content="Apple sells iPhone, Mac, iPad, Services, and Wearables. " * 50,
            metadata={"page": 1, "source": "apple_10k.pdf"},
        )
    ]

    chunks = chunk_documents(
        documents=documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)
    assert all(chunk.metadata["source"] == "apple_10k.pdf" for chunk in chunks)


def test_chunk_documents_raises_error_for_empty_input():
    with pytest.raises(ValueError, match="No documents provided"):
        chunk_documents([])