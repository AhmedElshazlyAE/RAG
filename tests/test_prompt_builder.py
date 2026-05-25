import pytest
from langchain_core.documents import Document

from src.prompt_builder import (
    format_retrieved_context,
    build_financial_rag_prompt,
)


def test_format_retrieved_context_includes_sources_and_metadata():
    documents = [
        Document(
            page_content="Apple reported net sales from iPhone and Services.",
            metadata={"page": 10, "source": "apple_10k.pdf"},
        ),
        Document(
            page_content="Services revenue includes advertising, AppleCare, and cloud services.",
            metadata={"page": 12, "source": "apple_10k.pdf"},
        ),
    ]

    context = format_retrieved_context(documents)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Page: 10" in context
    assert "Page: 12" in context
    assert "apple_10k.pdf" in context
    assert "iPhone" in context
    assert "Services" in context


def test_build_financial_rag_prompt_contains_question_context_and_rules():
    question = "What were Apple's main sources of revenue?"

    documents = [
        Document(
            page_content="Apple net sales were categorized by iPhone, Mac, iPad, Services, and Wearables.",
            metadata={"page": 20, "source": "apple_10k.pdf"},
        )
    ]

    prompt = build_financial_rag_prompt(question, documents)

    assert "Financial Report RAG Assistant" in prompt
    assert question in prompt
    assert "[Source 1]" in prompt
    assert "Use only the retrieved context" in prompt
    assert "Do not use outside knowledge" in prompt
    assert "iPhone, Mac, iPad, Services" in prompt


def test_build_financial_rag_prompt_raises_error_for_empty_question():
    documents = [
        Document(
            page_content="Some financial text.",
            metadata={"page": 1, "source": "apple_10k.pdf"},
        )
    ]

    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_financial_rag_prompt("", documents)


def test_format_retrieved_context_raises_error_for_empty_documents():
    with pytest.raises(ValueError, match="No retrieved documents provided"):
        format_retrieved_context([])