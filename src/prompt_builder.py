from langchain_core.documents import Document


def format_retrieved_context(documents: list[Document]) -> str:
    """
    Convert retrieved LangChain Documents into a numbered context block.

    Each chunk gets a citation ID like [Source 1], [Source 2], etc.
    The page number is included when available.
    """
    if not documents:
        raise ValueError("No retrieved documents provided.")

    context_blocks = []

    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")

        context_blocks.append(
            f"[Source {i}]\n"
            f"Source file: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n" + ("-" * 80 + "\n\n").join(context_blocks)


def build_financial_rag_prompt(question: str, documents: list[Document]) -> str:
    """
    Build a grounded prompt for answering financial-report questions.

    The model must answer only from the retrieved context and cite sources.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    context = format_retrieved_context(documents)

    prompt = f"""
You are a Financial Report RAG Assistant.

Your job is to answer questions using only the provided excerpts from Apple's 10-K annual report.

Rules:
1. Use only the retrieved context below.
2. Do not use outside knowledge.
3. If the answer is not available in the context, say:
   "I could not find enough information in the retrieved report excerpts to answer this confidently."
4. Cite every factual claim using the format [Source X].
5. Give concise business-oriented answers.
6. When useful, explain the financial meaning, not just copy the text.

User question:
{question}

Retrieved context:
{context}

Answer:
""".strip()

    return prompt