def format_retrieved_chunks(documents) -> str:
    """
    Convert retrieved LangChain Documents into a numbered context block.

    Each chunk gets a citation ID like [Source 1], [Source 2], etc.
    The page number is included when available.
    """
    if not documents:
        raise ValueError("No retrieved documents found")
    
    context_blocks = []
    
    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        
        context_blocks.append(
            f"Source {i}\n"
            f"source: {source}\n"
            f"page: {page}\n"
            f"{doc.page_content}\n"
        )
        
    return "\n\n" + "=" * 50 + "\n\n".join(context_blocks)

def build_financial_rag_prompt(question, documents) -> str:
    """
    Build a prompt for financial question answering using retrieved documents.

    The prompt includes instructions, the question, and the formatted context from retrieved chunks.
    """
    
    context = format_retrieved_chunks(documents)
    
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