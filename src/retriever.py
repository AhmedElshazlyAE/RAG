def retrieve_relevant_chunks(
    vector_store,
    question,
    k: int = 4
):
    """
    Retrieve the top-k chunks most relevant to the user's question.
    """
    if not question:
        raise "Question cannot be empty"
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k":k}
    )
    
    documents = retriever.invoke(question)
    
    if not documents:
        raise "No relevant documents found"
    
    return documents