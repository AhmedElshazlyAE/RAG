from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
    Split page-level documents into smaller chunks for retrieval.
    Keeps metadata such as source and page number.
"""
def chunk_documents(documents, chunk_size, chunk_overlap):
    if not documents:
        raise "No documents found for chunking"
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    
    if not chunks:
        raise "Chunk produced no documents"
    
    return chunks