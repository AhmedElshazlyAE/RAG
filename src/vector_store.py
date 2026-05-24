from pathlib import Path

from langchain_chroma import Chroma

def build_vector_store(
    chunks,
    embedding_model,
    persist_directory: str | Path,
    collection_name: str,
) -> Chroma:
    """
    Build a Chroma vector store from document chunks.

    This embeds the chunks and stores them locally in persist_directory.
    """
    if not chunks:
        raise "No chunks provided"
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    return vector_store

def loader_vector_store(
    embedding_model,
    persist_directory: str | Path,
    collection_name: str
) -> Chroma:
    """
    Load an existing Chroma vector store from disk.
    """
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    return vector_store