from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create a local HuggingFace embedding model.

    This model converts text chunks into vectors so Chroma can perform
    semantic search.
    """
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )