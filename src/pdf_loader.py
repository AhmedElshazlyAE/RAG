from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


"""
    Load a PDF file and return a list of LangChain Document objects.
    Each page becomes one Document.
"""
def load_pdf(pdf_path: str):
    path = Path(pdf_path)
    
    if not path.exists:
        raise "File Not Found"
    
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    if not documents:
        raise "No Pages loaded"
    
    return documents