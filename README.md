# Financial Report RAG Assistant

A business-oriented Retrieval-Augmented Generation (RAG) project that answers questions from Apple's 10-K annual report using **Chroma**, **local SentenceTransformer embeddings**, and **Gemini**.

This is not a generic PDF chatbot. The goal is to build a financial document intelligence assistant that retrieves relevant excerpts from an annual report and generates grounded, citation-backed answers.

---

## Project Overview

Financial reports are long, dense, and difficult to analyze manually. This project uses RAG to make Apple's 10-K annual report easier to query through natural-language questions.

The assistant follows a manual RAG pipeline:

```text
Apple 10-K PDF
      ↓
Load PDF pages
      ↓
Split report text into chunks
      ↓
Generate local embeddings
      ↓
Store vectors in Chroma
      ↓
Retrieve top-k relevant chunks
      ↓
Build a grounded financial QA prompt
      ↓
Generate answer with Gemini
      ↓
Return answer with citations
```

The system is designed to answer questions only from the retrieved report excerpts. If the retrieved context does not contain enough information, the assistant should say that it cannot answer confidently instead of hallucinating.

---

## Why I Built This

I built this project to understand how RAG works internally instead of relying only on high-level ready-made chains.

As a learning step, I first built a RAG pipeline from scratch in a notebook:

```text
notebooks/rag_from_scratch.ipynb
```

That notebook helped me understand the full flow manually:

- loading and inspecting PDF text
- splitting a long financial report into chunks
- creating embeddings
- storing vectors
- retrieving relevant chunks
- manually building the prompt
- connecting retrieved context to Gemini
- checking whether the answer is grounded in the retrieved sources

After validating the workflow in the notebook, I modularized the project into reusable Python files inside `src/`.

---

## Example Questions

The assistant can answer questions such as:

```text
What were Apple's main sources of revenue?
```

```text
What does Apple say about Services revenue?
```

```text
What risks does Apple mention in the annual report?
```

```text
How does Apple describe its business?
```

```text
What are Apple's major product categories?
```

---

## Example RAG Outputs

### Example 1: Revenue Sources

**Question**

```text
What were Apple's main sources of revenue?
```

**Example answer**

```text
Apple's main sources of revenue are its product and service categories, including iPhone, Mac, iPad, Wearables, Home and Accessories, and Services. The retrieved report excerpts show that Apple reports net sales across these categories, with iPhone and Services being major contributors. [Source 1] [Source 2]
```

**Retrieved source format**

```text
[Source 1]
Source file: data/raw/apple_10k.pdf
Page: <page_number>
Content: <retrieved Apple 10-K excerpt>

[Source 2]
Source file: data/raw/apple_10k.pdf
Page: <page_number>
Content: <retrieved Apple 10-K excerpt>
```

---

### Example 2: Services Revenue

**Question**

```text
What does Apple say about Services revenue?
```

**Example answer**

```text
Apple describes Services revenue as revenue generated from offerings such as advertising, AppleCare, cloud services, digital content, payment services, and other services. This category is separate from Apple's hardware product categories. [Source 1]
```

---

### Example 3: Out-of-Scope Question

**Question**

```text
What is Tim Cook's favorite food?
```

**Expected answer**

```text
I could not find enough information in the retrieved report excerpts to answer this confidently.
```

This behavior is important because a financial RAG assistant should avoid answering from outside knowledge when the source document does not support the answer.

---

## Tech Stack

- Python
- Jupyter Notebook
- LangChain
- Chroma vector database
- HuggingFace / SentenceTransformer embeddings
- Gemini via `langchain-google-genai`
- PyPDF
- python-dotenv
- Pytest

---

## Project Structure

```text
RAG/
│
├── data/
│   └── raw/
│       └── apple_10k.pdf
│
├── notebooks/
│   ├── rag_from_scratch.ipynb
│   └── basic_retrieval.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   └── rag_pipeline.py
│
├── tests/
│   ├── test_chunker.py
│   └── test_prompt_builder.py
│
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

> Note: `chroma_db/` is intentionally not included in the project structure because it is generated locally and should not be committed to GitHub.

---

## Main Components

### `config.py`

Stores central project settings such as file paths, model names, chunk size, chunk overlap, Chroma collection name, and retrieval settings.

### `pdf_loader.py`

Loads Apple's 10-K PDF using `PyPDFLoader` and returns LangChain `Document` objects.

### `chunker.py`

Splits long page-level documents into smaller chunks using `RecursiveCharacterTextSplitter`.

### `embeddings.py`

Creates local HuggingFace embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This keeps the embedding step local instead of sending document chunks to an external embedding API.

### `vector_store.py`

Builds or loads a Chroma vector store using embedded document chunks.

### `retriever.py`

Retrieves the top-k most relevant chunks for a user question using Chroma.

### `prompt_builder.py`

Builds a grounded financial QA prompt that instructs Gemini to:

- use only the retrieved context
- avoid outside knowledge
- cite factual claims using source markers
- say when the answer is not available in the retrieved excerpts

### `rag_pipeline.py`

Combines retrieval, prompt building, and Gemini answer generation into a single pipeline.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/AhmedElshazlyAE/RAG.git
cd RAG
```

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

The real `.env` file should not be committed to GitHub.

---

## Requirements

Recommended `requirements.txt`:

```text
langchain
langchain-community
langchain-text-splitters
langchain-chroma
langchain-huggingface
langchain-google-genai
chromadb
sentence-transformers
pypdf
python-dotenv
streamlit
tqdm
pytest
```

Do not include FAISS in this project because the vector store used here is Chroma.

---

## Usage

Start with the from-scratch learning notebook:

```text
notebooks/rag_from_scratch.ipynb
```

Then use the retrieval notebook:

```text
notebooks/basic_retrieval.ipynb
```

The notebook workflow demonstrates:

1. loading Apple's 10-K PDF
2. chunking the report text
3. creating local embeddings
4. storing vectors in Chroma
5. retrieving relevant chunks
6. building a grounded prompt
7. generating a Gemini answer with citations

---

## Example Pipeline Usage

```python
from src.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_K,
)

from src.pdf_loader import load_pdf
from src.chunker import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store
from src.rag_pipeline import answer_question

documents = load_pdf(PDF_PATH)

chunks = chunk_documents(
    documents=documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)

vector_store = build_vector_store(
    chunks=chunks,
    embedding_model=embedding_model,
    persist_directory=CHROMA_DIR,
    collection_name=CHROMA_COLLECTION_NAME,
)

result = answer_question(
    question="What were Apple's main sources of revenue?",
    vector_store=vector_store,
    k=RETRIEVAL_K,
)

print(result["answer"])
```

---

## Testing

Run the tests:

```bash
python -m pytest
```

Current tests cover:

- chunk creation
- metadata preservation during chunking
- prompt formatting
- source/citation formatting
- validation errors for empty inputs

Example expected test result:

```text
6 passed
```

---

## Current Status

Implemented:

- PDF loading
- text chunking
- local embedding generation
- Chroma vector storage
- semantic retrieval
- grounded prompt construction
- Gemini answer generation
- citation-style answer formatting
- from-scratch RAG learning notebook
- modular Python source code
- unit tests for core components

Planned improvements:

- add clearer page-number citation display
- add retrieval quality evaluation questions
- add a Streamlit interface
- add example screenshots
- clean notebook outputs
- improve error handling
- add a small evaluation set for financial questions

---

## What I Learned

Through this project, I practiced:

- building a RAG pipeline manually
- avoiding deprecated high-level chains
- using Chroma as a vector database
- creating local embeddings with SentenceTransformers
- connecting Gemini to a custom retrieval pipeline
- designing grounded prompts for financial question answering
- organizing notebook experiments into reusable Python modules
- writing unit tests for important pipeline components
- separating generated files from source code

---

## Important Notes

The local Chroma database is generated when the pipeline runs. It should not be committed to GitHub.

The `.env` file is ignored because it contains the Gemini API key.

The from-scratch notebook is intentionally kept because it documents the learning process before the project was modularized.

---
