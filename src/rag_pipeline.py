from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompt_builder import build_financial_rag_prompt
from src.retriever import retrieve_relevant_chunks

def get_gemini_model(
    model_name = "gemini-2.5-flash",
    temperature = 0.0,
):
    """
    Create a Gemini chat model.

    temperature=0.0 is used because financial QA should be factual,
    stable, and grounded.
    """
    load_dotenv()
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        )

def answer_question(
    question,
    vector_store,
    k,
    model_name = "gemini-2.5-flash") -> dict:
    """
    Full RAG pipeline:

    1. Retrieve relevant chunks from Chroma.
    2. Build a grounded financial prompt.
    3. Send the prompt to Gemini.
    4. Return the answer and retrieved source documents.
    """
    # Step 1: Retrieve relevant chunks
    documents = retrieve_relevant_chunks(
        vector_store=vector_store,
        question=question,
        k=k
    )
    
    # Step 2: Build the prompt
    prompt = build_financial_rag_prompt(
        question=question,
        documents=documents
    )
    
    # Step 3: Get the model and generate the answer
    model = get_gemini_model(model_name=model_name)
    answer = model.invoke(prompt)

    # Step 4: Return the answer and retrieved source documents
    return {
        "question": question, 
        "answer": answer,
        "sources": documents,
        "prompt": prompt
    }