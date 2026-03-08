import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_pdf_index"

def get_context_query(question, vector_db, k=3):
    docs_with_score = vector_db.similarity_search_with_score(question, k=k)
    retrieved_texts = []
    seen_content = set()

    for doc, score in docs_with_score:
        content = doc.page_content.strip()
        if content not in seen_content:
            retrieved_texts.append(content)
            seen_content.add(content)
            
    return "\n\n---\n\n".join(retrieved_texts)

def process_pdf_locally(path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    loader = PyPDFLoader(path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(data)

    vector_db = FAISS.from_documents(chunks, embeddings)

    vector_db.save_local(FAISS_INDEX_PATH)
    
    query = "Дайте петте най-важни думи от документа?"
    context = get_context_query(query, vector_db)
    
    return chunks, context

if __name__ == "__main__":
    pdf_path = ".pdf" 
    if os.path.exists(pdf_path):
        chunks, context = process_pdf_locally(pdf_path)
