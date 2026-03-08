import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_pdf_index"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI(title="AI PDF RAG Service")

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = None

def load_vector_db():
    global vector_db
    if os.path.exists(FAISS_INDEX_PATH):
        vector_db = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )

load_vector_db()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_db
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Моля, качете PDF файл.")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    try:
        loader = PyPDFLoader(filepath)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        if vector_db is None:
            vector_db = FAISS.from_documents(chunks, embeddings_model)
        else:
            vector_db.add_documents(chunks)

        vector_db.save_local(FAISS_INDEX_PATH)
        
        return {"message": f"Файлът {file.filename} е индексиран успешно!", "chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if vector_db is None:
        raise HTTPException(status_code=400, detail="Базата данни е празна. Първо качете PDF.")

    docs_with_score = vector_db.similarity_search_with_score(request.question, k=request.top_k)
    
    retrieved_texts = []
    seen_content = set()

    for doc, score in docs_with_score:
        content = doc.page_content.strip()
        if content not in seen_content:
            retrieved_texts.append({
                "text": content,
                "metadata": doc.metadata,
                "score": float(score)
            })
            seen_content.add(content)

    context_block = "\n\n---\n\n".join([item["text"] for item in retrieved_texts])

    return {
        "question": request.question,
        "context": context_block,
        "sources": retrieved_texts
    }