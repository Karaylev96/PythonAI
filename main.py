import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from services import vector_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_manager.load_database()
    yield

app = FastAPI(title="RAG service", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files.")
    
    temp_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        num_chunks = await vector_manager.process_pdf(temp_path)
        return {"status": "success", "message": f"Chunk numbers {num_chunks}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    results = vector_manager.search(request.question, k=request.top_k)
    
    if not results:
        raise HTTPException(status_code=404, detail="DB is empty.")

    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })

    return {
        "question": request.question,
        "context": "\n\n".join([r["content"] for r in formatted_results]),
        "sources": formatted_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)