from contextlib import asynccontextmanager
from fastapi import FastAPI
from services import vector_manager
from upload import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_manager.load_database()
    yield

app = FastAPI(title="RAG service", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)