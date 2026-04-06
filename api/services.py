import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.index_path = "faiss_pdf_index"
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = None
        self._initialized = True

    def load_database(self):
        if os.path.exists(self.index_path):
            self.vector_db = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("DB is loaded")

    async def process_pdf(self, filepath: str):
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_db.add_documents(chunks)
        
        self.vector_db.save_local(self.index_path)
        return len(chunks)

    def search(self, query: str, k: int = 3):
        if not self.vector_db:
            return None
        return self.vector_db.similarity_search_with_score(query, k=k)

vector_manager = VectorStoreManager()