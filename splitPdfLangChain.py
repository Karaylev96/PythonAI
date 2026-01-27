from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from HuggingFaceEmbeddingsService import get_ai_model
from AiApiKey import get_api_key

def split_pdf_lang_chain(path):
    loader = PyPDFLoader(path)
    data = loader.load()

    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_split.split_documents(data)

    texts = [chunk.page_content for chunk in chunks]

    local_embeddings = get_ai_model()
    vectors = local_embeddings.embed_documents(texts)

    vector_db = FAISS.from_documents(chunks, local_embeddings)

    vector_db.save_local("faiss_pdf")

    new_db = FAISS.load_local("faiss_pdf", local_embeddings, allow_dangerous_deserialization=True)

    query = "Дайте петте най-важни думи от документа?"
    docs = new_db.similarity_search(query, k=3)

    data = []
    for i, chunk in enumerate(chunks):
        item = {
            "vector": vectors[i],
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        data.append(item)
    return data


api_key = get_api_key()
pdf_path = "/Users/georgikarajlev/Downloads/PrezentaciqZashtitaIvaVukova.pdf"
all_chuncks = split_pdf_lang_chain(pdf_path)

print(f"Chunks count: {len(all_chuncks)}")
print(f"Chunks count: {len(all_chuncks[0]['vector'])}")

