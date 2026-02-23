from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from HuggingFaceEmbeddingsService import get_ai_model
from AiApiKey import get_api_key

def get_contex_query(question, vector_db, k=3):
    docs_with_score = vector_db.similarity_search_with_score(question, k)
    relevant_docs = sorted(docs_with_score, key=lambda x: x[1])
    retrived_texts = []
    seen_content = set()

    for doc, score in relevant_docs:
        content =doc.page_content.strip()
        if content not in seen_content:
            retrived_texts.append(content)
            seen_content.add(content)
    context_block = "\n\n---\n\n".join(retrived_texts)
    return context_block

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
    context = get_contex_query(query)

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

