import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key.startswith("sk-"):
    print("api_key Work")
else:
    print("api_key not Work")

def split_pdf_lang_chain(path):
    loader = PyPDFLoader(path)
    data = loader.load()

    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_split.split_documents(data)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings_model.embed_documents(texts)
    data = []
    for i, chunk in enumerate(chunks):
        item = {
            "vector": vectors[i],
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        data.append(item)
    return data

pdf_path = "C:/Users/Public/test_chunks.pdf"
all_chuncks = split_pdf_lang_chain(pdf_path)

print(f"Chunks count: {len(all_chuncks)}")
print(f"Chunks count: {len(all_chuncks[0]['vector'])}")

