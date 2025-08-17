import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone
import hashlib

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-chatbot-index"
    
loader = TextLoader("data.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

data_to_upsert = []
for doc in docs:
    hash_id = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
    data_to_upsert.append(
        {
            "_id": hash_id,
            "text": doc.page_content
        }
    )

print(f"########Loaded {len(data_to_upsert)} documents and split into {len(data_to_upsert)} chunks.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host="https://rag-chatbot-index-2-nygbzf5.svc.aped-4627-b74a.pinecone.io")

index.upsert_records(
    "__default__",
    data_to_upsert  
)

print(f"Records inserted into pinecone!")