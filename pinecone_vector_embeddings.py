import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone
import hashlib
from google import genai
import random

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-chatbot-index"

client = genai.Client(api_key=GEMINI_API_KEY)

folder_path = "References for RAG"
all_text = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        full_file_path = os.path.join(root, file)
        
        if file.endswith('.pdf'):
            print(f"Found PDF: {full_file_path}")
            file_path = os.path.join(root, file)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])
            all_text.append(full_text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)

batch_size = 100

vectors = []
for text in all_text:
    chunks = text_splitter.split_text(text)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch
            )

            embeddings = result.embeddings
    
            for index, text in enumerate(batch):
                hash_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
                vectors.append(
                    {
                        "id": hash_id,
                        "values": embeddings[index].values
                    }
                )
            print(f"Processed batch starting from index {i}")
        except genai.errors.ClientError as e:
            print(f"Error processing batch starting from index {i}: {e}")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host="https://rag-vector-embedding-test-nygbzf5.svc.aped-4627-b74a.pinecone.io")

index.upsert(
    vectors,
    "__default__"
)

print(f"Records inserted into pinecone!")
