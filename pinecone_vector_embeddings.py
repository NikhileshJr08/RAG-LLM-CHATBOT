import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone
import hashlib
from google import genai
from google.genai import types
import random
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

folder_path = "References for RAG"
all_text = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        full_file_path = os.path.join(root, file)
        if file.endswith('.pdf'):
            # print(f"Found PDF: {full_file_path}")
            file_path = os.path.join(root, file)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])
            all_text.append(full_text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)

embed_query = []
for text in all_text:
    chunks = text_splitter.split_text(text)
    for index, text in enumerate(chunks):
        hash_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
        embed_query.append(
            {
                "_id": hash_id,
                "text": text
            }
        )
        if len(embed_query) == 5:
            index = pc.Index(host="https://rag-text-embed-test-nygbzf5.svc.aped-4627-b74a.pinecone.io")
            index.upsert_records(
                "__default__",
                embed_query
            )
            embed_query = []
            print(f"Records inserted into pinecone!")

# vectors = []
# for text in all_text:
#     chunks = text_splitter.split_text(text)
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         try:
#             time.sleep(2.5)
#             embeddings = client.models.embed_content(
#                 model="gemini-embedding-001",
#                 contents=batch,
#                 config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")).embeddings
    
#             for index, text in enumerate(batch):
#                 hash_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
#                 vectors.append(
#                     {
#                         "_id": hash_id,
#                         "chunk_text": embeddings[index].values,
#                         "metadata": {"text": text}
#                     }
#                 )
#             print(f"Processed batch starting from index {i}")
#         except genai.errors.ClientError as e:
#             print(f"Error processing batch starting from index {i}: {e}")

