import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize the Large Language Model (LLM) using Gemini
llm = ChatGoogleGenerativeAI(temperature=0.7, model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY)

def pinecone_similarity_search(query):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host="https://rag-text-embed-test-nygbzf5.svc.aped-4627-b74a.pinecone.io")

    similar_data = index.search(
        namespace="__default__", 
        query={
            "inputs": {"text": query}, 
            "top_k": 3
        },
        fields=["text"]
    )

    retrieved_docs = []
    for hits_data in similar_data.get("result", {}).get("hits", []):
        retrieved_docs.append(hits_data.get("fields", {}).get("text"))

    return retrieved_docs

def generate_prompt(query):
    
    template = """
    Use the following context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context:
    {context}

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)

    retrieved_docs = pinecone_similarity_search(query)
    context = "\n\n---\n\n".join([text for text in retrieved_docs])
    final_prompt = prompt.format(context=context, question=query)
    return final_prompt



def chat_with_rag(query):
    """
    Manually performs the RAG process: retrieve, format, and invoke.
    """    
    final_prompt = generate_prompt(query)
    response = llm.invoke(final_prompt)
        
    return response.content