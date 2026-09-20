from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from pathlib import Path


# LOAD ENVIRONMENT VARIABLES

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
BASE_DIR = Path(__file__).resolve().parent


# STEP 1: LOAD PDF

loader = PyPDFLoader(str(BASE_DIR / "document_loaders" / "JAVA.pdf"))

docs = loader.load()

print("Number of pages:", len(docs))


# STEP 2: SPLIT PDF INTO CHUNKS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print("Number of chunks:", len(chunks))


# STEP 3: CREATE EMBEDDING MODEL

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# STEP 4: CREATE CHROMA VECTOR DATABASE

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(BASE_DIR / "chroma_db")
)

print("ChromaDB created successfully!")
