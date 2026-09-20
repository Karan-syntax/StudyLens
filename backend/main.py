# STEP 6: GENERATE THE FINAL ANSWER

# Goal:
# Retrieve relevant chunks from ChromaDB
# and give them to Gemini to generate the answer.

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma


# 1. Load environment variables

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
BASE_DIR = Path(__file__).resolve().parent


# 2. Create the SAME embedding model
# used when creating the database

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# 3. Load existing ChromaDB

vector_db = Chroma(
    persist_directory=str(BASE_DIR / "chroma_db"),
    embedding_function=embedding_model
)


# 4. Create retriever

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)


# 5. Create Gemini chat model

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=api_key
)


# 6. Ask the user for a question

question = input("Ask a question: ")


# 7. Retrieve relevant chunks

retrieved_docs = retriever.invoke(question)


# 8. Combine retrieved chunks into context

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)


# 9. Create prompt

prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I could not find the answer in the PDF."

Do not make up information.

Context:
{context}

User Question:
{question}
"""


# 10. Send prompt to Gemini

response = llm.invoke(prompt)


# 11. Display answer

print("\nAnswer:")
print(response.content[0]["text"])
