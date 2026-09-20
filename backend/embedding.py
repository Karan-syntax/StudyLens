from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


# LOAD ENVIRONMENT VARIABLES

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# CREATE EMBEDDING MODEL

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key
)


# TEST EMBEDDING

text = "Java is a programming language."

vector = embeddings.embed_query(text)

print("Original text:")
print(text)

print("\nVector size:", len(vector))

print("\nFirst 10 vector values:")
print(vector[:10])