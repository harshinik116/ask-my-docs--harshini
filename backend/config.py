import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_PATH = "vector_store"
UPLOAD_FOLDER = "data/uploaded_docs"