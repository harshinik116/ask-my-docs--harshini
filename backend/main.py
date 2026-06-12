import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import UPLOAD_FOLDER
from backend.ingestion import extract_text_from_pdf, chunk_text
from backend.retrieval import add_chunks_to_vector_db
from backend.bm25_store import add_to_corpus
from backend.rag_pipeline import answer_question

app = FastAPI(title="Ask My Docs RAG API")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = extract_text_from_pdf(file_path)
    chunks = chunk_text(pages)

    add_chunks_to_vector_db(chunks, file.filename)
    add_to_corpus(chunks, file.filename)

    return {
        "message": "Document uploaded and indexed successfully",
        "file": file.filename,
        "chunks_created": len(chunks)
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)

    return {
        "question": request.question,
        "answer": answer
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")