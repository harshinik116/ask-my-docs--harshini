# Ask My Docs

A RAG (Retrieval-Augmented Generation) app that lets you upload PDF documents and ask questions about them.

## Stack

- **Backend**: FastAPI + ChromaDB + SentenceTransformers + Groq (LLaMA3)
- **Frontend**: Plain HTML/CSS/JS (no build step needed)

## Setup

```bash
cd ask-my-docs

# Activate virtual environment
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
# Start the backend API (from the project root)
uvicorn backend.main:app --reload --port 8000

# Open the frontend
# Just open frontend/index.html in your browser
```

## Usage

1. Open `frontend/index.html` in a browser
2. Upload a PDF using the sidebar
3. Ask questions in the chat — answers include page citations

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question, get an answer |
