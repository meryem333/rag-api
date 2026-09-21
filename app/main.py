from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.ingestion import (
    extract_text_from_pdf,
    chunk_text,
    store_chunks,
    answer_question,
    ollama_client,
)

app = FastAPI(title="RAG Document Q&A API")


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Bienvenue sur le RAG Document Q&A API"}


@app.get("/health")
def health_check():
    """Vérifie que l'API tourne et qu'Ollama est joignable."""
    try:
        models = ollama_client.list()
        return {
            "status": "ok",
            "ollama_connected": True,
            "available_models": [m["model"] for m in models.get("models", [])]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "ollama_connected": False,
            "error": str(e)
        }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Reçoit un PDF, l'indexe (extraction + chunking + embeddings + stockage)."""
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)
    store_chunks(chunks, source_name=file.filename)

    return {
        "filename": file.filename,
        "chunks_indexed": len(chunks)
    }


@app.post("/ask")
def ask(question: Question):
    """Pose une question sur les documents indexés."""
    result = answer_question(question.question)
    return result